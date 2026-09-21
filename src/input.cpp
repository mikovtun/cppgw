// ============================================================================
//  Input framework (Story 05): registry, parser, sanitization, catalog
// ============================================================================

#include "input.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cppgw {

namespace {

// ----- small text helpers -----

std::string to_lower(std::string s) {
  for (char& c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string trim_copy(std::string_view v) {
  const char* ws = " \t\v\f\r\n";
  size_t b = v.find_first_not_of(ws);
  if (b == std::string_view::npos)
    return std::string();
  size_t e = v.find_last_not_of(ws);
  return std::string(v.substr(b, e - b + 1));
}

// KEYWORD = [A-Za-z][A-Za-z0-9_]*   (case folded before the lookup)
bool valid_keyword_token(std::string_view k) {
  if (k.empty() || !std::isalpha(static_cast<unsigned char>(k.front())))
    return false;
  for (char c : k)
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      return false;
  return true;
}

std::string line_what(int line) {
  return "input line " + std::to_string(line) + ": ";
}

// Double parsing with Fortran `D` exponents (`1D-5`, `1.2d+3`, `1D10`).
// Rejects partial tokens (full consumption required) and non-finite values.
double parse_double_token(std::string token, const std::string& kw, int line) {
  for (size_t i = 0; i < token.size(); ++i) {
    const unsigned char prev = i > 0 ? static_cast<unsigned char>(token[i - 1]) : 0;
    if ((token[i] == 'D' || token[i] == 'd')
        && (std::isdigit(prev) || token[i - 1] == '.'))
      token[i] = (token[i] == 'D') ? 'E' : 'e';
  }
  const char* begin = token.c_str();
  char* end = nullptr;
  const double x = std::strtod(begin, &end);
  if (end == begin || *end != '\0')
    throw std::invalid_argument(line_what(line)
        + "cannot parse '" + token + "' as a double value for keyword " + kw
        + " (expected e.g. 1e-5 or 1D-5)");
  if (!std::isfinite(x))
    throw std::invalid_argument(line_what(line)
        + "keyword " + kw + ": value must be finite (got infinite or NaN)");
  return x;
}

// Strict integer parsing (future grid-size parameters such as
// MATSUBARA_HALF_N): decimal digits only. Rejects signs, fractions,
// exponents, and anything after the digits; also guards the conversion.
long long parse_int_token(std::string token, const std::string& kw, int line) {
  if (token.empty())
    throw std::invalid_argument(line_what(line)
        + "cannot parse '' as an integer value for keyword " + kw);
  for (char c : token)
    if (!std::isdigit(static_cast<unsigned char>(c)))
      throw std::invalid_argument(line_what(line)
          + "cannot parse '" + token + "' as an integer value for keyword " + kw
          + " (expected decimal digits only)");
  errno = 0;
  char* end = nullptr;
  const long long x = std::strtoll(token.c_str(), &end, 10);
  if (errno == ERANGE || end == token.c_str() || *end != '\0')
    throw std::invalid_argument(line_what(line)
        + "cannot parse '" + token + "' as an integer value for keyword " + kw);
  return x;
}

const KeywordDefinition* lookup_registry(const std::string& kw) {
  const std::string lk = to_lower(kw);
  for (const auto& def : keyword_registry())
    if (lk == to_lower(def.keyword))
      return &def;
  return nullptr;
}

// The sanitizers are data-driven through the registry (KeywordDefinition::
// sanitize); this is the single place where value *ranges* are enforced,
// after parsing and before any calculation can observe the value.
void sanitize_value(const KeywordDefinition& def, const ResolvedValue& rv, int line) {
  switch (def.sanitize) {
    case Sanitize::None:
      return;
    case Sanitize::PositiveFiniteDouble: {
      const double x = std::get<double>(rv.value);
      if (!std::isfinite(x) || x <= 0.0)
        throw std::invalid_argument(line_what(line)
            + "keyword " + def.keyword + ": value must be finite and strictly positive"
            + " (got " + std::to_string(x) + ")");
      return;
    }
    case Sanitize::PositiveInt: {
      const long long x = std::get<long long>(rv.value);
      if (x <= 0)
        throw std::invalid_argument(line_what(line)
            + "keyword " + def.keyword + ": value must be a positive integer"
            + " (got " + std::to_string(x) + ")");
      return;
    }
  }
}

struct Assignment {
  std::string token;   // keyword as written
  std::string value;   // trimmed value text
  int         line = 0;
};

// Split `text` into non-blank, non-comment assignment candidates. Comments:
// lines whose FIRST non-whitespace character starts a `#` or `//` comment
// (whole-line comments only). Extension of the input file is irrelevant.
std::vector<Assignment> collect_assignments(const std::string& text) {
  std::vector<Assignment> out;

  size_t start = 0;
  int lineno = 0;
  for (;;) {
    const size_t nl = text.find('\n', start);
    std::string line(text.substr(start, nl == std::string::npos ? std::string::npos : nl - start));
    ++lineno;
    if (!line.empty() && line.back() == '\r')
      line.pop_back();

    const std::string trimmed = trim_copy(line);
    if (!trimmed.empty()
        && !(trimmed.front() == '#'
             || (trimmed.size() >= 2 && trimmed[0] == '/' && trimmed[1] == '/'))) {
      const size_t eq = trimmed.find('=');
      if (eq == std::string::npos)
        throw std::invalid_argument(line_what(lineno)
            + "expected an assignment `KEYWORD = VALUE` but found '"
            + trimmed + "'");
      Assignment a;
      a.line = lineno;
      a.token = trim_copy(trimmed.substr(0, eq));
      a.value = trim_copy(trimmed.substr(eq + 1));
      out.push_back(std::move(a));
    }
    if (nl == std::string::npos)
      break;
    start = nl + 1;
  }
  return out;
}

} // namespace

// ----- the registry (single source of truth) -----

const std::vector<KeywordDefinition>& keyword_registry() {
  static const std::vector<KeywordDefinition> registry = {
    {"CALCULATION", ValueKind::String,
     std::nullopt, /*required_for_gf2=*/false, Sanitize::None,
     "calculation to run (GF2)"},
    {"HCORE", ValueKind::String,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::None,
     "HDF5 path to the AO x AO core Hamiltonian (rank 2, square)"},
    {"MO_COEFF", ValueKind::String,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::None,
     "HDF5 path to the AO x MO coefficients (rank 2)"},
    {"ERI3", ValueKind::String,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::None,
     "HDF5 path to the RI density-fitting tensor, shape (ri, ao, ao) (rank 3)"},
    {"OVERLAP", ValueKind::String,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::None,
     "HDF5 path to the AO x AO overlap matrix (rank 2, square)"},
    {"DENSITY_MATRIX", ValueKind::String,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::None,
     "HDF5 path to the AO x AO spin-summed density matrix (rank 2, square)"},
    {"BETA", ValueKind::Double,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::PositiveFiniteDouble,
     "positive inverse temperature for the Matsubara grid"},
    {"MATSUBARA_HALF_N", ValueKind::Int,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::PositiveInt,
     "positive number of positive fermionic Matsubara frequencies"},
    {"MU", ValueKind::Double,
     std::nullopt, /*required_for_gf2=*/true, Sanitize::None,
     "initial chemical potential for the AO Matsubara initial guess"},
    {"ETA", ValueKind::Double,
     KeywordValue{1e-5}, /*required_for_gf2=*/false, Sanitize::PositiveFiniteDouble,
     "Green's-function offset/broadening (retained for the GF2 stages that use it)"},
  };
  return registry;
}

// ----- ResolvedInput accessors -----

const ResolvedValue& ResolvedInput::find(const std::string& canonical) const {
  auto it = values.find(canonical);
  if (it == values.end())
    throw std::invalid_argument("keyword " + canonical + " is not set in the resolved input");
  return it->second;
}

bool ResolvedInput::supplied(const std::string& canonical) const {
  auto it = values.find(canonical);
  return it != values.end() && it->second.supplied;
}

const std::string& ResolvedInput::get_string(const std::string& canonical) const {
  return std::get<std::string>(find(canonical).value);
}

double ResolvedInput::get_double(const std::string& canonical) const {
  return std::get<double>(find(canonical).value);
}

long long ResolvedInput::get_int(const std::string& canonical) const {
  return std::get<long long>(find(canonical).value);
}

// ----- parser -----

ResolvedInput parse_input(const std::string& text) {
  const auto& registry = keyword_registry();
  const std::vector<Assignment> assignments = collect_assignments(text);

  // canonical keyword -> resolved value (+ first line for diagnostics)
  std::map<std::string, ResolvedValue> values;
  std::map<std::string, int> first_line;

  for (const Assignment& a : assignments) {
    const KeywordDefinition* def = lookup_registry(a.token);
    if (def == nullptr)
      throw std::invalid_argument(line_what(a.line)
          + "'" + a.token + "' is not a known keyword (known keywords: "
          + [&] {
              std::string all;
              for (const auto& k : registry) {
                if (!all.empty()) all += ", ";
                all += k.keyword;
              }
              return all;
            }()
          + ")");
    if (values.count(def->keyword)) {
      throw std::invalid_argument("keyword " + def->keyword
          + " is given more than once (first on line " + std::to_string(first_line.at(def->keyword))
          + ", again on line " + std::to_string(a.line) + "); duplicates are ambiguous and rejected"
          + " rather than last-wins");
    }

    ResolvedValue rv;
    rv.supplied = true;
    switch (def->kind) {
      case ValueKind::String:
        if (a.value.empty())
          throw std::invalid_argument(line_what(a.line)
              + "keyword " + def->keyword + " requires a non-empty value"
              + " (got none after the '=')");
        rv.value = a.value;
        break;
      case ValueKind::Double:
        rv.value = parse_double_token(a.value, def->keyword, a.line);
        break;
      case ValueKind::Int:
        rv.value = parse_int_token(a.value, def->keyword, a.line);
        break;
    }
    values.emplace(def->keyword, std::move(rv));
    first_line[def->keyword] = a.line;
  }

  // Resolve the calculation selector.
  auto calc_it = values.find("CALCULATION");
  if (calc_it == values.end())
    throw std::invalid_argument("input must select a calculation: the 'CALCULATION' keyword is missing");
  const std::string calc_text = std::get<std::string>(calc_it->second.value);
  ResolvedInput resolved;
  const std::string low = to_lower(calc_text);
  if (low == "gf2")
    resolved.calc = Calc::Gf2;
  else
    throw std::invalid_argument(line_what(first_line.at("CALCULATION"))
        + "unrecognized calculation '" + calc_text + "' (known calculations: GF2)");

  // Requirement check for the SELECTED calculation, reported all at once.
  if (resolved.calc == Calc::Gf2) {
    std::string missing;
    for (const auto& def : registry)
      if (def.required_for_gf2 && values.count(def.keyword) == 0)
        missing += (missing.empty() ? "" : ", ") + def.keyword;
    if (!missing.empty())
      throw std::invalid_argument("GF2 calculation requires missing keyword(s): " + missing);
  }

  // Default application (registry-driven, not special-cased here) and
  // sanitization of every value the calculation will see.
  for (const auto& def : registry) {
    auto it = values.find(def.keyword);
    if (it == values.end()) {
      if (def.default_value.has_value())
        values.emplace(def.keyword, ResolvedValue{*def.default_value, /*supplied=*/false});
      continue;
    }
    const int line = first_line.at(def.keyword);
    sanitize_value(def, it->second, line);
  }

  resolved.values = std::move(values);
  return resolved;
}

// ----- catalog -----

namespace {
std::shared_ptr<HighFive::File> open_hdf5(const std::string& path) {
  return std::make_shared<HighFive::File>(path, HighFive::File::ReadOnly);
}
} // namespace

InputCatalog::InputCatalog(ResolvedInput resolved, std::string hdf5_path)
    : resolved_(std::move(resolved)), hdf5_path_(std::move(hdf5_path)) {}

InputCatalog InputCatalog::from_file(const std::string& input_path,
                                     const std::string& hdf5_path) {
  std::ifstream in(input_path);
  if (!in.is_open())
    throw std::invalid_argument("cannot open input file '" + input_path + "'");
  std::ostringstream buf;
  buf << in.rdbuf();
  if (in.bad())
    throw std::runtime_error("I/O error while reading input file '" + input_path + "'");
  return from_text(buf.str(), hdf5_path);
}

InputCatalog InputCatalog::from_text(const std::string& text,
                                     const std::string& hdf5_path) {
  InputCatalog catalog(parse_input(text), hdf5_path);
  try {
    catalog.file_ = open_hdf5(hdf5_path);
  } catch (const std::exception& e) {
    throw std::invalid_argument("cannot open HDF5 data file '" + hdf5_path
        + "' for reading (" + e.what() + ")");
  }
  return catalog;
}

} // namespace cppgw
