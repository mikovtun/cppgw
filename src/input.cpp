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
     std::nullopt, Sanitize::None,
     "calculation to run (GF2)"},
    {"HCORE", ValueKind::String,
     std::nullopt, Sanitize::None,
     "HDF5 path to the AO x AO core Hamiltonian (rank 2, square)"},
    {"MO_COEFF", ValueKind::String,
     std::nullopt, Sanitize::None,
     "HDF5 path to the AO x MO coefficients (rank 2)"},
    {"ERI3", ValueKind::String,
     std::nullopt, Sanitize::None,
     "HDF5 path to the RI density-fitting tensor, shape (ri, ao, ao) (rank 3)"},
    {"OVERLAP", ValueKind::String,
     std::nullopt, Sanitize::None,
     "HDF5 path to the AO x AO overlap matrix (rank 2, square)"},
    {"DENSITY_MATRIX", ValueKind::String,
     std::nullopt, Sanitize::None,
     "HDF5 path to the AO x AO spin-summed density matrix (rank 2, square)"},
    {"BETA", ValueKind::Double,
     std::nullopt, Sanitize::PositiveFiniteDouble,
     "positive inverse temperature for the Matsubara grid"},
    {"MATSUBARA_HALF_N", ValueKind::Int,
     std::nullopt, Sanitize::PositiveInt,
     "positive number of positive fermionic Matsubara frequencies"},
    {"MU", ValueKind::Double,
     std::nullopt, Sanitize::None,
     "initial chemical potential for the AO Matsubara initial guess"},
    {"ETA", ValueKind::Double,
     KeywordValue{1e-5}, Sanitize::PositiveFiniteDouble,
     "Green's-function offset/broadening (retained for the GF2 stages that use it)"},
  };
  return registry;
}

// ----- the calculation registry (per-calculation requirement sets) -----
// The single source of truth for WHICH keywords each calculation requires.
// The parser's calculation-name resolution and missing-keyword report are
// driven by this table; there is no per-calculation bool on the keywords.
// Row order is the order of the "known calculations" listing in errors.
const std::vector<CalculationDefinition>& calculation_registry() {
  static const std::vector<CalculationDefinition> registry = {
    {
      Calc::Gf2,
      "GF2",
      {"HCORE", "MO_COEFF", "ERI3", "OVERLAP", "DENSITY_MATRIX"},
      {"BETA", "MATSUBARA_HALF_N", "MU"},
      {"ETA"},
    },
  };
  return registry;
}

const CalculationDefinition& calculation_def(Calc calc) {
  for (const CalculationDefinition& cd : calculation_registry())
    if (cd.calc == calc)
      return cd;
  // A resolved input can never carry Calc::None and a missing row is an
  // author error: fail loudly, not silently.
  throw std::logic_error("calculation_def: no calculation definition for calc "
      + std::to_string(static_cast<int>(calc)));
}

// ----- the dataset definitions (the single place layouts are declared) -----
// Complete, authoritative table of dataset keywords: rank (= axes.size()),
// axis labels FASTEST-to-slowest, the binding kind for the repeated-label
// family, and the cross-dataset extent reference. The generic loader is
// driven entirely by these rows.
const std::vector<DatasetDefinition>& dataset_definitions() {
  static const std::vector<DatasetDefinition> definitions = {
    {
      "HCORE",
      {DatasetAxisSpec{TensorDimLabel("ao")}, DatasetAxisSpec{TensorDimLabel("ao")}},
      SymKind::Hermitian,
      std::nullopt,   // defines the 'ao' reference extent
      "AO x AO core Hamiltonian (rank 2, square, Hermitian)"},
    {
      "MO_COEFF",
      {DatasetAxisSpec{TensorDimLabel("mo")}, DatasetAxisSpec{TensorDimLabel("ao")}},
      std::nullopt,   // plain axes
      "HCORE",
      "MO x AO coefficients (rank 2, AO extent shared with HCORE)"},
    {
      "ERI3",
      {DatasetAxisSpec{TensorDimLabel("ao")}, DatasetAxisSpec{TensorDimLabel("ao")},
       DatasetAxisSpec{TensorDimLabel("ri")}},
      SymKind::Symmetric,
      "HCORE",
      "RI density-fitting tensor (rank 3, symmetric AO pair, AO extent shared with HCORE)"},
    {
      "OVERLAP",
      {DatasetAxisSpec{TensorDimLabel("ao")}, DatasetAxisSpec{TensorDimLabel("ao")}},
      SymKind::Hermitian,
      "HCORE",
      "AO x AO overlap matrix (rank 2, square, Hermitian)"},
    {
      "DENSITY_MATRIX",
      {DatasetAxisSpec{TensorDimLabel("ao")}, DatasetAxisSpec{TensorDimLabel("ao")}},
      SymKind::Hermitian,
      "HCORE",
      "AO x AO spin-summed density matrix (rank 2, square, Hermitian)"},
  };
  return definitions;
}

const DatasetDefinition& dataset_definition(const std::string& keyword) {
  const std::string lk = to_lower(keyword);
  for (const DatasetDefinition& d : dataset_definitions())
    if (lk == to_lower(d.keyword))
      return d;
  std::string all;
  for (const DatasetDefinition& d : dataset_definitions()) {
    if (!all.empty()) all += ", ";
    all += d.keyword;
  }
  throw std::invalid_argument("'" + keyword + "' is not a known dataset keyword "
      + "(known datasets: " + all + ")");
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

  // Resolve the calculation selector (definition-driven, case-insensitive).
  auto calc_it = values.find("CALCULATION");
  if (calc_it == values.end())
    throw std::invalid_argument("input must select a calculation: the 'CALCULATION' keyword is missing");
  const std::string calc_text = std::get<std::string>(calc_it->second.value);
  const std::string low = to_lower(calc_text);
  ResolvedInput resolved;   // calc stays Calc::None until a row matches
  for (const CalculationDefinition& cd : calculation_registry())
    if (low == to_lower(cd.name)) {
      resolved.calc = cd.calc;
      break;
    }
  if (resolved.calc == Calc::None)
    throw std::invalid_argument(line_what(first_line.at("CALCULATION"))
        + "unrecognized calculation '" + calc_text + "' (known calculations: "
        + [&] {
            std::string all;
            for (const CalculationDefinition& cd : calculation_registry()) {
              if (!all.empty()) all += ", ";
              all += cd.name;
            }
            return all;
          }()
        + ")");

  // Requirement check for the SELECTED calculation, reported all at once
  // (datasets list order, then parameters list order).
  {
    const CalculationDefinition& cd = calculation_def(resolved.calc);
    std::string missing;
    for (const std::string& kw : cd.required_datasets)
      if (values.count(kw) == 0)
        missing += (missing.empty() ? "" : ", ") + kw;
    for (const std::string& kw : cd.required_parameters)
      if (values.count(kw) == 0)
        missing += (missing.empty() ? "" : ", ") + kw;
    if (!missing.empty())
      throw std::invalid_argument(cd.name + " calculation requires missing keyword(s): " + missing);
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

// ----- generic, definition-driven dataset loading -----

Tensor<double, Executor::Host> InputCatalog::require_dataset(const std::string& keyword) const {
  // Unknown (or a parameter, not a dataset) keyword fails BEFORE any HDF5
  // access; the message keeps one source of truth (the table's order).
  const DatasetDefinition& def = dataset_definition(keyword);
  auto it = datasets_.find(def.keyword);   // cache keys are CANONICAL
  if (it != datasets_.end())
    return it->second;   // cached: no HDF5 re-touch
  return load_dataset(def, {});
}

Tensor<double, Executor::Host> InputCatalog::load_dataset(
    const DatasetDefinition& def,
    const std::set<std::string>& inflight) const {
  // Extent-reference cycle (a ref chain that comes back to itself) is an
  // author error; it is unreachable with the shipped table.
  if (inflight.count(def.keyword) != 0)
    throw std::logic_error("dataset extent-reference cycle detected at keyword "
        + def.keyword);

  std::set<std::string> child_inflight = inflight;
  child_inflight.insert(def.keyword);

  // Existence: `getDataSet` throws (typically a HighFive error derived from
  // std::runtime_error) when the path does not name an existing object;
  // rethrow with keyword context so the message identifies the request.
  const std::string path = resolved_.get_string(def.keyword);
  HighFive::DataSet ds;
  try {
    ds = file_->getDataSet(path);
  } catch (const std::exception& e) {
    throw std::invalid_argument("keyword " + def.keyword + ": HDF5 dataset '" + path
        + "' does not exist in the data file (" + e.what() + ")");
  }

  // Element type: float64 only.
  const HighFive::DataType dt = ds.getDataType();
  if (!(dt == HighFive::AtomicType<double>())) {
    throw std::invalid_argument("keyword " + def.keyword + ": HDF5 dataset '" + path
        + "' must have float64 elements (found " + dt.string()
        + "); only double-precision datasets are accepted for this calculation");
  }

  // Rank: the file's rank must equal the number of declared axes.
  const std::vector<size_t> dims = ds.getDimensions();   // slowest index first
  const size_t rank = dims.size();
  if (rank != def.axes.size()) {
    throw std::invalid_argument("keyword " + def.keyword + ": HDF5 dataset '" + path
        + "' must have rank " + std::to_string(def.axes.size())
        + " (found rank " + std::to_string(rank) + ")");
  }

  // Rendered size list in file order (slowest first), "R x C [x ...]".
  auto sizes_str = [&dims]() {
    std::string s;
    for (size_t f = 0; f < dims.size(); ++f) {
      if (f) s += " x ";
      s += std::to_string(dims[f]);
    }
    return s;
  };

  // R1: axes sharing a repeated label must have equal extents (the implicit
  // squareness rule, generalized to any repeated-label family).
  {
    std::string bad_family;
    for (const DatasetAxisSpec& axis : def.axes) {
      size_t count = 0;
      for (const DatasetAxisSpec& other : def.axes)
        if (other.label == axis.label) ++count;
      if (count < 2)
        continue;
      bool equal = true;
      size_t first_extent = 0;
      bool seen = false;
      for (size_t i = 0; i < def.axes.size() && equal; ++i) {
        if (def.axes[i].label != axis.label)
          continue;
        const size_t e = dims[rank - 1 - i];   // file dims are slowest-first
        if (!seen) { first_extent = e; seen = true; }
        else if (e != first_extent)
          equal = false;
      }
      if (!equal) { bad_family = label_to_string(axis.label); break; }
    }
    if (!bad_family.empty())
      throw std::invalid_argument("keyword " + def.keyword + ": HDF5 dataset '" + path
          + "' must have a square repeated '" + bad_family
          + "' axis pair (a square " + bad_family + " x " + bad_family
          + " pair; found " + sizes_str() + ")");
  }

  // R2: axes whose label also appears in `extent_ref`'s definition must match
  // the reference dataset's extent for that label (the reference is resolved
  // through the same loading path, cached or recursive); other labels are
  // free (e.g. 'mo', 'ri').
  if (def.extent_ref.has_value()) {
    const DatasetDefinition& ref = dataset_definition(*def.extent_ref);
    // Resolve the reference through the same loading path (already cached
    // if present; load_dataset caches its result either way).
    if (datasets_.count(*def.extent_ref) == 0) {
      const auto _ = load_dataset(ref, child_inflight);   // caches under its keyword
      (void)_;                                            // value unused: extent read from cache below
    }
    const Tensor<double, Executor::Host>& ref_t = datasets_.at(*def.extent_ref);
    for (size_t i = 0; i < def.axes.size(); ++i) {
      const TensorDimLabel& label = def.axes[i].label;
      size_t ref_extent = 0;
      bool shared = false;
      for (size_t j = 0; j < ref.axes.size(); ++j) {
        if (ref.axes[j].label == label) { ref_extent = ref_t.dims()[j].dim; shared = true; break; }
      }
      if (!shared)
        continue;
      const size_t e = dims[rank - 1 - i];
      if (e != ref_extent) {
        std::string lbl = label_to_string(label);
        for (char& c : lbl) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        throw std::invalid_argument("keyword " + def.keyword + ": HDF5 dataset '" + path
            + "' " + lbl + " extent does not match " + ref.keyword
            + " (found " + sizes_str() + ", expected " + std::to_string(ref_extent) + ")");
      }
    }
  }

  // Construction: extents fast->slow (file dims are slow->fast); ONE fresh
  // group (def.symmetry) shared by exactly the repeated-label family, plain
  // axes elsewhere. Symmetry is declared and constructed here; the dedicated
  // validation pass is the follow-up story (05.3).
  std::vector<size_t> extents(def.axes.size());
  for (size_t i = 0; i < def.axes.size(); ++i)
    extents[i] = dims[def.axes.size() - 1 - i];

  std::optional<TensorDimLabel> fam;
  for (const DatasetAxisSpec& axis : def.axes) {
    size_t count = 0;
    for (const DatasetAxisSpec& other : def.axes)
      if (other.label == axis.label) ++count;
    if (count >= 2) { fam = axis.label; break; }
  }

  SymGroup group;
  if (def.symmetry.has_value() && fam.has_value())
    switch (*def.symmetry) {
      case SymKind::Symmetric:     group = Symmetric(); break;
      case SymKind::Hermitian:     group = Hermitian(); break;
      case SymKind::Antisymmetric: group = Antisymmetric(); break;
    }

  std::vector<TensorDim> tds;
  tds.reserve(def.axes.size());
  for (size_t i = 0; i < def.axes.size(); ++i) {
    const bool in_family = def.symmetry.has_value() && fam.has_value()
        && def.axes[i].label == *fam;
    TensorDim td(def.axes[i].label, extents[i]);
    if (in_family) td.symmetry = group;
    tds.push_back(std::move(td));
  }

  Tensor<double, Executor::Host> t(std::move(tds));
  ds.read_raw(t.data());   // C-order maps 1:1 to the fast->slow Tensor layout
  datasets_[def.keyword] = t;   // cache under the CANONICAL spelling
  return t;
}

} // namespace cppgw
