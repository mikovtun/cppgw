#pragma once
// ============================================================================
//  Input framework (Story 05)
// ----------------------------------------------------------------------------
//  `InputCatalog` is the central input/data service (not a global singleton).
//  It owns the input vocabulary, defaults, value sanitization, dataset
//  semantics, and calculation requirements:
//
//    1. Keyword definitions (the registry below) are the single source of
//       truth for every known keyword: its value kind, optional default,
//       sanitizer, GF2 requirement, and documentation.
//    2. `ResolvedInput` is one parsed input file: canonical keyword -> typed
//       value, plus whether each value was user-supplied or defaulted.
//    3. `InputCatalog` resolves a ResolvedInput against one (single) opened
//       HDF5 file and serves the semantic data a calculation requires.
//       `require_gf2()` returns a fully loaded `Gf2Input`; calculations never
//       use raw keyword strings or open HDF5 files themselves.
//
//  `parse_input(text)` is the pure text parser (no files); it is shared by
//  both the catalog and the parser unit tests.
//
//  Error policy: structural input problems (unknown/duplicate keyword,
//  malformed value, missing CALCULATION, missing required keyword, invalid
//  value range) throw std::invalid_argument naming the keyword and line
//  number. Dataset problems throw std::invalid_argument naming the keyword,
//  dataset path, and offending axis; file-open problems propagate the
//  underlying exception.
// ============================================================================

#include "gf2.hpp"

#include <highfive/highfive.hpp>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace cppgw {

// ----- Calculation selection -----

enum class Calc { Gf2 };

inline const char* calc_name(Calc calc) {
  switch (calc) {
    case Calc::Gf2: return "GF2";
  }
  return "?";
}

// ----- Keyword value model -----

enum class ValueKind { String, Double, Int };

// A typed keyword value: string values (dataset paths, calculation names),
// double scalars, and strict integers (for future grid-size parameters).
using KeywordValue = std::variant<std::string, double, long long>;

// Value sanitization performed by the catalog AFTER parsing and BEFORE any
// calculation sees the value. Parsed doubles are already finite (the parser
// rejects NaN/Inf); PositiveFiniteDouble additionally enforces x > 0.
enum class Sanitize {
  None,
  PositiveFiniteDouble,
  PositiveInt,
};

// One registry row = one known keyword. New keywords are added HERE and
// nowhere else: parser, catalog, requirements, and GF2 stubs pick them up
// through this table.
struct KeywordDefinition {
  std::string keyword;                    // canonical (upper-case) spelling
  ValueKind   kind;                       // String (path/name) | Double | Int
  std::optional<KeywordValue> default_value; // present -> optional keyword
  bool        required_for_gf2;           // member of the GF2 requirement set
  Sanitize    sanitize;
  const char* doc;
};

// THE single registry. Order here is the order keywords are listed in errors
// (e.g. the "missing required keywords" report).
const std::vector<KeywordDefinition>& keyword_registry();

// A resolved (canonical) keyword value plus its provenance.
struct ResolvedValue {
  KeywordValue value;
  bool supplied = false;   // true  - written in the input file
                            // false - filled from the registry default
};

// One validated, resolved input file: the selected calculation plus the
// canonical keyword -> value table (defaults applied, sanitizers applied).
struct ResolvedInput {
  Calc calc = Calc::Gf2;
  // canonical keyword -> resolved value (keys are upper-case spellings)
  std::map<std::string, ResolvedValue> values;

  [[nodiscard]] bool supplied(const std::string& canonical) const;
  // Typed getters. Throw std::invalid_argument if the keyword is not
  // present in the resolved input.
  [[nodiscard]] const std::string& get_string(const std::string& canonical) const;
  [[nodiscard]] double             get_double(const std::string& canonical) const;
  [[nodiscard]] long long          get_int   (const std::string& canonical) const;

private:
  // Shared lookup: finds `canonical` and returns the ResolvedValue or throws
  // std::invalid_argument naming the keyword. (Defined in src/input.cpp.)
  const ResolvedValue& find(const std::string& canonical) const;
};

// Parse input-file text into a ResolvedInput. Pure (no file or HDF5 access),
// so it is directly unit-testable. Throws std::invalid_argument with an
// actionable message (keyword + line number where known) on any problem.
ResolvedInput parse_input(const std::string& text);

// The catalog: a parsed input file bound to ONE opened HDF5 data file.
class InputCatalog {
public:
  // Parse the input file and open the HDF5 data file (read-only).
  // The HDF5 file is opened exactly ONCE per catalog; typed requirement
  // requests (require_gf2) share this connection.
  static InputCatalog from_file(const std::string& input_path,
                                const std::string& hdf5_path);

  // Same as from_file but the input text is supplied inline. Used by the
  // in-process tests so they can drive the real catalog pipeline without
  // writing input files to disk.
  static InputCatalog from_text(const std::string& text,
                                const std::string& hdf5_path);

  const ResolvedInput& resolved()    const { return resolved_; }
  const std::string&   hdf5_path()   const { return hdf5_path_; }
  HighFive::File&      file()        { return *file_; }

  // Typed GF2 requirement request: verifies the selected calculation, loads
  // the HCORE / MO_COEFF / ERI3 datasets through the catalog's single file
  // connection, and returns the fully constructed, labelled, symmetry-bearing
  // tensors plus the sanitized ETA. (Defined in src/gf2.cpp.)
  [[nodiscard]] Gf2Input require_gf2() const;

private:
  InputCatalog(ResolvedInput resolved, std::string hdf5_path);
  ResolvedInput                 resolved_;
  std::string                   hdf5_path_;
  std::shared_ptr<HighFive::File> file_;
};

} // namespace cppgw
