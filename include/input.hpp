#pragma once
// ============================================================================
//  Input framework (Story 05)
// ----------------------------------------------------------------------------
//  `InputCatalog` is the central input/data service (not a global singleton).
//  It owns the input vocabulary, defaults, value sanitization, dataset
//  semantics, and calculation requirements:
//
//    1. Calculations (the definition table below) are the single source of
//       truth for each supported calculation's required/optional keyword
//       sets; the parser's calculation-name resolution and missing-keyword
//       check are driven by this table (no per-calculation booleans).
//    2. Dataset definitions are the single place where a dataset keyword's
//       semantics are declared: its rank, axis labels (fastest-to-slowest),
//       the symmetry binding kind for its repeated-label family, and its
//       cross-dataset extent reference. `InputCatalog::require_dataset`
//       is the one, definition-driven loading path (rank/type/extent
//       validation upon entry), cached per keyword.
//    3. Keyword definitions (the registry below) are the single source of
//       truth for every known keyword: its value kind, optional default,
//       sanitizer, and documentation.
//    4. `ResolvedInput` is one parsed input file: canonical keyword -> typed
//       value, plus whether each value was user-supplied or defaulted.
//    5. `InputCatalog` resolves a ResolvedInput against one (single) opened
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
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace cppgw {

// ----- Calculation selection -----

enum class Calc { None, Gf2 };   // None: unselected sentinel only (the parser
                                  // never lets a resolved input carry None)

// One row per supported calculation: its required/optional keyword sets.
// Keyword strings are CANONICAL (upper-case) spellings; the order in each
// list is the order of missing-keyword reporting.
struct CalculationDefinition {
  Calc calc;
  std::string name;                                // "GF2"
  std::vector<std::string> required_datasets;      // e.g. { "HCORE", ... }
  std::vector<std::string> required_parameters;    // e.g. { "BETA", ... }
  std::vector<std::string> optional_keywords;      // defaulted keywords, e.g. { "ETA" }
};

// THE calculation registry: the single source of truth for the set of known
// calculations and their requirement sets (defined in src/input.cpp).
const std::vector<CalculationDefinition>& calculation_registry();

// The requirement-set row of one calculation. Calc::None (or any missing
// row) is an author error: throws std::logic_error.
[[nodiscard]] const CalculationDefinition& calculation_def(Calc calc);

// Derived from the registry (single source of truth for calculation names):
// "GF2" for Calc::Gf2, "?" for Calc::None or any unknown value.
inline const char* calc_name(Calc calc) {
  if (calc == Calc::None)
    return "?";
  try { return calculation_def(calc).name.c_str(); }
  catch (...) { return "?"; }
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
// nowhere else: parser, catalog, and requirements pick them up through this
// table. (Which CALCULATION requires a keyword is the calculation registry's
// job, not the keyword's.)
struct KeywordDefinition {
  std::string keyword;                    // canonical (upper-case) spelling
  ValueKind   kind;                       // String (path/name) | Double | Int
  std::optional<KeywordValue> default_value; // present -> optional keyword
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
  // Selected by the parser through the calculation registry; `None` only
  // appears on a default-constructed value that never escapes the parser.
  Calc calc = Calc::None;
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

// ----- Dataset definitions (the single place dataset layouts are declared) -----

// One declared axis of a dataset: its label in the Tensor frontend.
struct DatasetAxisSpec  { TensorDimLabel label; };      // e.g. "ao", "mo", "ri"

// One row per dataset keyword: its expected rank (axes.size()), its axis
// labels FASTEST-to-slowest, the symmetry binding kind for the repeated-label
// family (a fresh Hermitian()/Symmetric() group; nullopt = plain axes), and
// an optional keyword whose like-labelled axes fix this dataset's extents.
// The generic loader (InputCatalog::require_dataset) is driven entirely by
// these rows; adding or reshaping a dataset is a data-only edit.
struct DatasetDefinition {
  std::string keyword;                 // canonical, e.g. "DENSITY_MATRIX"
  std::vector<DatasetAxisSpec> axes;   // FASTEST-to-slowest; rank == axes.size()
  std::optional<SymKind> symmetry;     // binding kind for the repeated-label family
  std::optional<std::string> extent_ref;  // keyword whose like-labelled axes fix this one's extents
  const char* doc;
};

// THE dataset-definition table (defined in src/input.cpp); order is the
// order of the "known datasets" listing in error messages.
const std::vector<DatasetDefinition>& dataset_definitions();

// Case-insensitive lookup of one dataset definition. Unknown (e.g. a
// parameter) keyword -> std::invalid_argument naming it and the known list.
[[nodiscard]] const DatasetDefinition& dataset_definition(const std::string& keyword);

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

  // String-keyed dataset request (canonical keyword, case-insensitive).
  // Loads + validates per the dataset definition (existence, float64
  // elements, rank, and the R1/R2 extent rules), caches in the catalog, and
  // returns the fully constructed, labelled, symmetry-bearing tensor by
  // value. Repeated calls return the (cached) same tensor; different
  // keywords never share storage. Single-threaded: no synchronization.
  [[nodiscard]] Tensor<double, Executor::Host>
  require_dataset(const std::string& keyword) const;

  // Typed GF2 requirement request: verifies the selected calculation, loads
  // each of the GF2 calculation row's required datasets THROUGH
  // require_dataset (the single loading path), and returns them as a
  // fully-loaded Gf2Input plus the sanitized scalars. (Defined in
  // src/gf2.cpp.)
  [[nodiscard]] Gf2Input require_gf2() const;

private:
  InputCatalog(ResolvedInput resolved, std::string hdf5_path);
  ResolvedInput                 resolved_;
  std::string                   hdf5_path_;
  std::shared_ptr<HighFive::File> file_;
  // Load-once cache of dataset tensors keyed by CANONICAL keyword.
  mutable std::map<std::string, Tensor<double, Executor::Host>> datasets_;

  // Definition-driven load of one dataset row (no cache lookup; `inflight`
  // guards against extent-reference cycles, which would be an author
  // error). Shared by require_dataset and the recursive extent-reference
  // resolution.
  [[nodiscard]] Tensor<double, Executor::Host>
  load_dataset(const DatasetDefinition& def,
               const std::set<std::string>& inflight) const;
};

} // namespace cppgw
