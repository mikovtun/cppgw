# Story 05 — Input parsing & data loading (GF2)

> **Dependencies:** Story 05 **depends on Story 06** (*Tensor index symmetry & the Tensor backend seam*).
> Story 06 introduces `include/symmetry.hpp` (`SymKind`, `SymmetryGroup`, `SymGroup`, the `Symmetric()`/
> `Hermitian()`/`Antisymmetric()` factories), relaxes the `TensorDim` label invariant so a *pair* of
> identical axes may share one label when bound by a shared `SymmetryGroup`, and defines the
> `TensorBackend` op contract. This story uses that API to **declare the physical symmetries of the
> loaded tensors** (see *Data loading (GF2) & dimension labels* below: `hcore` → Hermitian, `eri3` → symmetric
> on its two AO axes, `mo_coeff` → no symmetries). Story 06 must land first; the label/symmetry validation
> in `Tensor` construction is what makes the shared-`ao` axes below constructible.

This software needs to read in data provided by other programs in the form of an HDF5 file.
For the calculations this program is supposed to do (GF2 and GW), there are 3 key tensors needed:

1. Core Hamiltonian in the AO basis: **N_AO × N_AO**
2. MO Coefficients: **N_AO × N_MO**
3. 3-index density-fitted (or resolution-of-identity) interaction tensor: **N_RI × N_AO × N_AO**

The example HDF5 file I've provided (`rhf_df.h5`, currently in `build/`) has all of these
tensors in the root directory with the names `hcore`, `mo_coeff`, and `eri3` respectively, and a
few other tensors (`density_matrix`, `overlap`, `mo_energy`, `mo_occ`, `atom_charges`,
`atom_coords`) which this story does not use. Their shapes/dtypes (all `float64`) are:

| name       | shape          | physical meaning |
|------------|----------------|------------------|
| `hcore`    | `(24, 24)`     | N_AO × N_AO      |
| `mo_coeff` | `(24, 24)`     | N_AO × N_MO      |
| `eri3`     | `(116, 24, 24)`| N_RI × N_AO × N_AO|

The input file should specify the paths in the HDF5 file of where these objects live in a
variable-assignment input style:

```text
ERI3 = eri3
HCORE = hcore
MO_COEFF = mo_coeff
eta = 0.00001
```

should be a valid input file (with a `CALCULATION` line added, see below).

## Requirements (from the original brief)

- The keywords `ERI3`, `HCORE`, and `MO_COEFF` must be parsed **case-insensitively** (the value on
  the right-hand side is an HDF5 path).
- The HDF5 **paths** they specify are **case-sensitive** and must correspond to **actual dataset paths**
  in the HDF5 data file. A path that does not name an existing, readable dataset with `float64` elements is
  an error.
- The `eta` keyword is a **double-valued** input. `0.00001` must parse as a double, as must
  `1e-5`, `1E-5`, `1D-5`, `1d-5` (Fortran-style `D` exponents are required). `eta` is the offset used in
  the Green's function for the calculation, but this story's scope is limited to **data ingestion and parsing**.
- In general, keywords will: specify data paths (in H5 files), and select calculations and their parameters.
- We are now preparing for **GF2** calculations (in the future: **GW** and several others). Each
  calculation may require a different set of data structures / parameters.

## Decisions (agreed)

- **Calculation selection:** the input file must contain a `CALCULATION` keyword whose value selects the
  calculation (e.g. `CALCULATION = GF2`). The calculation name is **case-insensitive**. A missing
  `CALCULATION` or an unrecognized calculation name is a **hard error** (the program must know what to run).
- **Unknown keywords are a hard error**, whether or not they would be mandatory for the selected calculation.
  A keyword not present in the registry is rejected even if it is not required, so a typo
  (e.g. `HERI3 = eri3`) fails loudly instead of being silently ignored. Recognition is global (per-registry);
  *requirement* is per-calculation.
- **`eta` is NOT required** for GF2 and defaults to **`1e-5`**.
- **Central parameter registry / data structure:** all known keywords, their kinds, their defaults, and
  per-calculation requirement are defined in **one place**, so adding a new parameter (or changing a default)
  is a single, local edit rather than a hunt across the codebase.
- **Tensor dimensions are labelled by physical meaning** (the point of the Tensor labelling system).
  See the Dimension labels section. (Story 06 provides the no-op symmetry transpose
  `T.transpose(i,j)` for symmetry-bound pairs; Story 06.1 provides `permute_axes` / pairwise
  `transpose` for non-identical axes, which this story does not use during loading.)
- **Both AO axes of `hcore`/`eri3` share the label `ao`, bound by ONE shared `SymmetryGroup`** (Story 06
  label rule 2: a repeated label is legal iff every occurrence is bound to the same non-null group). The
  interchangeable axes are physically identical, so one label names the family; `label_index("ao")` is a
  valid representative. `mo` is the MO axis, `ri` the RI axis (both plain, single occurrences).
- **`CALCULATION` is required to be present.** The program must know which calculation to run, so an input
  file with no `CALCULATION` line is a hard error (not defaulted to GF2).
- **Duplicate keywords are a hard error** (the same keyword on two lines is ambiguous), *not* last-wins.
- **Comments:** both `#` and `//` at the start of a line (first non-whitespace char) start a whole-line comment.
- **Physical symmetries are DECLARED on the loaded tensors** (using Story 06's API; see *Data loading (GF2)
  & dimension labels*):
  `hcore`'s two AO axes are bound by a shared `Hermitian()` group, `eri3`'s two AO axes by a shared
  `Symmetric()` group, `mo_coeff` carries no symmetries. Symmetry is **advisory** (Story 06, decision Q3):
  it is metadata on the `Tensor` that names the physics and lets no-op work be skipped. The data is still
  stored raw and dense exactly as it appears in the file — no storage compression, no symmetry-exploiting
  contractions, and **no consistency checks** (e.g. checking `hcore == hcore^†`) are performed at load
  time; those belong to the GF2 numerics story.

## Context for implementation

- **Data type is `double` (`float64`).** All three tensors in the example file are `H5T_NATIVE_DOUBLE`.
  `HighFive` is already a FetchContent dependency (Story 1) and linked into the `cppgw` target; `main.cpp`
  already opens a `HighFive::File` for the data-file health check.
- **Storage layout matches, so a copy is enough.** HDF5 stores datasets C-order (last index fastest);
  `Tensor::data()` is last-dim-fast. In the label order chosen above, a dataset's contiguous memory maps 1:1
  onto the tensor's flat storage, so loading is a direct `dataset.read(tensor.data(), dims)` — no transpose,
  no per-element loop.
- **Dimension sizes are data-driven, not hardcoded.** `N_AO`, `N_MO`, `N_RI` come from the dataset shapes at
  runtime (the `24`/`116` in the brief are just the example file's values). `N_AO` must agree across `hcore`,
  `mo_coeff`, and both AO axes of `eri3`; `N_MO` is the second axis of `mo_coeff`; `eri3`'s RI axis size is independent.
- **Keep `build/rhf_df.h5`** (Story 2 placed it there for manual runs) — do NOT delete it. The `gf2.in` input
  fixture for the CLI smoke check can live alongside the other inputs (e.g. `build/`).
- **Testing is in-process** (Story 2's `-t`/`-test` suite) plus the CLI acceptance commands; there is no `ctest`
  harness, so the self-contained HighFive "write-then-load" integration tests deliberately avoid depending on
  any particular `.h5` file path.

---

## Goals

1. **InputCatalog** that parses, resolves defaults, tracks provenance, sanitizes values, and validates
   the selected calculation's requirements.
2. **GF2 calculation input** that:
   - requests the semantic requirement set (`HCORE`, `ERI3`, `MO_COEFF`, `ETA`) from the catalog,
   - verifies each dataset keyword names an existing, readable `float64` dataset in the HDF5 file,
   - loads each into a `Tensor<double, Executor::Host>` with physically meaningful dimension labels, and
   - **declares the physical index symmetries** on the loaded tensors via Story 06's `SymmetryGroup` API
     (`include/symmetry.hpp`): `hcore` → its two `ao` axes bound by a shared `Hermitian()` group;
     `eri3` → its two `ao` axes bound by a shared `Symmetric()` group; `mo_coeff` → plain (no symmetries), and
   - records `eta` (default `1e-5`) alongside the tensors.
   - No actual GF2 numerics in this story — only ingestion + wiring.

Out of scope: any GF2/GW numerics, transposing/permuted reads (labels only), reading tensors other than the
three above, and any device-side loading. (Symmetry *data-validity* checks — “is the loaded `hcore` actually
Hermitian?” — are out of scope; symmetry here is advisory metadata only, per Story 06 decision Q3.)

---

## Proposed input framework

The framework is intentionally a request-oriented catalog rather than a global
singleton. It centralizes input semantics while keeping calculation code
explicit about what it needs.

Story 05 introduces an **InputCatalog**. This is the name of the central
input/data service; it is not a global singleton or a vague "central
authority". The catalog is the single owner of input vocabulary, defaults,
validation rules, dataset semantics, and calculation requirements.

The framework has three deliberately small layers:

1. **Definitions** are one registry of known keywords and semantic data
objects. A definition says how a value is parsed, whether it has a default,
how it is sanitized, and—for a dataset—what Tensor object it means.
2. **ResolvedInput** is one validated input file. It retains whether each value
was supplied by the user or came from a default and exposes typed values.
3. **InputCatalog** resolves a `ResolvedInput` against one HDF5 file and
provides the semantic data required by a selected calculation. It opens the
file once and owns all dataset lookup/validation. Calculations never use raw
keyword strings, duplicate HDF5 paths, or construct input tensors themselves.

The intended usage is:

```cpp
InputCatalog catalog = InputCatalog::from_file(input_path, hdf5_path);
Gf2Input gf2 = catalog.require_gf2();

// `gf2.hcore` is already a Hermitian AO matrix Tensor;
// `gf2.eta` is sanitized and `gf2.eta_was_supplied` reports provenance.
run_gf2(gf2);
```

`require_gf2()` is a typed calculation request, not a second loader. It asks
the catalog for the GF2 requirement set. The catalog checks that the selected
calculation is GF2, verifies all required values, loads the corresponding
semantic datasets, and returns an immutable-by-convention value object. A
future calculation adds another typed requirement set and definitions to the
same catalog; it does not open the HDF5 file independently.

### Keyword and semantic-data definitions

A definition is the single source of truth for a keyword. The exact C++
representation is implementation detail, but it must express the following
information:

```cpp
enum class ValueKind { String, Double, Int };

struct ValueDefinition {
  std::string keyword;                 // canonical spelling, e.g. "ETA"
  ValueKind kind;
  std::optional<Value> default_value;  // absent means required when requested
  std::function<Value(Value)> sanitize;
  const char* documentation;
};

struct DatasetDefinition {
  std::string keyword;                 // e.g. "HCORE"
  DatasetKind semantic_kind;           // Hcore, MoCoeff, Eri3, ...
  std::function<Tensor<double, Executor::Host>(DataSet&)> load;
  const char* documentation;
};
```

The registry also records which calculation requirement sets consume each
keyword. It must not be possible for a calculation to request `HCORE` and
receive an untyped array: the `HCORE` dataset definition constructs a rank-2
`double` Tensor with two AO dimensions bound to one fresh `Hermitian()` group.
Likewise, `ERI3` constructs the documented `{ao, ao, ri}` Tensor with one
fresh `Symmetric()` group on its AO pair, and `MO_COEFF` constructs a plain
`{mo, ao}` Tensor. These semantic constructors belong to the catalog/data
layer, not to GF2 numerical code.

For this story the GF2 requirement set is:

```text
HCORE, ERI3, MO_COEFF, ETA
```

`ETA` has a default of `1e-5`; the three datasets are required. Story 07.1
will extend this set with `OVERLAP`, `DENSITY_MATRIX`, `BETA`,
`MATSUBARA_HALF_N`, and `MU` in the same framework. Story 07.1 must not add a
second parser or loader to do so.

### Resolved values and provenance

`ResolvedInput` stores canonical keyword values and provenance:

```cpp
struct ResolvedInput {
  Calc calculation() const;
  bool supplied(std::string_view keyword) const;
  double eta() const;
  // analogous typed accessors for paths and future values
};
```

A defaulted value is present in the resolved input but `supplied("ETA")` is
false. A user-provided value is sanitized, stored, and reports true. This
allows callers to distinguish configuration provenance without re-parsing the
input or comparing against the default.

Sanitization belongs to the catalog. For example, `ETA` must be finite and
strictly positive; invalid values are rejected before a calculation receives
them. The same rule applies whether the value was read from a file or supplied
through a future programmatic input interface. The catalog should not silently
repair invalid values. It may normalize syntax such as Fortran `D` exponents
before validation.

### Proposed public API

The names and exact ownership may be adjusted during implementation, but the
responsibility boundary should remain:

```cpp
class InputCatalog {
public:
  static InputCatalog from_file(const std::string& input_path,
                                const std::string& hdf5_path);

  const ResolvedInput& resolved() const;
  Gf2Input require_gf2() const;
};

struct Gf2Input {
  Tensor<double, Executor::Host> hcore;
  Tensor<double, Executor::Host> mo_coeff;
  Tensor<double, Executor::Host> eri3;
  double eta;
  bool eta_was_supplied;
};
```

`Gf2Input` is a semantic requirement object, not a second source of truth.
It contains already-loaded tensors and validated values. The catalog remains
responsible for constructing it. The GF2 algorithm receives this object and
does not know the input keywords or HDF5 paths.

A separate `parse_input(text)` helper is still useful for parser unit tests,
but it returns the catalog's resolved-input representation rather than an
independent ad-hoc parameter map. `from_file` performs parsing, default
resolution, sanitization, file opening, and dataset validation in one normal
route. A `from_text(text, hdf5_path)` test helper may reuse the same catalog
pipeline without changing production ownership.

## Parser and registry behavior

The parser accepts the existing assignment format and remains deliberately
small. Keywords are case-insensitive and canonicalized; HDF5 path values are
preserved case-sensitively. Unknown and duplicate keywords are hard errors.
Full-line `#` and `//` comments and blank lines are ignored. Values are trimmed
at the assignment boundary but are not otherwise rewritten except where the
value definition explicitly permits syntax normalization.

The keyword grammar should support the planned names:

```text
KEYWORD = [A-Za-z][A-Za-z0-9_]*
```

The catalog resolves the selected calculation, applies defaults, sanitizes all
resolved values, and reports all missing required values together. It must
reject malformed doubles, non-finite values, invalid integer syntax, and
invalid semantic ranges with actionable keyword/line diagnostics.

For this story, `ETA` accepts ordinary and Fortran `D`/`d` exponent notation,
requires full-token consumption, and must be finite and strictly positive.
`ETA` is retained for compatibility with later Green's-function calculations;
this story does not use it numerically.

### Parser spec

Input is text; extension is irrelevant (consistent with Story 2). Line grammar:

- A **line** is: `(blank)`, a **comment**, or an **assignment**.
- **Blank**: empty or whitespace-only → ignored.
- **Comment**: a line whose first non-whitespace char is `#` (and `//` is ALSO treated as a comment start)
  → the whole line is ignored. (Comments keep input files readable; both `#` and `//` are cheap to support.)
- **Assignment**: `KEYWORD [ws] = [ws] VALUE` where
  - `KEYWORD` matches `[A-Za-z][A-Za-z0-9_]*` and is looked up case-insensitively.
  - `VALUE` is the remainder of the line, trimmed of leading/trailing whitespace. Any whitespace inside a
    string/path value is preserved as-is; quoting and inline comments are out of scope.
- A line that has content but no `=`, or a `= ` with an empty KEYWORD, is a hard error.

Parse rules:

1. Look up `KEYWORD` in the registry (case-insensitive). **Not found → hard error** whose message names the
   offending keyword and line number. (This is the "unrecognized keyword" rule. There is no warn/ignore path.)
2. Dispatch by the keyword definition's value kind:
   - `String`/`DatasetPath`: store the trimmed value in the resolved-value builder; an empty value is an
     error for required values.
   - `Double`: parse VALUE as a double (see Double parsing). On failure → hard error naming the keyword,
     the offending token, and the line.
3. After all lines are read, resolve the **calculation**:
   - No `CALCULATION` keyword → hard error ("must select a calculation").
   - `CALCULATION` value → matched case-insensitively against known calcs; `GF2` → `Calc::Gf2`. Unrecognized
     name → hard error naming the value.
4. **Requirement check** for the selected calculation: every keyword in its catalog requirement set must be
   present either in the input or as a defined default; otherwise report all missing required keywords at once.
5. **Default application and sanitization**: apply typed defaults, record provenance, and run every value
   through its definition's sanitizer. For example, `ETA` becomes `1e-5` when omitted and must be finite and
   strictly positive when supplied.
6. Return the resolved-input portion of the `InputCatalog`.

**Duplicate keywords:** a keyword given twice on different lines is a hard error (ambiguous input), not last-wins.

**Double parsing** (`1e-5`, `1E-5`, `1D-5`, `1d-5`, `0.00001` all → `1e-5`):
- Trim the token.
- Normalize a Fortran `D`/`d` exponent marker to `E` (so a `D` that is functioning as an exponent is
  accepted). Simplest robust approach: run the token through a scan that replaces a `D`/`d` in exponent
  position with `E`, then `std::strtod`.
- Require **full consumption** (no trailing non-whitespace characters), otherwise hard error. (So `1e-5x`
  fails, and a bare non-numeric token like `foo` fails.)
- Reject NaN/Inf tokens as out of scope for `eta` (documented; treat as a parse error).

### Data loading (GF2) & dimension labels

`InputCatalog::require_gf2()` returns the following semantic requirement
object (the exact header placement is implementation detail):

```cpp
struct Gf2Input {
  Tensor<double, Executor::Host> hcore;     // AO × AO, shared Hermitian group
  Tensor<double, Executor::Host> mo_coeff;  // AO × MO, plain axes
  Tensor<double, Executor::Host> eri3;      // RI × AO × AO, shared symmetric AO group
  double eta;
  bool eta_was_supplied;
};
```

The catalog opens the HDF5 file once and loads the datasets named by the
resolved keyword paths. It constructs each Tensor with its physical labels and
advisory symmetries, validates element type/rank/extents, and returns the
requirement object. There is no public `load_gf2(pin, path)` function that
allows GF2 code to bypass the catalog.

**Dimension labels** (physical meaning; by Story 06 label rule 2 the interchangeable pair may — must —
share one label bound by a shared `SymmetryGroup`):

| tensor     | H5 shape (slow→fast) | `Tensor` dims (fast→slow) | labels / symmetries |
|------------|----------------------|---------------------------|---------------------|
| `hcore`    | `(ao, ao)` 24×24     | `{ ao=24, ao=24 }`        | `ao`,`ao` — same label, bound by ONE shared `Hermitian()` SymGroup |
| `mo_coeff` | `(ao, mo)` 24×24     | `{ mo=24, ao=24 }`        | `ao` (plain), `mo` (plain) — no SymGroup |
| `eri3`     | `(ri, ao, ao)` 116×24×24 | `{ ao=24, ao=24, ri=116 }` | `ao`,`ao` — same label, bound by ONE shared `Symmetric()` SymGroup; `ri` (plain) |

Construction (Story 06 API; each factory call is a FRESH group, so each tensor's pair is its own family):

```cpp
auto hcore_g = Hermitian();   // one shared group object for hcore's AO pair
auto eri3_g  = Symmetric();   // one shared group object for eri3's AO pair

Tensor<double, Executor::Host> hcore   ( { TensorDim{"ao", N_AO, hcore_g}, TensorDim{"ao", N_AO, hcore_g} });
Tensor<double, Executor::Host> mo_coeff( { TensorDim{"mo", N_MO},          TensorDim{"ao", N_AO} });
Tensor<double, Executor::Host> eri3    ( { TensorDim{"ao", N_AO, eri3_g},  TensorDim{"ao", N_AO, eri3_g}, TensorDim{"ri", N_RI} });
```

Conventions:
- `ao` is the **AO** index, `mo` the **MO** index, `ri` the **RI / density-fit** index.
- The two interchangeable AO axes of a tensor are **identical axes** and therefore *share the label `ao`*;
  Story 06 label rule 2 makes that legal precisely when every occurrence of `ao` in the tensor is bound to
  the SAME (non-null) `SymGroup` — which is exactly the symmetry we declare here. `mo_coeff` keeps `ao` as
  a plain (single-occurrence) label alongside its `mo` axis.
- Symmetry is **advisory** (Story 06, decision Q3): it names the physics and lets no-op work be skipped
  (e.g. `hcore.transpose(0,1)` is the identity on `double`; `eri3.transpose(0,1)` likewise). It does NOT
  alter storage (tensors stay dense and full), is not passed to the `TensorBackend`, and does **no**
  consistency checking of the loaded data.
- HDF5 datasets are C-order (last index fastest) and `Tensor`'s `data()` is last-dim-fast, so a dataset's
  contiguous `float64` memory maps directly onto the tensor's flat storage **in this label order**. Loading
  is therefore a direct `HighFive` read of `tensor.data()` with no transposition and no symmetry-dependent
  work (the two `ao` axes are identical sizes by construction, so a `SymGroup` pair never changes layout).
  (If a future tensor is stored in H5 in a different axis order than we want labelled, a transpose story will
  handle the swap — out of scope here.)
- **Validation before/while loading:** the dataset must exist, be a dataset whose element type is equivalent
  to `double` (not an integer or silently converted float type), and have the expected rank and dimension
  sizes (N_AO, N_MO, N_RI) consistent with the other semantic objects. The N_AO in `hcore`, `mo_coeff`,
  and both AO axes of `eri3` must agree. Mismatches are hard errors naming the semantic object, keyword,
  dataset path, and offending axis.

### Error handling & exit codes

- Structural problems (bad line, unknown keyword, bad double, missing `CALCULATION`, missing required
  keyword, duplicate keyword, unrecognized calculation name) → `std::invalid_argument` from the parser, with a
  message that names the keyword **and line number** where known.
- Data problems (dataset not found, wrong type, rank/size mismatch) → `std::runtime_error` /
  `std::invalid_argument` from `InputCatalog::require_gf2()`, naming the semantic object, keyword, and
  dataset path value.
- The CLI (`main.cpp`) already routes `-i/-d` into `check_input_file`/`check_data_file`; update the normal
  mode to: `InputCatalog::from_file(opt.input, opt.data)` → `require_gf2()` → print a short summary of the
  loaded tensors (shapes/labels + `eta` and whether it was supplied) → return 0. A thrown exception prints
  the message to `std::cerr` and returns non-zero (2), **not** the help page (help is only for bad
  *invocations*, per Story 2).

### Files touched / added

- `include/input.hpp` / `src/input.cpp` (new) — `InputCatalog`, resolved values/provenance,
  semantic definitions, parser, defaulting, and sanitization.
- `include/gf2.hpp` / `src/gf2.cpp` (new) — `Gf2Input` and the catalog's typed GF2 requirement
  implementation, including HighFive validation and semantic Tensor construction.
- `src/main.cpp` — wire GF2 selection + loading into normal mode; add `Calc` selection from parsed input.
- `src/tests.cpp` / `src/tests.hpp` — add `run_input_data_loading_tests()`; call it from `run_tests()`.
- `CMakeLists.txt` — add `src/input.cpp` and `src/gf2.cpp` to `add_executable(cppgw ...)`.

---

## Tests

Split into **unit tests** (pure, no files — run in the process under `-t`) and
**HDF5 integration tests** (self-contained: they write a small known HDF5 file via HighFive into a temp path,
then load it — no dependency on `build/rhf_df.h5`), plus a **CLI smoke check** against the real `rhf_df.h5`.

### A. Input parser unit tests (`-t` suite, no files)

- **Case-insensitive keywords:** `eri3 = path/eri3`, `Eri3 = ...`, `ERI3 = ...` all resolve to canonical `ERI3`.
  Similarly `eta`, `ETA`, `Eta`.
- **Path case preserved exactly:** `ERI3 = My/Path/eri3` → stored VALUE is exactly `My/Path/eri3`
  (case-sensitive passthrough), trimmed only of surrounding whitespace.
- **`eta` double forms** → all equal `1e-5` (within a tight tolerance): `1e-5`, `1E-5`, `1D-5`, `1d-5`,
  `0.00001`, `1.0E-5`, ` +0.00001 ` (leading space / `+`), `0.1e-4`.
- **`eta` default:** input with no `eta` line → `pin.eta() == 1e-5`; with `eta = 2.5e-4` → `pin.eta() == 2.5e-4`
  (default NOT applied when the value is present).
- **Unknown keyword (hard error):** e.g. `HERI3 = x` or `foo = 1` → throws `std::invalid_argument`; message
  contains the offending keyword text.
- **`CALCULATION`:** `CALCULATION = GF2`, `= gf2`, `= Gf2` all select `Calc::Gf2`. `CALCULATION = GW`
  (no GW calc) → throws (unrecognized name). No `CALCULATION` line → throws (must select a calculation).
- **Missing required keyword (hard error):** GF2 input lacking `HCORE` → throws, and the message names `HCORE`
  (and any other missing required keywords in one go).
- **Malformed line (hard error):** a line with content but no `=` (e.g. `just some words`), and a lone
  `= value` (empty keyword) → both throw.
- **Double parse failures (hard error):** `eta = foo`, `eta = 1e-5x`, `eta = 5!`, `eta = ""` (empty) →
  each throws `std::invalid_argument`.
- **Duplicate keyword (hard error):** the same keyword on two lines → throws (not last-wins).
- **Comments / blanks ignored:** lines `# comment`, `// comment`, and blank/whitespace-only lines are
  ignored; the file still parses.
- **A full happy-path file** exactly like the brief (plus `CALCULATION = GF2`) parses to the expected
  `calc`, the three stored path strings, and `eta`.

### B. GF2 data-loading integration tests (`-t` suite, self-contained HDF5 via HighFive)

For each test, create a temp HDF5 file at a known path with HighFive, write datasets with known values:

- **Happy load:** write `hcore`(3×3),`mo_coeff`(3×3),`eri3`(5×3×3) float64 datasets with a deterministic
  fill (e.g. linear ramp); construct an `InputCatalog` pointing at them (`ERI3/HCORE/MO_COEFF`); call
  `catalog.require_gf2()`; assert
  - `hcore` rank 2, dims `{ao=3, ao=3}`, `mo_coeff` rank 2 dims `{mo=3, ao=3}` (H5 `(ao, mo)`:
    `mo` is the fast axis), `eri3` rank 3 dims `{ao=3, ao=3, ri=5}`;
  - **declared symmetries:** `hcore.dims()[0].symmetry` and `hcore.dims()[1].symmetry` are `same_symgroup`
    with `kind() == Hermitian`; `eri3`'s two `ao` axes (positions 0 and 1) are `same_symgroup` with
    `kind() == Symmetric`; `mo_coeff`'s axes (and `eri3`'s `ri` axis) carry a null `SymGroup`
    (`null_symgroup`);
  - **symmetry behaviour on the loaded tensors:** `hcore.transpose(0,1)` equals `hcore` (Hermitian on a
    real scalar is the identity no-op); `eri3.transpose(0,1)` equals `eri3` (Symmetric no-op);
    the plain (unbound) `mo_coeff` axes, if asked for, would be physically permuted by `transpose`
    (Story 06.1) rather than treated as a symmetry no-op — this story does not rely on that.
  - every stored element equals the written value (full-array sweep, max|err| == 0.0, since it is a copy);
  - `eta` round-trips from the input.
- **AO-size consistency:** write `hcore` as 3×3 but `mo_coeff` as 4×3 (N_AO mismatch) → `catalog.require_gf2()` throws
  naming the offending tensor/axis.
- **Missing dataset:** point a path at a name that does not exist (e.g. `ERI3 = nope`) → throws.
- **Wrong element type:** write `hcore` as `int32` → throws (only `double` accepted).

### C. CLI smoke check against the real `rhf_df.h5` (command-line acceptance, Story-2 style)

Create a real input file (e.g. `build/test.in` updated, or a new `gf2.in`) and verify, mirroring Story 2's
"finished when …" phrasing:

- `./cppgw -i gf2.in -d rhf_df.h5` → runs the GF2 stub successfully, prints a summary (loaded tensor
  shapes/labels/symmetries: `hcore{ao=24[hermitian] ×2}`, `mo_coeff{mo=24, ao=24}` (plain),
  `eri3{ao=24[symmetric] ×2, ri=116}`, `eta=…`), exits 0.
- `./cppgw gf2.in rhf_df.h5` → identical (positional form).
- `./cppgw -i badkw.in -d rhf_df.h5` (a file with an unknown keyword such as `HERI3 = eri3`) → prints a
  clear error naming the keyword, exits non-zero.
- `./cppgw -i missingcalc.in -d rhf_df.h5` (no `CALCULATION`) → clear error, exits non-zero.
- No `eta` in the input → still succeeds and reports the defaulted `eta = 1e-5`.
- Existing invariants still hold: `-t`/`-test` run the **full** suite (including the new A/B tests) and the
  suite is excluded from bad invocations; `-h` still prints help (Story 2 behavior unchanged).

### Definition of done

- `run_input_data_loading_tests()` is wired into `run_tests()`, so `./cppgw -t` runs the existing Story 2–4
  and Story 6 tests **and** the new A/B tests, all passing (no throw).
- `./cppgw -i gf2.in -d rhf_df.h5` (and positional form) runs the GF2 stub, loads the three tensors, and
  exits 0 with a visible summary.
- **The loaded tensors carry their declared symmetries (Story 06 API):** `hcore`'s two `ao` axes are bound
  by a shared `Hermitian()` `SymGroup`, `eri3`'s two `ao` axes by a shared `Symmetric()` `SymGroup`, and
  `mo_coeff` is plain; the no-op `transpose(i,j)` on the bound pairs behaves per Story 06.
- Every hard-error path (A + C) prints an actionable message identifying keyword/line or tensor/axis and
  exits non-zero.
- No GF2/GW numerics are implemented — ingestion and wiring only.

---

## Open / noted for later (out of scope here)

- **GW** calculation: add its semantic definitions and typed requirement set to `InputCatalog` in the same
  files; do not add a second loader.
- **Transpose / index-permutation on load:** not needed for the tensors as stored today; a future story.
- **Other `float64` tensors** in `rhf_df.h5` (`overlap`, `density_matrix`, `mo_energy`, …): not used here;
  the registry makes them easy to add if a future calc needs them.
- **Device-side loading** and multi-file HDF5 layouts: out of scope. Dataset values may use ordinary HDF5
  paths (including groups), but the catalog opens one file and reads dense datasets only; sparse, compound,
  or object-valued datasets are out of scope.
