# Story 06 — Tensor index symmetry & the Tensor backend seam

## Motivation

Some physical quantities have symmetry in their indices, and the current `Tensor` machinery has no
notion of it. At this point it's worth revisiting the Tensor design.

The main reason we "roll our own" Tensor is the use of special `TensorDimLabel`s — the
`ChebyshevExpansionDimLabel` and `GridDimLabel` tags (and more to come). These encode, as a single
source of information, *which temporal/function space* an axis lives in and *what representation*
(grid vs. basis expansion) a Tensor uses. That part is working and load-bearing (the Grid and
Chebyshev expansions in Stories 03–04 depend on it).

What we lack is the standard Tensor-library feature set. The reference points:

- **Eigen's tensor** module (technically unsupported, widely used) has index-permutation symmetry and
  automatic linear-algebra kernel delegation — exactly the gaps we have.
- **cuTENSOR** (and friends) offer the same on-device. Notably cuTENSOR does **not** auto-dispatch on
  tensor index symmetries — that would be our job in a *future* story and is out of scope here.

We've built two genuinely useful primitives we want to keep:
1. **DimLabels** — name the indices meaningfully; a natural, physics-matching process.
2. **Host/Device abstraction** — `Tensor<scalar, Executor::{Host,Device}>` already templates over the
   executor.

What we still need:
1. **Index permutation symmetry** — physical tensors have sets of interchangeable indices under some
   group. Examples: the core Hamiltonian is Hermitian; the two AO indices of the 3-index ERI/DF tensor
   are symmetric: `(ri, ao1, ao2) == (ri, ao2, ao1)`.
2. **A performant backend** — the numerical kernels currently live hand-written behind `LinAlgBackend`.

Because features (1) and (2) are **critical, hard to get right, and largely available in existing
libraries**, this story does *not* rewrite the engine. It makes the Tensor implementation **thinner**
and narrows its scope to what only *we* own:

- the **DimLabel / function-space semantics** (non-portable, ours),
- **index symmetry** as an advisory, physics-matching concept (ours), and
- a clean **TensorBackend seam** so a performant library (BLAS / Eigen / cuTENSOR) can be slotted in
  later *without touching the frontend* (the seam is this story; the library is not).

## Agreed decisions

- **(Q1) Duplicate labels are ALLOWED, on purpose.** Mirroring the symmetry (two `"ao"` axes) matches
  the physics, and it makes it obvious that indexing `"rab"` vs. `"rba"` is the same. We **restructure**
  the label-addressing machinery (uniqueness check, `label_index`, `gemm`, `prepare_output`) to support
  duplicate labels bound by a symmetry relation. (Not the "distinct-label + side relation" alternative.)
- **(Q2) Symmetry identity is instance-based, stored as a `shared_ptr`.** Each symmetry *group* object
  carries a stable identity; two tensor axes are "the same interchangeable-axis family" iff they were
  constructed with the **same `SymmetryGroup` pointer**. `Symmetric`-kind groups `g1` and `g2` are
  distinct even though same kind (so a rank-4 `(ao1,ao1,ao2,ao2)` can carry two independent symmetric
  pairs).
- **(Q3) Symmetry is ADVISORY, not enforced.** It exists to (A) give a user-facing API that matches the
  physics, and (B) let operations that mathematically reduce to a no-op be *skipped*. It is **never**
  used to compress storage (every Tensor stays dense and full — a triangle-aware kernel would be the
  *wrong* abstraction against BLAS/`Tensor` libraries) and it does **not** trigger any data copy/permute
  on construction.
- **(Q4) The backend seam is in scope; the backend *replacements* are not.** We introduce a
  `TensorBackend` interface (set of required unary + linear-algebra ops, templated over
  `Executor::{Host,Device}`). The **host kernels stay hand-written** and are simply re-homed behind the
  seam. **BLAS / Eigen / cuTENSOR integration is a future story** (it lands as new `TensorBackend`
  implementations / kernel bodies under the same op contract, with no changes to `Tensor` or to how
  labels/symmetry are validated).
- **(Q5) Symmetry kinds in scope:** `Symmetric`, `Hermitian`, **and `Antisymmetric`**.
  - `Hermitian` on a **real** scalar reduces to `Symmetric` (complex-conjugation is a no-op on `double`)
    so the machinery templates cleanly over real/complex.
  - `Hermitian`/`Antisymmetric` on a **complex** scalar carry their conjugation/sign semantics.
- **Scope limit (the real one, not "adjacency"):** only **pairwise (2-axis) relations** this story, at
  *any* index positions (adjacency is not required — see *Why adjacency isn't the right boundary*).
  **Multi-member / transitive symmetry groups** (the full rank-4 double-exchange orbit) and the **active
  transpose/permutation primitive** for non-identical axes are deferred to a future story.

---

## Why adjacency is not the right boundary (and what we instead defer)

The instinct is to restrict symmetries to *adjacent* index pairs, because that's the only way to make
**duplicate-label + contraction** bookkeeping trivially unambiguous. But it isn't needed:

1. **Both motivating purposes are adjacency-agnostic.** (A) is pure metadata. (B) — "skip transposes
   that reduce to a no-op" — only ever *avoids* physically reordering storage (Symmetric→identity,
   Hermitian→conjugate, Antisymmetric→negate); it never asks the engine to actually permute.
2. **`gemm` already contracts by position.** Its existing rule (label = last dim of `X`, first dim of
   `Y`) means the *surviving* twin of a symmetric pair simply keeps its position. Whether the twins are
   neighbours depends on nothing but the user's axis order; the bookkeeping is per-position either way.
3. **The actual scope explosion is group closure.** A pairwise relation ("axes `p` and `q` are
   interchangeable under kind K") is a single fact to store and look up. A *multi-member* group (an axis
   with >2 interchangeable twins, e.g. the full `(μν|λσ)` double-exchange) is a permutation *set*; using
   it correctly requires orbit math and — for `gemm`/output-shape consistency — a genuine transpose
   primitive for non-identical axes. That is the expensive part, so **that** (not adjacency) is what is
   deferred.

Concretely, in scope: any number of *distinct* symmetric/Hermitian/antisymmetric **pairs**, at any
positions, e.g. `Tensor<...>{ {ao_i,30,g1},{ao_j,30,g2},{ao_k,40},{ao_i,30,g1},{ao_j,30,g2} }`.
Out of scope: asserting all four above are in one group, or that `ao_i` is interchangeable with `ao_k`.

---

## Goals

1. **`IndexSymmetry` / `SymmetryGroup` API** and the extension to `TensorDim` so a tensor can
   declare that a set of its axes are pairwise symmetric / Hermitian / antisymmetric, using *identical
   labels* for the interchangeable axes.
2. **Relax + rework the label invariant** so duplicate labels are legal *iff* they are bound by a shared
   `SymmetryGroup` (advisory), and keep every existing (label-less-symmetry) Tensor valid.
3. **Rework label-addressing** (`label_index`/`has_label`/`gemm`/`prepare_output`/shape printing) to be
   correct in the presence of duplicate (symmetric) labels — `gemm` stays position-based and now works
   on symmetric tensors.
4. **Introduce the `TensorBackend` seam** (required-op contract, `Executor`-templated) and re-home the
   existing host `gemm` behind it; add the minimal **unary (elementwise)** ops (`fill`, `zero`, `scale`,
   `conjugate`) on the host. Host kernels remain hand-written this story.
5. **Realize goal (B)** with a small, honest primitive: `Tensor::transpose(i, j)` that, when `(i,j)` is
   a symmetry-bound pair, returns the no-op result *without* reordering storage (identity / conjugate /
   negate), and throws a clear "not yet implemented" for a non-identical axis pair (the real transpose is
   the future story).
6. **Wire the physics into GF2** (linking Stories 05 and 07): `hcore`→Hermitian, `eri3`→symmetric on its two
   AO axes, `mo_coeff`→none, using the new API.

Out of scope: multi-member symmetry groups, active transpose of non-identical axes, symmetry *propagation*
to `gemm` outputs (the caller declares the output's symmetry), data-validity checks ("is `hcore` actually
Hermitian?"), storage compression, and any BLAS/Eigen/cuTENSOR backend.

---

## API design

### `IndexSymmetry` / `SymmetryGroup`

New `include/symmetry.hpp`:

```cpp
enum class SymKind { Symmetric, Hermitian, Antisymmetric };

// A pairwise symmetry "group": a value (kind) whose IDENTITY is what matters.
// Two tensor axes are considered interchangeable iff they were built with the SAME
// SymmetryGroup instance (the same SymGroup handle).
class SymmetryGroup {
public:
  explicit SymmetryGroup(SymKind k) : kind_(k) {}
  SymKind  kind() const noexcept { return kind_; }
  // value-comparison is deliberately NOT meaningful for group identity;
  // equality is by handle (same object) — see SymGroup below.
private:
  SymKind kind_;
};

// The handle TensorDim stores. "Same group" == same underlying SymmetryGroup object.
using SymGroup = std::shared_ptr<const SymmetryGroup>;

// Ergonomic factories (as in the original draft). Each call is a FRESH group.
inline SymGroup Symmetric()     { return std::make_shared<const SymmetryGroup>(SymKind::Symmetric); }
inline SymGroup Hermitian()     { return std::make_shared<const SymmetryGroup>(SymKind::Hermitian); }
inline SymGroup Antisymmetric() { return std::make_shared<const SymmetryGroup>(SymKind::Antisymmetric); }

// Handle-based identity + printable rendering.
inline bool same_symgroup(const SymGroup& a, const SymGroup& b) { return a == b; }
inline bool null_symgroup(const SymGroup& a)                     { return a == nullptr; }
```

Usage (the draft's three examples, now well-defined):

```cpp
// 3-index ERI / DF:  (ri, ao1, ao2) with ao1 ~ ao2
{
  auto g = Symmetric();
  Tensor<double, Executor::Host> eri3(
      { {TensorDimLabel("ri"), 120 },
        { TensorDimLabel("ao"), 50, g },   // the two "ao" axes are the SAME family (g)
        { TensorDimLabel("ao"), 50, g } });
}

// rank-4 ERI: two INDEPENDENT symmetric pairs (ao1 ~ ao1, ao2 ~ ao2)
{
  auto g1 = Symmetric();                 // first pair
  auto g2 = Symmetric();                 // second pair (distinct group, same kind)
  Tensor<std::complex<double>, Executor::Host> eri4(
      { {TensorDimLabel("ao1"), 50, g1}, {TensorDimLabel("ao1"), 50, g1},
        {TensorDimLabel("ao2"), 50, g2}, {TensorDimLabel("ao2"), 50, g2} });
}

// one-electron Hamiltonian: Hermitian (reduces to Symmetric when the scalar is real)
{
  auto h = Hermitian();
  Tensor<double, Executor::Host> hcore(
      { {TensorDimLabel("ao"), 50, h}, {TensorDimLabel("ao"), 50, h} });
}
```

### `TensorDim` / `TensorShape` changes

```cpp
struct TensorDim {
  TensorDimLabel label;
  size_t         dim = 0;
  SymGroup       symmetry = nullptr;   // NEW: null => plain axis (the old, common case)
  TensorDim(TensorDimLabel l, size_t d, SymGroup s = nullptr)
      : label(std::move(l)), dim(d), symmetry(std::move(s)) {}
  bool operator==(const TensorDim&) const = default;  // now compares the SymGroup handle
};
```

- Existing `TensorDim{label, dim}` still compiles (symmetry `nullptr`); all current Grid/Chebyshev
  Tensors and the `gemm`/Fourier tests are unaffected.
- `TensorShape` (the `std::vector<TensorDim>` wrapper used by the Chebyshev expansions) needs no change
  to members, but any `==`/hash over a shape now includes the symmetry handles (add a hash of `kind` +
  handle if a shape is ever used as a map key; currently it is only compared member-wise).

### The new label invariant (replaces `validate_unique_labels`)

An axis is **plain** (symmetry `nullptr`) or **bound** (symmetry a `SymGroup` handle). Rules:

1. **All members of one `SymGroup` are identical axes**: they MUST carry the same string label and the
   same size. (A symmetry group is a set of interchangeable, *identical* axes.) A mismatch → error at
   construction.
2. **A string label may be repeated only as one symmetry family.** Concretely: if a label occurs on more
   than one axis, **every** occurrence MUST be bound to the **same** (non-null) `SymGroup`. Two axes with
   the same label but a null/different/absent `SymGroup` is an **error** — that would make two distinct,
   non-interchangeable axes inseparable by name, breaking addressing.
3. **Addressability:**
   - A **plain** label occurs exactly once (rule 2 implies plain labels are unique).
   - A **symmetric-family** label may occur many times; it names the *family*. Any one member is a
     valid representative ("indexing `rab` vs. `rba` is the same").
   - `Tensor::label_index(label)` keeps its old meaning and simply returns the **first** position bearing
     the label (fine when the label is plain; a valid representative when it is a family).
   - `Tensor::label_indices(label) -> std::vector<size_t>` (new) returns **all** positions bearing the
     label — the tool `gemm`/`prepare_output` use to reason about duplicates.

These invariants make "the axis named `ao`" a well-defined family and restore unambiguous name-based
addressing without ever requiring the axes to be adjacent.

### `Tensor` frontend rework

- Replace `validate_unique_labels` with `validate_dims_and_symmetry` implementing the rules above;
  throw `std::invalid_argument` naming the offending label/axis/position on any violation.
- `gemm(X, Y, Z, labelX, labelY)` — **semantics unchanged** (contract X's *last* dim `labelX` with Y's
  *first* dim `labelY`; result = X minus last, then Y minus first). What changes is only that it now
  *permits* `labelX`/`labelY` to be duplicated symmetric families, and the "contracted size must agree"
  check compares the specific last/first positions. Output-dims bookkeeping is per-position, so it is
  unaffected by duplicate/`non-adjacent` twins.
  - **No symmetry auto-propagation:** `Z` is *not* given any symmetry by `gemm`. The caller declares the
    output's symmetry (e.g. "contracting `eri3` over one AO yields a Hermitian 2-index AO tensor") if
    desired. Deriving the output's symmetry from the operands' groups is a future story.
- `add_dim(newdim, slow, fill)`: the new axis carries whatever `newdim.symmetry` is (default plain/null).
  No inheritance from existing axes.
- `operator()`, `linear()`, `data()` — **unchanged** (always position/offset based, unaffected).
- **New (goal B):** `Tensor::transpose(size_t i, size_t j)` — the *no-op-equivalent* of transposing a
  symmetry-bound pair, **implemented without any index permutation**. It is a small, cheap primitive that
  reuses the new `TensorBackend` unary ops rather than a general transpose (so it stays "easy to implement"):
  - first resolve the pair through the new symmetry machinery: both positions in range, and
    `dims_[i].symmetry` and `dims_[j].symmetry` are the **same non-null `SymGroup`** (handle-identity via
    `same_symgroup`). This is exactly the machinery this story adds, so it directly *uses* the new symmetry API.
  - then return the correct no-op result by **kind** (`SymmetryGroup::kind()`): `Symmetric` → identity,
    `Hermitian` → elementwise complex conjugate, `Antisymmetric` → elementwise negation:
    - `Symmetric`, or `Hermitian` on a **real** scalar (which reduces to `Symmetric`) → return `*this`
      (an elementwise-identical copy — no restride/permute computed),
    - `Hermitian` on a **complex** scalar → a conjugated copy (built via the `TensorBackend::conjugate` op),
    - `Antisymmetric` → a negated copy (built via `TensorBackend::scale(T, -1)`).
  - **Not zero-copy.** `Tensor` has no view type; returning by value is a deep copy. That is fine and
    intended: the payoff is that we *never implement a real index permutation* — the no-op cases are served
    by a trivial elementwise op (or identity copy). (A zero-copy symmetric "view" would need a view/reference
    type on `Tensor`, which is out of scope.)
  - **Position-agnostic** (consistent with *Why adjacency isn't the right boundary*): the symmetric pair has
    identical sizes (rule 1), so transposing axes `i`,`j` leaves the flat layout unchanged, and the elementwise
    op is correct whether or not `i`,`j` are adjacent, and regardless of other axes.
  - otherwise (out-of-range position, or `i`,`j` are *not* bound by the same `SymGroup`) → throw
    `std::logic_error("Tensor::transpose: transpose of non-identical axes not yet implemented (Story 06);
    use a SymmetryGroup-bound pair for the no-op case")`.
  - The *active* permutation of genuinely distinct axes remains the deferred story.

### `TensorBackend` seam (replaces/augments `LinAlgBackend`)

`include/backend.hpp` (new). The contract is **the set of operations**, executor-templated. The host
implementation is hand-written (moved out of `LinAlgBackend`); BLAS/Eigen/cuTENSOR are later
implementations of the same contract.

```cpp
// The required-operation contract a backend must provide (the "seam").
// Templated over the OUTPUT scalar type and the Executor, exactly as today's
// LinAlgBackend. Kernels operate on raw TensorBuffers and know nothing about
// labels or symmetry (that stays in the Tensor frontend).
template <typename TOut, typename Executor>
concept TensorBackendOps = /* provides: gemm, fill, zero, scale, conjugate (below) */;

// Host implementation (hand-written this story; BLAS/Eigen land here later):
template <typename TOut>
class TensorBackend<TOut, Executor::Host> {
  // linear algebra (moved from LinAlgBackend)
  template<typename TA, typename TB>
  static void gemm(const TensorBuffer<TA,Host>& A, size_t M, size_t K,
                   const TensorBuffer<TB,Host>& B, size_t N,
                   TensorBuffer<TOut,Host>& C);
  // unary / elementwise
  static void fill      (TensorBuffer<TOut,Host>& T, TOut value);
  static void zero      (TensorBuffer<TOut,Host>& T);
  static void scale     (TensorBuffer<TOut,Host>& T, TOut alpha);
  static void conjugate (TensorBuffer<TOut,Host>& T);   // no-op for real scalars
};
```

- The **Tensor frontend** calls *only* ops in this contract, so swapping/adding a backend later touches
  no validation, label, or symmetry code. This is the "get the seam right" part of the decision.
- **Where symmetry lives:** the **frontend**. Symmetry lets the frontend *prune no-op work* (skip the
  `transpose`) and *validate* the tensor; it is **not** passed to the backend. The backend sees sanitized
  buffers and just performs `gemm`/`fill`/… on positions. (A *symmetry-aware* kernel — e.g. cuTENSOR
  dispatching on symmetries — is explicitly deferred, as cuTENSOR doesn't provide that today.)
- Keep a thin compatibility alias so existing `LinAlgBackend<TOut, Exec>` references keep compiling
  (or do a small rename pass in `main.cpp`/tests); both are acceptable — pick the rename for clarity.

### Error / exit policy

- Construction violations (rules 1–3 above) → `std::invalid_argument` naming the label + axis position(s).
- `transpose` on a non-bound pair → `std::logic_error` (documented, intentional, not a bug).
- `TensorBackend` kernels retain their current behaviour (assumed-sanitized inputs); no new failure modes.
- The `-t` suite (Story 2) gains a new test function; existing behaviour unchanged.

### Files

- `include/symmetry.hpp` (new) — `SymKind`, `SymmetryGroup`, `SymGroup`, factories, handle helpers.
- `include/tensor_shape.hpp` — `TensorDim` gains `SymGroup symmetry`; `TensorShape==`/hash touches.
- `include/tensor.hpp` — `validate_dims_and_symmetry`, `label_indices`, reworked `gemm`/`prepare_output`
  label handling, `Tensor::transpose(i,j)`, `print`/diagnostics disambiguation.
- `include/backend.hpp` (new) — `TensorBackend` contract + host ops; move host `gemm` here.
- `include/linalgbackend.hpp` — becomes a compatibility shim (or is folded into `backend.hpp`).
- `src/main.cpp` / `src/tests.*` — new `run_tensor_symmetry_tests()`; keep all existing tests green.
- No change to `CMakeLists.txt` (new files are headers + the existing `main` TU compiles them; if a
  `.cpp` is added for the backend ops, add it to the target).

---

## Tests

New `run_tensor_symmetry_tests()` in the Story-2 in-process `-t` suite (throw `std::runtime_error` on
failure). No external files needed.

### Construction & invariants
- Plain/legacy path: `TensorDim{label,dim}` (null symmetry) still constructs; unique plain labels still
  enforced (two plain `"ao"` axes with no group → throws).
- **Duplicate label WITH one shared `SymGroup`** (the `eri3` example) → constructs; `rank()==3`; the two
  `ao` axes carry the same handle.
- **Duplicate label with TWO distinct groups** (two plain `"ao"` axes, or `"ao"`+`g1` and `"ao"`+`g2`)
  → throws (a label must not span groups).
- **SymGroup members with mismatched sizes** (`{"ao",50,g},{"ao",40,g}`) → throws.
- **eri4 case** (`g1`,`g2`, both `Symmetric`, distinct handles) → constructs; `same_symgroup(g1,g2)==false`
  but both `kind()==Symmetric`; each label occurs exactly twice.
- `same_symgroup`/`null_symgroup` behave as specified; a `shared_ptr` COPY of the same handle is the same
  group (identity by object, so copies are equivalent).

### Symmetry queries
- `hcore` (Hermitian on `double`): `transpose(0,1)` equals `hcore` (conjugation is a no-op on reals).
- `eri3` (Symmetric): `transpose` over the two `ao` axes (positions 1 and 2) → equals `eri3`;
  `transpose(0, 1)` (ri↔ao, NOT a bound pair) → throws `std::logic_error`.
- An **Antisymmetric** pair (complex or real): `transpose(i,j) == -T` elementwise (diagonal would be 0 in a
  true antisymmetric matrix — we do NOT check this; just verify the sign flip).
- A **complex Hermitian** tensor: `transpose(i,j)` returns `conj(T)` elementwise (values checked).
- `label_indices("ao")` on `eri3` → `{1,2}`; `label_index("ao")` → `1` (first); `has_label` true.
- `mo_coeff {ao, mo}`: `label_indices("ao")=={0}`, `label_indices("mo")=={1}`, no bound pair
  (any `transpose` throws not-yet-implemented).

### `gemm` on symmetric tensors (regression + new)
- Contract the `eri3`-style `{ri, ao, ao}` over one `ao` against a vector/matrix on `ao`; confirm the
  numeric result is correct and the **output shape** keeps the surviving `ao` at its position (i.e. the
  per-position bookkeeping is right even though a label is duplicated).
- Re-run the existing real×complex→complex `gemm` test unchanged (backward compatibility).

### Backend seam
- The op concept is satisfied for host: `fill`/`zero`/`scale`/`conjugate` produce expected values
  (real and complex); `conjugate` on a real tensor is a no-op; `gemm` matches today's hand-written values
  exactly (same results, new home).
- `static_assert` (or a runtime check) that `Tensor` builds/uses its kernels through the backend contract
  only.

### `transpose` no-op contract
- Symmetric pair → result is elementwise-identical to the input (`T.transpose(i,j) == T`), produced via an
  identity **copy, not a permutation** (no restride — this is the "avoids a real transpose" payoff).
- Hermitian (complex) pair → result equals `conj(T)` elementwise; Hermitian (real) pair → equals `T`
  (conjugation is a no-op on reals).
- Antisymmetric pair → result equals `-T` elementwise.
- `transpose(i,j)` is identical to `transpose(j,i)` (symmetric in its arguments) for every bound pair.
- Non-bound pair (or out-of-range position) → throws `std::logic_error` with the documented message.

### GF2 linkage (ties to Stories 05 and 07)
- Build the GF2 input set with the new API: `hcore{ao,ao}=Hermitian()`, `mo_coeff{ao,mo}`=plain,
  `eri3{ri,ao,ao}=Symmetric()`, all `Tensor<double, Executor::Host>`; construct, `print()`, and assert the
  shapes/labels/symmetry queries above. (No numerics — just proves the physics-level construction is
  expressible and consistent with the loader from Story 05.)

### Definition of done
- `run_tensor_symmetry_tests()` wired into `run_tests()`; `./cppgw -t` runs the **full** suite (Stories
  2–5 + these) with no throws.
- `eri3`/`eri4`/`hcore` examples in the draft construct using identical labels + a shared `SymGroup`, as
  sketched above.
- `Tensor::transpose(i,j)` is a genuine no-op (identity / conjugate / negate) on symmetry-bound pairs and
  throws a clear "not yet implemented" otherwise — no physical permutation code is added.
- The `TensorBackend` seam is in place; host `gemm` + unary ops live behind it; BLAS/Eigen/cuTENSOR are
  **not** integrated (that is a future story that only adds backend implementations).
- All pre-existing tests (Story 2 grid/tensor/gemm, Story 3 Fourier, Story 4 inverse Fourier) still pass
  unchanged.

---

## Deferred (future stories)

- **Multi-member / transitive symmetry groups** (rank-4 `(μν|λσ)` double-exchange orbit) — the pairwise
  model here generalizes to them, but group closure + orbit reasoning is the hard part.
- **Active transpose / index-permutation primitive** for *non-identical* axes (reorder + stride
  recompute + storage move). Only the *no-op* case is implemented here.
- **Symmetry-aware kernels** (delegate to / exploit backend symmetries; e.g. cuTENSOR symmetry dispatch,
  Hermitian `gemm` shortcuts).
- **Symmetry propagation** to `gemm` outputs (derive the output's symmetry groups from the operands'
  groups) rather than having the caller declare them.
- **BLAS / Eigen / cuTENSOR** implementations of the `TensorBackend` contract (the "performant backend").
- **Data-validity checks** (assert a tensor actually satisfies its declared symmetry) — advisory only here.
