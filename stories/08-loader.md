# Story 08 — Structured HDF5 calculation-data Loader (stub)

> **Status:** Deferred design stub. This story is intentionally not implemented
> as part of Story 07.1. It exists to prevent incremental GF2 stories from each
> growing their own HDF5-reading and dataset-validation path.

## Motivation

Story 05 introduced a deliberately small concrete route:

```text
ParsedInput → load_gf2(...) → Gf2Input
```

Story 07.1 narrowly extends that route with the AO overlap, density matrix, and
GF2 initial-guess scalars. That is the right short-term choice: it reuses one
parser, one HDF5 opening/validation path, and one calculation input object.

It is not, however, a scalable representation for all future GF2 stages, GW,
multiple basis choices, optional restart data, or richer HDF5 file schemas.
This follow-up should introduce a structured Loader only after the required
calculation data and conventions have stabilized.

## Intended goal

Design and implement one calculation-data loading boundary that:

- owns HDF5 file access and dataset validation;
- maps registry-selected HDF5 paths to typed, labelled `Tensor` objects;
- validates dimensions and cross-dataset consistency once;
- produces calculation-stage-specific typed data bundles;
- keeps HDF5/HighFive details out of GF2 numerical kernels; and
- reuses Story 05's central `ParsedInput` registry rather than creating a
  competing configuration system.

The Loader is a data-ingestion abstraction. It is not a physics basis
transformation abstraction, a lazy tensor engine, or a general HDF5 ORM.

## Proposed direction (not final API)

A future design may resemble:

```cpp
class CalculationDataLoader {
public:
  CalculationDataLoader(const ParsedInput& input, const std::string& h5_path);

  // Typed data sets required by individual calculation stages.
  Gf2StaticInput load_gf2_static_input();
  Gf2InitialGuessInput load_gf2_initial_guess_input();
  // GwInput load_gw_input();  // future
};
```

The exact names and bundle boundaries must be chosen only after deciding which
objects are stable shared inputs versus iteration/restart outputs. It should
not preserve `Gf2Input` merely for compatibility if a clearer set of typed
bundles emerges.

A Loader should centralize reusable operations such as:

- opening an HDF5 file once with clear ownership/lifetime;
- locating a dataset from a parsed path;
- checking dataset existence, scalar `float64` type, rank, and extents;
- direct contiguous reads into the established fastest-to-slowest `Tensor`
  layouts;
- creation of `TensorDim` labels and advisory `SymGroup` metadata;
- cross-tensor AO/MO/RI extent validation; and
- error messages naming calculation field, HDF5 path, expected shape/type, and
  observed shape/type.

It must retain Story 06's boundary: labels and advisory symmetry belong to the
Tensor frontend/data model, while HDF5 itself and numerical backends do not
interpret them.

## Explicit non-goals

- Do not implement this class in Story 07.1.
- Do not replace the Story-05 parser or duplicate its keyword/default/required
  registry.
- Do not add arbitrary HDF5 schema discovery, reflection, or automatic dataset
  name guessing.
- Do not add lazy on-demand tensors whose buffers depend on an open HDF5 file
  lifetime unless a later story explicitly needs and designs that behavior.
- Do not silently transpose datasets based only on label names. Axis-order
  adaptation must use explicit metadata and Story 06.1's physical permutation
  primitive.
- Do not infer mathematical symmetry from numerical data or use it to compress
  storage.
- Do not mix loading with GF2 self-consistency, chemical-potential search,
  Fourier transforms, or numerical kernel dispatch.

## Migration expectation

When implemented, this story should first characterize and preserve successful
Story-05/07.1 loading behavior with integration tests. It may then move the
concrete `load_gf2(...)` internals behind the Loader and keep a compatibility
wrapper during migration. There must remain exactly one authoritative HDF5
validation/read implementation for any given field.

The migration should be behavior-preserving for:

- case-sensitive HDF5 paths selected through case-insensitive input keywords;
- `float64` and shape validation;
- direct contiguous layout reads;
- AO/RI/MO cross-dimension checks;
- existing advisory symmetry handles; and
- actionable failures for missing or malformed data.

## Definition of done (future)

- A documented Loader API produces typed bundles for at least the GF2 static
  and initial-guess stages.
- Story-05 and Story-07.1 HDF5 loading code is consolidated rather than
  duplicated.
- Loader unit/integration tests cover all migrated datasets and failure paths.
- Existing CLI and `-t` behavior remain unchanged.
- Numerical GF2 code receives typed tensors/scalars and contains no HighFive
  calls or raw dataset-shape parsing.
