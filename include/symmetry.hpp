#pragma once
#include <cstddef>
#include <memory>
#include <ostream>

namespace cppgw {

// ============================================================================
//  IndexSymmetry (Story 06)
// ----------------------------------------------------------------------------
// Pairwise symmetry *groups* over the axes of a Tensor. A group is a value
// (its SymKind) whose **identity** is what matters, not its value: two tensor
// axes are considered interchangeable iff they were constructed with the SAME
// SymmetryGroup object (handle identity). A `Symmetric` group g1 and a
// `Symmetric` group g2 are TWO distinct groups, so a rank-4 tensor can carry
// two independent symmetric pairs.
//
// Symmetry is ADVISORY (Story 06, decision Q3):
//   * it names the axis family in the Tensor frontend (user-facing API),
//   * it lets the frontend skip work that mathematically reduces to a no-op
//     (e.g. Tensor::transpose on a bound pair),
// and it is:
//   * never passed to the TensorBackend (kernels stay symmetry-agnostic),
//   * never used to compress storage (every Tensor stays dense and full).
// ============================================================================

// The kind of relation a pairwise symmetry group expresses.
enum class SymKind { Symmetric, Hermitian, Antisymmetric };

inline const char* sym_kind_name(SymKind k) {
  switch (k) {
    case SymKind::Symmetric:     return "symmetric";
    case SymKind::Hermitian:     return "hermitian";
    case SymKind::Antisymmetric: return "antisymmetric";
  }
  return "unknown-symmetry";
}

// A pairwise symmetry "group": a value (kind) whose IDENTITY is what matters.
// Two tensor axes are considered interchangeable iff they were built with the
// SAME SymmetryGroup instance (the same SymGroup handle / underlying object).
// Value-comparison is deliberately NOT meaningful for group identity: equality
// is by handle (see `same_symgroup` below).
class SymmetryGroup {
public:
  explicit SymmetryGroup(SymKind k) : kind_(k) {}
  SymKind kind() const noexcept { return kind_; }

private:
  SymKind kind_;
};

// The handle TensorDim stores. "Same group" == same underlying SymmetryGroup
// object. `nullptr` means "no group" (a plain axis).
using SymGroup = std::shared_ptr<const SymmetryGroup>;

// Ergonomic factories. Each call creates a FRESH group, so two calls yield
// distinct (non-interchangeable) families even for the same kind.
inline SymGroup Symmetric()     { return std::make_shared<const SymmetryGroup>(SymKind::Symmetric); }
inline SymGroup Hermitian()     { return std::make_shared<const SymmetryGroup>(SymKind::Hermitian); }
inline SymGroup Antisymmetric() { return std::make_shared<const SymmetryGroup>(SymKind::Antisymmetric); }

// Handle-based identity + null checks. `nullptr == nullptr` is true, so
// compare against a real handle only when at least one side is non-null.
inline bool same_symgroup(const SymGroup& a, const SymGroup& b) { return a == b; }
inline bool null_symgroup(const SymGroup& a)                     { return a == nullptr; }

inline std::ostream& operator<<(std::ostream& os, SymKind k)    { return os << sym_kind_name(k); }
inline std::ostream& operator<<(std::ostream& os, const SymGroup& g) {
  return g ? (os << "SymmetryGroup<" << g->kind() << ">") : (os << "(no symmetry)");
}

} // namespace cppgw
