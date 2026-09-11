### Background
This story is designed to implement the Matsubara to Tau grid direction of FourierTransform.
So far, the Tensor-valued-representation and FourierTransform machinery has been agnostic to statistics: any tensor could be expanded on the τ-interval [0,β] and transformed to Matsubara frequencies.
However, implementing the inverse fourier transform, from Matsubara frequencies back to τ, depends on the physical symmetries of the function/tensor.
From now on, it will be important to distinguish whether a Tensor quantity is fermionic or bosonic, which determines the boundary conditions on the endpoints of the [0,β] interval.
Fermionic tensors have a jump discontinuity: f(τ + β) = -f(τ), while Bosonic tensors are smooth: f(τ + β) = f(τ)

Bosonic tensors can be transformed back using the obvious inverse transform:
```text
G(τ) = 1/β sum_n e^{-i ω_n τ} G(i ω_n)
```
However, Fermionic tensors have a high-frequency component that will lead to catastrophic numerical convergence issues.
Thus, it's necessary to subtract this high-frequency tail before the transformation:
```text
G(τ) = 1/β sum_n e^{-i ω_n τ} (G(i ω_n) - G_{tail}(i ω_n)) + G_{tail}(τ)
```
where G_{tail} is the high-frequency expansion:
```text
G_{tail}(i ω_n) = c_1/(i ω_n) + c_2/(i ω_n)^2 + c_3/(i ω_n)^3 + ...
```

The coefficients c_n are given by
```text
c_{k+1} = (-1)^{k+1} (G^(k)(0+) + G^(k)(β-))
```
where G^(k) is the kth derivative of G.

The corresponding powers of inverse (i ω_n) have an analytic form in τ-space: the first few powers are these polynomials in β and τ:
```text
T_1(τ) = -1/2
T_2(τ) = -1/4*β + 1/2*τ
T_3(τ) = 1/2(1/2*βτ - 1/2*τ^2)
T_4(τ) = 1/6(1/8*β^3 - 3/4*βτ^2 + 1/2*τ^3)
T_5(τ) = 1/24(-1/2*β^3τ + βτ^3 - 1/2*τ^4)
T_6(τ) = 1/120(-1/4*β^5 + 5/4*β^3τ^2 - 5/4*βτ^4 + 1/2*τ^5)
```
so the corresponding function in τ-space is 
```text
G_{tail}(τ) = sum_{k} c_k*T_k(τ)
```

So, the inverse Fourier transform boils down to evaluating higher and higher derivatives of the Tensor function (G), e.g. for a 6th order expansion, the 7th derivative is needed.
This is where a different approach for each grid/basis set needs to be implemented.

For grids, fitting a local polynomial through the nearest M points (start with ~8, but keep variable) and differentiating it analytically is always a decent approach.

This story focuses on implementing the Tensor boundary conditions on the [0,β] via a StatisticsTag, implementing the Fermionic Tensor tail in both matsubara and imaginary time space through the equations above.



### Objectives
1. TensorExpansions gain a StatisticsTag that determine boundary conditions on ImaginaryTime
2. Implement a new Expansion for the high frequency tail of Fermionic Tensors that stores the coefficients c_k defined above for a particular Tensor
3. Implement the polynomial fitting procedure for a general grid in the neighborhood of the (wrapped) boundary, with functionality for evaluating derivatives analytically
4. Implement functionality to calculate the coefficients for a tail expansion through the polynomial fit on a general grid
5. Implement tests for this Story in a new source file and hook it into the main test suite

### Further notes
1. This Story is not an infrastructure addition (except for maybe the Tensor statistics), it should fall entirely into existing infrastructure. If there is a discrepancy or missing feature that this Story requires, advise before continuing.






