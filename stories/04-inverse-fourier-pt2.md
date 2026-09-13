### Goal of Story:
Implementing Bosonic tail subtraction

Refactor Fermionic tail subtraction into a general MatsubaraTailSubtraction for the InverseFourierTransform direction.
Bosonic functions can have a high-frequency tail expansion too, even though c_1=0 by construction.
This arises from discontinuities in the derivatives.
