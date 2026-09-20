#!/usr/bin/env python
"""
Run a density-fitted RHF calculation in PySCF and save the key quantities
to an HDF5 file:
    - eri_3c    : 3-index density-fitted ERI tensor (P|mu nu), AO basis,
                  shape (naux, nao, nao)
    - mo_coeff  : MO coefficients, shape (nao, nmo)
    - dm        : AO density matrix, shape (nao, nao)
    - mo_energy : MO orbital energies, shape (nmo,)
    - n_elec    : number of electrons (scalar)
    - hcore     : core Hamiltonian (kinetic + nuclear attraction), AO basis,
                  shape (nao, nao)
"""

import h5py
import numpy as np
from pyscf import gto, scf, lib

# ------------------------------------------------------------------
# 1. Build the molecule (edit geometry / basis / charge / spin as needed)
# ------------------------------------------------------------------
mol = gto.M(
    atom='''
        O  0.000000  0.000000  0.117790
        H  0.000000  0.755453 -0.471161
        H  0.000000 -0.755453 -0.471161
    ''',
    basis='cc-pvdz',
    unit='Angstrom',
    verbose=4,
)

# ------------------------------------------------------------------
# 2. Run RHF with density fitting
# ------------------------------------------------------------------
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf.conv_tol = 1e-10
energy = mf.kernel()

if not mf.converged:
    raise RuntimeError("SCF did not converge!")

print(f"RHF (density-fitted) energy: {energy:.10f} Ha")

# ------------------------------------------------------------------
# 3. Collect the quantities to save
# ------------------------------------------------------------------
mo_coeff  = mf.mo_coeff          # AO -> MO coefficients, shape (nao, nmo)
mo_energy = mf.mo_energy         # MO energies, shape (nmo,)
dm        = mf.make_rdm1()       # AO density matrix, shape (nao, nao)
n_elec    = mol.nelectron        # number of electrons
hcore     = mf.get_hcore()       # core Hamiltonian, AO basis, shape (nao, nao)

# 3-center density-fitted ERIs (P|mu nu) in the AO basis.
# with_df.loop() yields blocks packed over the lower-triangle of AO pairs;
# unpack each block to the full square (nao, nao) form.
with_df = mf.with_df
nao = mol.nao_nr()
naux = with_df.get_naoaux()

eri_3c = np.empty((naux, nao, nao))
p0 = 0
for block in with_df.loop():
    p1 = p0 + block.shape[0]
    eri_3c[p0:p1] = lib.unpack_tril(block).reshape(p1 - p0, nao, nao)
    p0 = p1

# ------------------------------------------------------------------
# 4. Save everything to an HDF5 file
# ------------------------------------------------------------------
out_file = 'rhf_df_data.h5'
with h5py.File(out_file, 'w') as f:
    f.create_dataset('eri_3c', data=eri_3c)
    f.create_dataset('mo_coeff', data=mo_coeff)
    f.create_dataset('dm', data=dm)
    f.create_dataset('mo_energy', data=mo_energy)
    f.create_dataset('n_elec', data=n_elec)
    f.create_dataset('hcore', data=hcore)
    f.attrs['scf_energy'] = energy
    f.attrs['naux'] = naux
    f.attrs['nao'] = nao

print(f"Saved RHF/DF results to {out_file}")
