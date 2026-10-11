#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyscf import gto, scf, dft
from pyscf.prop import infrared

import numpy as np
import unittest


def dipderiv_fd(mf, step=1e-3):
    # central differences of the SCF dipole moment; shape (atom, coordinate, dipole component) as kernel_dipderiv
    mol = mf.mol
    coords = mol.atom_coords()
    dm0 = mf.make_rdm1()
    de = np.zeros((mol.natm, 3, 3))
    for a in range(mol.natm):
        for x in range(3):
            dip = []
            for s in (step, -step):
                c = coords.copy()
                c[a, x] += s
                mf1 = mf.copy().reset(mol.set_geom_(c, unit="Bohr", inplace=False))
                mf1.kernel(dm0=dm0)
                dip.append(mf1.dip_moment(unit="AU", verbose=0))
            de[a, x] = (dip[0] - dip[1]) / (2 * step)
    return de


class KnownValues(unittest.TestCase):

    def test_rhf(self):
        # compare results from qchem
        mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0).build()
        mf = scf.RHF(mol).run()
        mf_ir = infrared.rhf.Infrared(mf).run()
        ref_value = [23.708, 24.939, 12.931, 75.893, 25.018, 0.103]
        self.assertTrue(np.allclose(mf_ir.ir_inten, ref_value, rtol=5e-4, atol=1e-2))

    def test_uhf(self):
        mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0,
                       charge=1, spin=1).build()
        mf = scf.UHF(mol).run()
        mf_ir = infrared.uhf.Infrared(mf).run()
        ref_value = [150.819, 177.693, 225.259, 55.273, 102.520, 92.602]
        self.assertTrue(np.allclose(mf_ir.ir_inten, ref_value, rtol=5e-4, atol=1e-2))

    def test_rks(self):
        # compare results from qchem
        mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0).build()
        mf = dft.RKS(mol, xc="PBE0").run()
        mf_ir = infrared.rks.Infrared(mf).run()
        ref_value = [13.329, 10.066, 5.686, 57.957, 30.175, 4.158]
        self.assertTrue(np.allclose(mf_ir.ir_inten, ref_value, rtol=5e-4, atol=1e-2))

    def test_uks(self):
        mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0,
                       charge=1, spin=1).build()
        mf = dft.UKS(mol, xc="PBE0")
        mf.grids.atom_grid = (99, 590)
        mf.run()
        mf_ir = infrared.uks.Infrared(mf).run()
        ref_value = [129.094, 162.634, 179.495, 50.924, 79.706, 79.964]
        # TODO: this criteria is a little bit loose ...
        # could be grid setting, grid response, atomic mass, ...
        self.assertTrue(np.allclose(mf_ir.ir_inten, ref_value, rtol=1e-3, atol=1e-2))

    def test_dipderiv_finite_difference(self):
        # dipole derivatives against central differences of the SCF dipole, no external reference;
        # observed (pyscf 2.14): RHF 5e-6, UHF 9e-6, RKS 8e-5, UKS 3e-4. The DFT difference has the size of the
        # translation sum rule's residual: the analytic DFT terms leave out the grid response.
        cases = ((scf.RHF, 0, 0, infrared.rhf, 5e-5), (scf.UHF, 1, 1, infrared.uhf, 5e-5),
                 (lambda m: dft.RKS(m, xc="PBE0"), 0, 0, infrared.rks, 1e-3),
                 (lambda m: dft.UKS(m, xc="PBE0"), 1, 1, infrared.uks, 1e-3))
        for make, charge, spin, mod, atol in cases:
            mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0,
                           charge=charge, spin=spin).build()
            mf = make(mol)
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-8
            mf.run()
            de = mod.Infrared(mf).kernel_dipderiv()
            self.assertTrue(np.allclose(de, dipderiv_fd(mf), rtol=0, atol=atol))
            self.assertTrue(np.allclose(de.sum(axis=0), charge * np.eye(3), rtol=0, atol=atol))

    # example input card for qchem (test_uks)
    """
    $molecule
    1 2
    N    0.0 0.0 0.0
    H    0.8 0.0 0.0
    H    0.0 1.0 0.0
    H    0.0 0.0 1.2
    
    $end
    
    $rem
    JOBTYPE   freq
    METHOD    PBE0
    BASIS     6-31G
    SCF_CONVERGENCE 8
    XC_GRID 000099000590
    $end
    """
