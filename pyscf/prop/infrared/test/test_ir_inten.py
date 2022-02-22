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
