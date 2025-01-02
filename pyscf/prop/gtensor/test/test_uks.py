#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy
import copy
from pyscf import gto, lib, scf, dft
from pyscf.prop import gtensor
from pyscf.data import nist

def setUpModule():
    global mol, mf, dm0, dm1, ALPHA_backup
    ALPHA_backup = nist.ALPHA
    nist.ALPHA = 1./137.03599967994

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''
        H  0. , 0. , .917
        F  0. , 0. , 0.'''
    mol.basis = 'ccpvdz'
    mol.spin = 1
    mol.charge = 1
    mol.build()

    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf.conv_tol_grad = 1e-6
        mf.conv_tol = 1e-12
        mf.kernel()

    nao = mol.nao_nr()
    numpy.random.seed(1)
    dm0 = numpy.random.random((2,nao,nao))
    dm0 = dm0 + dm0.transpose(0,2,1)
    dm0 = numpy.einsum('xpi,xqi->xpq', dm0, dm0, optimize=True)
    dm1 = numpy.random.random((2,3,nao,nao))
    dm1 = dm1 - dm1.transpose(0,1,3,2)

def tearDownModule():
    global mol, mf, dm0, dm1, ALPHA_backup
    mol.stdout.close()
    del mol, mf, dm0, dm1
    nist.ALPHA = ALPHA_backup

class KnowValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_nr_lda_para_soc2e(self):
        mf1 = copy.copy(mf)
        mf1.xc = 'lda,vwn'
        g = gtensor.uks.GTensor(mf1)
        g.para_soc2e = 'SSO'
        dat = g.make_para_soc2e(dm0, dm1, 1)
        self.assertAlmostEqual(lib.fp(dat), -0.201859494, 9)

    def test_nr_bp86_para_soc2e(self):
        mf1 = copy.copy(mf)
        mf1.xc = 'bp86'
        g = gtensor.uks.GTensor(mf1)
        g.para_soc2e = 'SSO'
        dat = g.make_para_soc2e(dm0, dm1, 1)
        self.assertAlmostEqual(lib.fp(dat), -0.201794075, 7)

    def test_nr_b3lyp_para_soc2e(self):
        mf1 = copy.copy(mf)
        mf1.xc = 'b3lyp'
        g = gtensor.uks.GTensor(mf1)
        g.para_soc2e = 'SSO'
        dat = g.make_para_soc2e(dm0, dm1, 1)
        self.assertAlmostEqual(lib.fp(dat), -0.207296515, 7)

    def test_nr_uks(self):
        g = gtensor.uhf.GTensor(mf)
        g.dia_soc2e = None
        g.para_soc2e = 'SSO+SOO'
        g.so_eff_charge = True
        g.cphf = False
        dat = g.kernel()
        self.assertAlmostEqual(numpy.linalg.norm(dat), 3.4748005, 5)


if __name__ == "__main__":
    print("Full Tests for DFT g-tensor")
    unittest.main()
