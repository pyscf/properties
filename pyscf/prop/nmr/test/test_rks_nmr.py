import unittest
import numpy as np
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf.prop import nmr
from pyscf.data import nist

def setUpModule():
    global mol, mf
    mol = gto.M(
        verbose=7,
        output='/dev/null',
        atom=[['H' , (0. , 0. , .917)],
              ['F' , (0. , 0. , 0.)], ],
        nucmod = {'F': 2}, # gaussian nuclear model
        basis = '6-31g',
    )
    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mf = dft.RKS(mol).run()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_he(self):
        mol = gto.M(atom='Ne', basis='6-31g', verbose=0)
        mf = dft.RKS(mol).run()
        nmr = mf.NMR()
        msc = nmr.kernel()
        self.assertAlmostEqual(abs(msc - np.eye(3) * 551.315475).max(), 0, 5)

    def test_nr_giao_cpscf(self):
        nmr = mf.NMR()
        msc = nmr.kernel()
        self.assertAlmostEqual(msc[1][0,0], 368.881240, 5)
        self.assertAlmostEqual(msc[1][1,1], 368.881240, 5)
        self.assertAlmostEqual(msc[1][2,2], 482.413298, 5)
        self.assertAlmostEqual(lib.fp(msc), -131.708525548098, 5)

    def test_nr_giao_cpscf1(self):
        mol = gto.M(
            verbose=7,
            output='/dev/null',
            atom=[['H' , (0. , 0. , .917)],
                  ['F' , (0. , 0. , 0.)], ],
            nucmod = {'F': 2}, # gaussian nuclear model
            basis = 'ccpvdz',
        )
        mf = mol.RKS(xc='b3lyp').run()
        nmr = mf.NMR()
        msc = nmr.kernel()
        self.assertAlmostEqual(msc[1][0,0], 387.083740, 5)
        self.assertAlmostEqual(msc[1][1,1], 387.083740, 5)
        self.assertAlmostEqual(msc[1][2,2], 482.217408, 5)
        self.assertAlmostEqual(lib.fp(msc), -132.274436056168, 5)


if __name__ == "__main__":
    print("Full Tests of RKS-NMR")
    unittest.main()
