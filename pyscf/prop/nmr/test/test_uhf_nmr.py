import unittest
from pyscf import gto, lib
from pyscf import scf
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
    mf = scf.UHF(mol).run()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_nr_giao_cpscf(self):
        nmr = mf.NMR()
        nmr.cphf = True
        #nmr.gauge_orig = (0,0,0)
        msc = nmr.kernel()
        self.assertAlmostEqual(lib.fp(msc), -132.22895063293751, 5)

    def test_nr_common_gauge_cpscf(self):
        nmr = mf.NMR()
        nmr.cphf = True
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding()
        self.assertAlmostEqual(lib.fp(msc), -108.48532247186918, 5)

    def test_nr_giao_ucpscf(self):
        nmr = mf.NMR()
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding()
        self.assertAlmostEqual(lib.fp(msc), -133.26526049655627, 5)

    def test_nr_giao_ucpscf1(self):
        mol = gto.M(
            verbose=0,
            atom=[['H' , (0. , 0. , .917)],
                  ['F' , (0. , 0. , 0.)],
                  ['H' , (1. , 0.3, .417)],
                  ['H' , (0.2, 1. , 0.)],],
            nucmod = {'F': 2}, # gaussian nuclear model
            basis = '6-31g',
        )
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        nmr = mf.NMR()
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding()
        self.assertAlmostEqual(lib.fp(msc), -123.98599301453484, 3)


if __name__ == "__main__":
    print("Full Tests of UHF-NMR")
    unittest.main()
