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
    mf = scf.RHF(mol).run()

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
        self.assertAlmostEqual(msc[1][0,0], 375.2331965380235, 5)
        self.assertAlmostEqual(msc[1][1,1], 375.2331965380235, 5)
        self.assertAlmostEqual(msc[1][2,2], 483.0020662335623, 5)
        self.assertAlmostEqual(lib.fp(msc), -132.22895063293751, 5)

    def test_nr_common_gauge_cpscf(self):
        nmr = mf.NMR()
        nmr.cphf = True
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding()
        self.assertAlmostEqual(msc[1][0,0], 342.4476992768218, 5)
        self.assertAlmostEqual(msc[1][1,1], 342.4476992768218, 5)
        self.assertAlmostEqual(msc[1][2,2], 483.0020662335623, 5)
        self.assertAlmostEqual(lib.fp(msc), -108.48532247186918, 5)

    def test_nr_giao_ucpscf(self):
        nmr = mf.NMR()
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding()
        self.assertAlmostEqual(msc[1][0,0], 449.0322403351828, 5)
        self.assertAlmostEqual(msc[1][1,1], 449.0322403351828, 5)
        self.assertAlmostEqual(msc[1][2,2], 483.0020662335623, 5)
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
        mf = scf.RHF(mol).run()
        nmr = mf.NMR()
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding()
        self.assertAlmostEqual(msc[1][0,0], 283.514603973907, 5)
        self.assertAlmostEqual(msc[1][1,1], 292.57812554130226, 5)
        self.assertAlmostEqual(msc[1][2,2], 257.34817883746473, 5)
        self.assertAlmostEqual(lib.fp(msc), -123.98599301453484, 5)


if __name__ == "__main__":
    print("Full Tests of RHF-NMR")
    unittest.main()
