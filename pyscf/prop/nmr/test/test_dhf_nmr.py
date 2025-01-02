import unittest
import numpy as np
from pyscf import gto, lib
from pyscf import scf
from pyscf.prop import nmr
from pyscf.data import nist

class KnownValues(unittest.TestCase):
    def test_rmb_cpscf(self):
        mol = gto.M(
            verbose=7,
            output='/dev/null',
            atom = 'He 0.,0.,0.',
            basis = {
            'He': [(0, 0, (1., 1.)),
                   (0, 0, (3., 1.)),
                   (1, 0, (1., 1.)), ]}
        )
        mf = scf.dhf.UHF(mol).run()
        nmr = mf.NMR()
        nmr.mb = 'RMB'
        nmr.cphf = True
        msc = nmr.shielding()
        self.assertAlmostEqual(abs(msc - np.eye(3)*64.4318).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests of DHF-NMR")
    unittest.main()
