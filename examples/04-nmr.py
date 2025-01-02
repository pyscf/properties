#!/usr/bin/env python

'''
Computing NMR shielding constants
'''

from pyscf import gto, dft
from pyscf.prop import nmr

# non-relativistic NMR shielding
mol = gto.M(atom='''
            C 0 0 0
            O 0 0 1.1747
            ''',
            basis='ccpvdz', verbose=3)
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.run()
mf.NMR().kernel()

# 4-component DKS NMR shielding
mf = mol.DKS(xc='b3lyp').run()
mf.NMR().kernel()
