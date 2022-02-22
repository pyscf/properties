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
#
# Author: ajz34 <ajz34@outlook.com>
#

from pyscf.prop import infrared
from pyscf import hessian


class Infrared(infrared.uhf.Infrared):

    hess_cls = hessian.uks.Hessian


if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0,
                   charge=1, spin=1).build()
    mf = dft.UKS(mol, xc="PBE0").run()
    # results from qchem:
    # 129.094  162.634  179.495  50.924  79.706  79.964
    mf_ir = Infrared(mf).run()
    mf_ir.summary()
    fig = mf_ir.plot_ir()[0]
    fig.show()
