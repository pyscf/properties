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

"""
Infrared analysis for unrestricted self-consistent method (only work in real orbitals)
"""

from pyscf import hessian, lib
from pyscf.prop import infrared
from pyscf.prop.infrared.rhf import get_h2ao_dipderiv

import numpy as np


def proc_hessian_(mf_hess, mo_energy=None, mo_coeff=None, mo_occ=None, h1ao_grad=None):
    mf = mf_hess.base

    mo_energy = mo_energy if mo_energy else mf.mo_energy
    mo_coeff = mo_coeff if mo_coeff else mf.mo_coeff
    mo_occ = mo_occ if mo_occ else mf.mo_occ
    h1ao_grad = h1ao_grad if h1ao_grad else mf_hess.make_h1(mo_coeff, mo_occ)

    moao1_grad, mo_e1_grad = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_grad)

    hess_elec = mf_hess.hess_elec(mo1=moao1_grad, mo_e1=mo_e1_grad, h1ao=h1ao_grad)
    hess_nuc = mf_hess.hess_nuc()
    mf_hess.de = hess_elec + hess_nuc

    mo1_grad = [None] * 2
    for s in (0, 1):
        mo1_grad[s] = lib.einsum("up, uv, Axvi -> Axpi", mo_coeff[s], mf.get_ovlp(), moao1_grad[s])

    return mf_hess, h1ao_grad, mo1_grad, mo_e1_grad


def kernel_dipderiv(mf_ir):

    if mf_ir.mf_hess is None:
        mf_ir.mf_hess = mf_ir.hess_cls(mf_ir.base)
        # mf_ir.mf_hess.grid_response = True  # in case mf_hess is dft method, but seems no effect?
    if mf_ir.mf_hess.de is None \
            or np.abs(mf_ir.mf_hess.de).sum() < 1e-14 \
            or mf_ir._mo1_grad is NotImplemented:
        mf_hess, h1ao_grad, mo1_grad, mo_e1_grad = proc_hessian_(mf_ir.mf_hess)
        mf_ir._h1ao_grad = h1ao_grad
        mf_ir._mo1_grad = mo1_grad
        mf_ir._mo_e1_grad = mo_e1_grad
    mo1_grad = mf_ir._mo1_grad

    if mf_ir._h1_dip is NotImplemented:
        mf = mf_ir.base
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        int_r = mf.mol.intor_symmetric("int1e_r")
        h1_dip = [None] * 2
        for s in (0, 1):
            h1_dip[s] = lib.einsum("tuv, up, vi-> tpi", int_r, mo_coeff[s], mo_coeff[s][:, mo_occ[s] > 0])
        mf_ir._h1_dip = h1_dip
    h1_dip = mf_ir._h1_dip

    if mf_ir._h2ao_dipderiv is NotImplemented:
        mf_ir._h2ao_dipderiv = get_h2ao_dipderiv(mf_ir.base)
    h2ao_dipderiv = mf_ir._h2ao_dipderiv

    mol = mf_ir.mol
    natm = mol.natm
    de = np.zeros((natm, 3, 3))
    for a in range(natm):
        de[a] = np.eye(3) * mol.atom_charge(a)
    de += np.einsum("Axtuv, suv -> Axt", h2ao_dipderiv, mf_ir.base.make_rdm1())
    for s in (0, 1):
        de -= 2 * np.einsum("tpi, Axpi -> Axt", h1_dip[s], mo1_grad[s])

    mf_ir.de = de
    return de


class Infrared(infrared.rhf.Infrared):

    hess_cls = hessian.uhf.Hessian
    kernel_dipderiv = kernel_dipderiv


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0,
                   charge=1, spin=1).build()
    mf = scf.UHF(mol).run()
    # results from qchem:
    # 150.819  177.693  225.259  55.273  102.520  92.602
    mf_ir = Infrared(mf).run()
    mf_ir.summary()
    fig = mf_ir.plot_ir()[0]
    fig.show()
