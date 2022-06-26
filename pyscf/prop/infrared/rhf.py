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
Infrared analysis for restricted self-consistent method (only work in real orbitals)
"""

from pyscf import hessian, lib
from pyscf.hessian import thermo
from pyscf.hessian.rhf import Hessian
from pyscf.lib import logger

import numpy as np
from scipy.constants import physical_constants


def proc_hessian_(mf_hess, mo_energy=None, mo_coeff=None, mo_occ=None, h1ao_grad=None):
    """
    Process hessian evaluation for infrared analysis

    This function reuses hessian evaluation intermediate, mo_coeff rotation matrix (U matrix),
    to be further contracted with dipole tensor, without computing CP-HF equation a second time.
    """
    mf = mf_hess.base

    mo_energy = mo_energy if mo_energy else mf.mo_energy
    mo_coeff = mo_coeff if mo_coeff else mf.mo_coeff
    mo_occ = mo_occ if mo_occ else mf.mo_occ
    h1ao_grad = h1ao_grad if h1ao_grad else mf_hess.make_h1(mo_coeff, mo_occ)

    moao1_grad, mo_e1_grad = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_grad)

    hess_elec = mf_hess.hess_elec(mo1=moao1_grad, mo_e1=mo_e1_grad, h1ao=h1ao_grad)
    hess_nuc = mf_hess.hess_nuc()
    mf_hess.de = hess_elec + hess_nuc
    mo1_grad = lib.einsum("up, uv, Axvi -> Axpi", mo_coeff, mf.get_ovlp(), moao1_grad)
    return mf_hess, h1ao_grad, mo1_grad, mo_e1_grad


def get_h2ao_dipderiv(mf):
    """
    Generate core hamiltonian derivative of gradient-dipole

    Returns
    -------
    h2ao
        5-dim tensor: (atom number, atom coordinate components, dipole components, basis number, basis number)
    """
    mol = mf.mol
    natm, nao = mol.natm, mol.nao

    h2ao = np.zeros((natm, 3, 3, nao, nao))
    int1e_irp = mol.intor("int1e_irp").reshape(3, 3, nao, nao).swapaxes(0, 1)
    for a in range(natm):
        _, _, a1, a2 = mol.aoslice_by_atom()[a]
        h2ao[a, :, :, :, a1:a2] = int1e_irp[:, :, :, a1:a2]
    h2ao += h2ao.swapaxes(-1, -2)

    return h2ao


def kernel_dipderiv(mf_ir):
    """
    Dipole derivative main driver

    Dipole derivative is intermediate quantity for IR intensity evaluation.
    Note that if dipole derivative is the only purpose, then CP-HF equation wrt gradient rotation
    is not an efficient way, instead dipole rotation should make things easier. However, since
    hessian evaluation is inevitable when performing vibrational analysis and IR intensity,
    so we drop procedure that uses efficient CP-HF wrt dipole rotation.

    Returns
    -------
    de
        3-dim tensor: (atom number, atom coordinate components, dipole components)
    """
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
        orbo = mo_coeff[:, mo_occ > 0]
        int_r = mf.mol.intor_symmetric("int1e_r")
        h1_dip = lib.einsum("tuv, up, vi-> tpi", int_r, mo_coeff, orbo)
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
    de += np.einsum("Axtuv, uv -> Axt", h2ao_dipderiv, mf_ir.base.make_rdm1())
    de -= 4 * np.einsum("tpi, Axpi -> Axt", h1_dip, mo1_grad)

    mf_ir.de = de
    return de


def kernel_ir(mf_ir):
    """
    Infrared spectra intensity main driver

    Returns
    -------
    ir_inten
        1-dim vector; unit km/mol; same length to number of vibrational modes.

    See Also
    --------
    For unit conversion, also see
    https://github.com/psi4/psi4/blob/v1.5/psi4/driver/qcdb/vib.py#L589-L590
    """
    mol = mf_ir.mol
    if mf_ir.vib_dict is None:
        mf_ir.vib_dict = thermo.harmonic_analysis(mol, mf_ir.mf_hess.de)
    d = mf_ir.vib_dict
    q = d["norm_mode"].reshape(-1, mol.natm * 3)
    de = mf_ir.de.reshape(-1, 3)
    de_q = np.dot(q, de)

    alpha = physical_constants["fine-structure constant"][0]
    amu = physical_constants["atomic mass constant"][0]
    m_e = physical_constants["electron mass"][0]
    N_A = physical_constants["Avogadro constant"][0]
    a_0 = physical_constants["Bohr radius"][0]
    # unit_kmmol ~= 974.88011
    # another way is `(e_c * N_A)**2 / 10000000 * np.pi / 3`
    #     however, above formula uses deprecated definition of vacuum magnetic permeability value
    #     see also https://en.wikipedia.org/wiki/Vacuum_permeability
    unit_kmmol = alpha**2 * (1e-3 / amu) * m_e * N_A * np.pi * a_0 / 3

    ir_inten = unit_kmmol * np.einsum("qt, qt -> q", de_q, de_q)

    mf_ir.ir_inten = ir_inten
    return ir_inten


def ir_lorentz_broadening(v, v_0, w):
    """
    Lorentz boardening for infrared spectra at specific normal mode frequency

    Parameters
    ----------
    v : int
        Frequency to be plotted on spectrum. Unit in cm^-1.
    v_0 : int or np.ndarray
        Frequency of the normal mode to be studied. Unit in cm^-1.
    w : int
        FWHW of the broadening. Unit in cm^-1.

    Returns
    -------
    Broadening scaling coefficient. Dimensionless.

    See Also
    --------
    doi: 10.13140/RG.2.1.4181.6160
    """
    return 100 * 0.5 / np.log(10) / np.pi * w / ((v - v_0)**2 + 0.25 * w**2)


def ir_point(v, w, freq, ir_inten):
    """
    Lorentz boardening for infrared spectra

    Parameters
    ----------
    v : int
        Frequency to be plotted on spectrum. Unit in cm^-1.
    w : int
        FWHW of the broadening. Unit in cm^-1.
    freq : np.ndarray
        All vibrational frequencies. Unit in cm^-1.
    ir_inten : np.ndarray
        IR intensity for all vibrational frequencies.
        Dimension should be the same to `freq`.
        Unit in km/mol.
    """
    return (ir_lorentz_broadening(v, freq, w) * ir_inten).sum()


class Infrared(lib.StreamObject):
    """
    Main class of infrared intensity analysis

    Examples
    --------
    >>> mf = scf.RHF(mol).run()
    >>> mf_ir = Infrared(mf).run()
    >>> mf_ir.summary()

    Then you see IR intensity in standard output defined by `mf`.

    Attributes
    ----------
    mf_hess : hessian.rhf.Hessian
        For general user, leave this attribute to None.
        If a hessian object has been performed, and API user also stored `mo1_grad`,
        then by filling this attribute, hessian evaluation will not be performed a
        second time.
        After running infrared analysis, hessian tensor is stored in `mf_hess.de`.
    _mo1_grad : np.ndarray or list[np.ndarray]
        For general user, leave this attribute to None.
        Orbital rotation matrix wrt gradient in MO basis. Should be generated by CP-HF
        iteration. Also see `proc_hessian_`.
    vib_dict : dict
        For general user, leave this attribute to None.
        This should be auto-generated by `thermo.harmonic_analysis`. However, API user
        may cheat vibration analysis by modifing this dictionary, and consequently IR
        intensities could be also cheated.
    """

    hess_cls = hessian.rhf.Hessian

    def __init__(self, mf):
        mol = mf.mol
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self._scf = self.base = mf  # type: scf.hf.RHF

        self.mf_hess = None  # type: Hessian
        self.vib_dict = None  # type: dict

        self._h1ao_grad = NotImplemented
        self._mo_e1_grad = NotImplemented
        self._mo1_grad = NotImplemented
        self._h1_dip = NotImplemented
        self._h2ao_dipderiv = NotImplemented

        self.de = NotImplemented
        self.ir_inten = NotImplemented

    kernel_dipderiv = kernel_dipderiv
    kernel_ir = kernel_ir

    def kernel(self, *args, **kwargs):
        self.kernel_dipderiv()
        self.kernel_ir()
        return self.ir_inten

    def summary(self):
        log = logger.new_logger(self, 2)
        log.log(
            "------------------------------------------\n"
            " Mode      Frequency       Intensity      \n"
            "   #         cm^-1          km/mol        \n"
            "------------------------------------------\n")
        for i, (f, inten) in enumerate(zip(self.vib_dict["freq_wavenumber"], self.ir_inten)):
            flag_im = np.imag(f) > 1e-10
            chr_im = "i" if flag_im else " "
            log.log("{:5d}   {:12.4f}{:}   {:12.4f}".format(i, np.abs(f), chr_im, inten))
        log.log("------------------------------------------\n")

    def plot_ir(self, w=50, x=None, scale=1):
        """
        Plot IR spectrum by matplotlib

        Parameters
        ----------
        w : int
            FWHW of the broadening. Unit in cm^-1.
        x : np.ndarray
            Frequencies (x-axis) sampling points. Unit in cm^-1.
        scale : float
            Vibration frequency scaling factor. Should be chosen by method/basis combination.
            Also see https://cccbdb.nist.gov/vibscalejust.asp
        """
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid()

        freq = self.vib_dict["freq_wavenumber"] * scale
        ir_inten = self.ir_inten.copy()
        ir_inten[np.abs(np.imag(freq)) > 1e-10] = 0
        freq = np.real(freq)
        x = x if x else np.linspace(0, np.max([4000., np.max(freq) + 5*w]), 4000)
        ax.plot(x, [ir_point(xi, w, freq, ir_inten) for xi in x], label="Molar Absorption Coefficient")
        ax.set_ylabel("Molar Absorption Coefficient (L mol$^{-1}$ cm$^{-1}$)")
        ax.set_xlabel("Vibration Wavenumber (cm$^{-1}$)")
        ax.legend(loc="upper left")

        ax2 = ax.twinx()
        for i in range(ir_inten.size):
            if i == 0:
                ax2.plot([freq[i], freq[i]], [0, ir_inten[i]], c="C2", linewidth=1, label="IR Intensity")
            else:
                ax2.plot([freq[i], freq[i]], [0, ir_inten[i]], c="C2", linewidth=1)
        ax2.set_ylabel("IR Intensity (km mol$^{-1}$)")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        return fig, ax, ax2


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.Mole(atom="N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2", basis="6-31G", verbose=0).build()
    mf = scf.RHF(mol).run()
    # results from qchem:
    # 23.708  24.939  12.931  75.893  25.018  0.103
    mf_ir = Infrared(mf).run()
    mf_ir.summary()
    fig = mf_ir.plot_ir()[0]
    fig.show()

