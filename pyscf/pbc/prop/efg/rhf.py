#!/usr/bin/env python
# Copyright 2017-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Electric field gradients, nuclear quadrupolar coupling and Mossbauer
spectroscopy for non-relativistic (or sf-x2c) mean-field and post-HF methods.
See also pyscf/prop/efg/rhf.py

Ref:

[1] H. Petrilli, P. Blochl, P. Blaha, and K. Schwarz. Phys. Rev. B, 57, 14690 (1998)

[2] H. Akai et al. Prog. Theor. Phys. Suppl. 101, 11 (1990)
'''


import numpy as np
import scipy.special
from pyscf import lib
from pyscf.gto.mole import gaussian_int
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf
from pyscf.pbc import tools as pbctools
from pyscf.pbc import df
from pyscf.pbc.df import aft
from pyscf.pbc.df.gdf_builder import estimate_eta_for_ke_cutoff
from pyscf.prop.efg import rhf as rhf_efg

def kernel(method, efg_nuc=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    cell = method.cell
    if efg_nuc is None:
        efg_nuc = range(cell.natm)

    dm = method.make_rdm1()
    if isinstance(method, scf.khf.KSCF):
        if isinstance(dm[0][0], np.ndarray) and dm[0][0].ndim == 2:
            dm = dm[0] + dm[1]  # KUHF density matrix
    elif isinstance(method, scf.hf.SCF):
        if isinstance(dm[0], np.ndarray) and dm[0].ndim == 2:
            dm = dm[0] + dm[1]  # UHF density matrix
    else:
        mo = method.mo_coeff
        if isinstance(dm[0][0], np.ndarray) and dm[0][0].ndim == 2:
            dm_a = [lib.einsum('pi,ij,qj->pq', c, dm[0][k], c.conj())
                    for k, c in enumerate(mo)]
            dm_b = [lib.einsum('pi,ij,qj->pq', c, dm[1][k], c.conj())
                    for k, c in enumerate(mo)]
            dm = lib.asarray(dm_a) + lib.asarray(dm_b)
        else:
            dm = lib.asarray([lib.einsum('pi,ij,qj->pq', c, dm[k], c.conj())
                              for k, c in enumerate(mo)])

    if isinstance(method, scf.hf.SCF):
        with_df = getattr(method, 'with_df', None)
        with_x2c = getattr(method, 'with_x2c', None)
    else:
        with_df = getattr(method._scf, 'with_df', None)
        with_x2c = getattr(method._scf, 'with_x2c', None)
    if with_x2c:
        raise NotImplementedError

    log.info('\nElectric Field Gradient Tensor Results')
    if isinstance(with_df, df.fft.FFTDF):
        efg_e = _fft_quad_integrals(with_df, dm, efg_nuc)
    else:
        efg_e = _aft_quad_integrals(with_df, dm, efg_nuc)
    efg = []
    for i, atm_id in enumerate(efg_nuc):
        efg_nuc = _get_quad_nuc(cell, atm_id)
        v = efg_nuc - efg_e[i]
        efg.append(v)

        rhf_efg._analyze(cell, atm_id, v, log)

    return np.asarray(efg)

EFG = kernel


def _get_quad_nuc(cell, atm_id):
    ew_eta, ew_cut = cell.get_ewald_params()
    chargs = cell.atom_charges()
    coords = cell.atom_coords()
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    rLij = coords[atm_id,:] - coords + Lall[:,None,:]
    rr = np.einsum('Ljx,Ljy->Ljxy', rLij, rLij)
    r = np.sqrt(np.einsum('Ljxx->Lj', rr))
    r[r<1e-16] = 1e60
    idx = np.arange(3)
    erfc_part = scipy.special.erfc(ew_eta * r) / r**5
    ewovrl = 3 * chargs[atm_id] * np.einsum('Ljxy,j,Lj->xy', rr, chargs, erfc_part)
    ewovrl[idx,idx] -= ewovrl.trace() / 3

    exp_part = np.exp(-ew_eta**2 * r**2) / r**4
    ewovrl_part = (2./np.sqrt(np.pi) * ew_eta * chargs[atm_id] *
                   np.einsum('Ljxy,j,Lj->xy', rr, chargs, exp_part))
    ewovrl += ewovrl_part

    ewovrl += 2*ewovrl_part
    ewovrl[idx,idx] -= ewovrl_part.trace()

    exp_part = np.exp(-ew_eta**2 * r**2) / r**2
    ewovrl += (4./np.sqrt(np.pi) * ew_eta**3 * chargs[atm_id] *
               np.einsum('Ljxy,j,Lj->xy', rr, chargs, exp_part))
    # Fermi contact term
    ewovrl_fc = 4./3*ew_eta**3/np.pi**.5 * np.exp(-ew_eta**2*r**2)
    ewovrl[idx,idx] -= np.einsum('j,Lj->', chargs, ewovrl_fc)

    log_precision = np.log(cell.precision / (chargs.sum()*16*np.pi**2))
    ke_cutoff = -2*ew_eta**2*log_precision
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    GG = np.einsum('gx,gy->gxy', Gv, Gv)
    absG2 = np.einsum('gxx->g', GG)
    # Corresponding to FC term, that makes the tensor zero trace
    idx = np.arange(3)
    GG[:,idx,idx] -= 1./3 * absG2[:,None]

    absG2[absG2==0] = 1e200
    coulG = 4*np.pi / absG2
    coulG *= weights
    expG2 = np.exp(-absG2/(4*ew_eta**2))
    coulG *= expG2
    SI = cell.get_SI(Gv)
    ZSI = np.einsum("i,ij->j", chargs, SI)
    ewg =-chargs[atm_id] * np.einsum('ixy,i,i,i->xy',
                                        GG, SI[atm_id].conj(), ZSI, coulG).real
    return ewovrl + ewg


def _fft_quad_integrals(mydf, dm, efg_nuc):
    # Use FFTDF to compute the integrals of quadrupole operator
    # (3 \vec{r} \vec{r} - r^2) / r^5
    cell = mydf.cell
    if cell.dimension != 3:
        raise NotImplementedError

    mesh = mydf.mesh
    kpts = mydf.kpts
    kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    dm_kpts = dm.reshape((nkpts,nao,nao), order='C')

    ni = mydf._numint
    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm_kpts, hermi)
    ngrids = np.prod(mesh)
    rhoR = np.zeros(ngrids)
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
        ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        rhoR[p0:p1] += make_rho(0, ao_ks, mask, 'LDA')
        ao_ks = None
    rhoG = pbctools.fft(rhoR, mesh)

    Gv = cell.get_Gv(mesh)
    coulG = pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)
    GG = np.einsum('gx,gy->gxy', Gv, Gv)
    absG2 = np.einsum('gxx->g', GG)

    # Corresponding to FC term, that makes the tensor traceless
    idx = np.arange(3)
    GG[:,idx,idx] -= 1./3 * absG2[:,None]

    vG = 1./ngrids * np.einsum('g,g,gxy->gxy', rhoG, coulG, GG)
    SI = cell.get_SI(Gv)
    efg_e = lib.einsum('zg,gxy->zxy', SI[efg_nuc], vG.conj()).real
    return efg_e

def _aft_quad_integrals(mydf, dm, efg_nuc):
    # Use AFTDF to compute the integrals of quadrupole operator
    # (3 \vec{r} \vec{r} - r^2) / r^5
    cell = mydf.cell
    if cell.dimension != 3:
        raise NotImplementedError

    log = lib.logger.new_logger(mydf)
    t0 = t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    mesh = mydf.mesh
    kpts = mydf.kpts
    kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    dm_kpts = dm.reshape((nkpts,nao,nao), order='C')

    ngrids = np.prod(mesh)
    rhoG = np.zeros(ngrids, dtype=np.complex128)
    kpt_allow = np.zeros(3)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts_lst,
                                       max_memory=max_memory, aosym='s1'):
        rhoG[p0:p1] = np.einsum('kgpq,kqp->g', aoaoks.reshape(nkpts,p1-p0,nao,nao),
                                   dm_kpts)
        t1 = log.timer_debug1('contracting Vnuc [%s:%s]'%(p0, p1), *t1)
    t0 = log.timer_debug1('contracting Vnuc', *t0)
    rhoG *= 1./nkpts

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    coulG = pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)
    GG = np.einsum('gx,gy->gxy', Gv, Gv)
    absG2 = np.einsum('gxx->g', GG)

    # Corresponding to FC term, that makes the tensor traceless
    idx = np.arange(3)
    GG[:,idx,idx] -= 1./3 * absG2[:,None]
    vG = 1./cell.vol * np.einsum('g,g,gxy->gxy', rhoG, coulG, GG)

    assert cell.dimension == 3
    a = cell.lattice_vectors()
    ke_cutoff = min(pbctools.mesh_to_cutoff(a, mesh))
    eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)

    nuccell = _compensate_nuccell(cell, eta)
    # PP-loc part1 is handled by fakenuc in _int_nuc_vloc
    efg_e = _int_nuc_vloc(mydf, nuccell, kpts_lst, dm_kpts)
    t0 = log.timer_debug1('vnuc pass1: analytic int', *t0)

    aoaux = df.ft_ao.ft_ao(nuccell, Gv)
    efg_e += np.einsum('gz,gxy->zxy', aoaux[:,efg_nuc], vG.conj()).real
    return efg_e

def _compensate_nuccell(cell, eta):
    '''A cell of the compensated Gaussian charges for nucleus'''
    modchg_cell = cell.copy(deep=False)
    half_sph_norm = .5/np.sqrt(np.pi)
    norm = half_sph_norm/gaussian_int(2, eta)
    chg_env = [eta, norm]
    ptr_eta = cell._env.size
    ptr_norm = ptr_eta + 1
    chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
    modchg_cell._atm = cell._atm
    modchg_cell._bas = np.asarray(chg_bas, dtype=np.int32)
    modchg_cell._env = np.hstack((cell._env, chg_env))
    return modchg_cell

def _int_nuc_vloc(mydf, nuccell, kpts, dm_kpts):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    nkpts = len(kpts)

    # Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = aft._fake_nuc(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            pbcgto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                            fakenuc._atm, fakenuc._bas, fakenuc._env)

    kptij_lst = np.hstack((kpts,kpts)).reshape(-1,2,3)
    v3c = df.incore.aux_e2(cell, fakenuc, 'int3c2e_ipip1', aosym='s1', comp=9,
                           kptij_lst=kptij_lst)
    v3c += df.incore.aux_e2(cell, fakenuc, 'int3c2e_ipvip1', aosym='s1', comp=9,
                            kptij_lst=kptij_lst)

    nao = cell.nao_nr()
    natm = cell.natm
    v3c = v3c.reshape(nkpts,3,3,nao,nao,natm*2)
    efg_loc = 1./nkpts * np.einsum('kxypqz,kqp->zxy', v3c, dm_kpts)
    efg_loc+= 1./nkpts * np.einsum('kyxqpz,kqp->zxy', v3c, dm_kpts)
    v3c = None

    # Fermi contact
    fc = df.incore.aux_e2(cell, fakenuc, 'int3c1e', aosym='s1', kptij_lst=kptij_lst)
    fc = fc.reshape(nkpts,nao,nao,natm*2)
    vfc = 1./nkpts * np.einsum('kpqz,kqp->z', fc, dm_kpts)
    for i in range(3):
        efg_loc[:,i,i] += 4./3*np.pi * vfc

    nuc_part = efg_loc[:natm]
    modchg_part = efg_loc[natm:]
    efg_loc = nuc_part - modchg_part

    if cell.dimension == 3:
        fac = np.pi/cell.vol
        nucbar = np.array([fac/nuccell.bas_exp(i)[0] for i in range(natm)])

        ipip = cell.pbc_intor('int1e_ipipovlp', 9, lib.HERMITIAN, kpts)
        ipvip = cell.pbc_intor('int1e_ipovlpip', 9, lib.HERMITIAN, kpts)
        d2rho_bar = 0
        for k in range(nkpts):
            v = (ipip[k] + ipvip[k]).reshape(3,3,nao,nao)
            d2rho_bar += np.einsum('xypq,qp->xy', v, dm_kpts[k])
            d2rho_bar += np.einsum('yxqp,qp->xy', v, dm_kpts[k])
        d2rho_bar *= 1./nkpts

        efg_loc -= np.einsum('z,xy->zxy', nucbar, d2rho_bar)
    return efg_loc.real
