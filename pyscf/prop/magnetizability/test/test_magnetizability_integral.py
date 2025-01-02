import numpy
from pyscf import lib
from pyscf import gto, dft
from pyscf.scf import jk
from pyscf.dft import numint

def test_integral():
    mol = gto.M(atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587''',
                basis='ccpvdz')

    nao = mol.nao
    dm0 = numpy.random.random((nao,nao))
    dm0 = dm0 + dm0.T

    v1 = mol.intor('int2e_gg1', comp=9).reshape(3,3,nao,nao,nao,nao)
    v2 = mol.intor('int2e_g1g2', comp=9).reshape(3,3,nao,nao,nao,nao)
    v = v1 + v1.transpose(0,1,4,5,2,3)
    assert abs(v1 - v1.transpose(0,1,3,2,4,5)).max() < 1e-8
    assert abs(v1 - v1.transpose(0,1,2,3,5,4)).max() < 1e-8
    assert abs(v2 - v2.transpose(1,0,4,5,2,3)).max() < 1e-8

    assert abs(v - v.transpose(0,1,3,2,4,5)).max() < 1e-8
    assert abs(v - v.transpose(0,1,2,3,5,4)).max() < 1e-8
    assert abs(v - v.transpose(1,0,4,5,2,3)).max() < 1e-8
    assert abs(v - v.transpose(0,1,3,2,5,4)).max() < 1e-8
    jref = numpy.einsum('xyijkl,ji->xykl', v, dm0)
    kref = numpy.einsum('xyijkl,jk->xyil', v, dm0)

    vs = jk.get_jk(mol, [dm0]*4, ['ijkl,ji->s2kl',
                                  'ijkl,lk->s2ij',
                                  'ijkl,jk->s1il',
                                  'ijkl,li->s1kj'],
                   'int2e_gg1', 's4', 9, hermi=1)
    vj = vs[0] + vs[1]
    vk = vs[2] + vs[3]
    vj = vj.reshape(3,3,nao,nao)
    vk = vk.reshape(3,3,nao,nao)
    assert abs(vj - jref).max() < 1e-8
    assert abs(vk - kref).max() < 1e-8
    assert abs(jref - jref.transpose(1,0,3,2)).max() < 1e-8
    assert abs(kref - kref.transpose(1,0,3,2)).max() < 1e-8

    v += 2 * v2
    jref = numpy.einsum('xyijkl,ji->xykl', v, dm0)
    kref = numpy.einsum('xyijkl,jk->xyil', v, dm0)
    assert abs(jref - jref.transpose(1,0,3,2)).max() < 1e-8
    assert abs(kref - kref.transpose(1,0,3,2)).max() < 1e-8

    vs = jk.get_jk(mol, [dm0]*2, ['ijkl,ji->s2kl',
                                  'ijkl,jk->s1il'],
                   'int2e_g1g2', 'aa4', 9, hermi=0)
    kk = vs[1].reshape(3,3,nao,nao)
    assert abs(kk-kk.transpose(1,0,3,2)).max() < 1e-8

    vj += vs[0].reshape(3,3,nao,nao) * 2
    vk += vs[1].reshape(3,3,nao,nao) * 2
    assert abs(vj - jref).max() < 1e-8
    assert abs(vk - kref).max() < 1e-8


#########################
# DFT
    mol = gto.M(atom='''
        H  0. , 0. , 0.
        H  0. , .7 , .8
        H  .6 , 0. , .5
        H  .9 , .4 , 0.
                ''')
    mf = dft.RKS(mol).run(conv_tol=1e-12)
    grids = mf.grids
    ni = mf._numint
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dm = mf.make_rdm1()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm)
    vmat = numpy.zeros((3,nao,nao))
    ao_deriv = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv):
        rho = make_rho(0, ao, mask, 'LDA')
        vxc = ni.eval_xc('LDA,', rho, 0, deriv=1)[1]
        vrho = vxc[0]
        aow = numpy.einsum('pi,p->pi', ao, weight*vrho)
        giao = mol.eval_gto('GTOval_ig', coords, comp=3, non0tab=mask)
        vmat[0] += numint._dot_ao_ao(mol, aow, giao[0], mask, shls_slice, ao_loc)
        vmat[1] += numint._dot_ao_ao(mol, aow, giao[1], mask, shls_slice, ao_loc)
        vmat[2] += numint._dot_ao_ao(mol, aow, giao[2], mask, shls_slice, ao_loc)
        rho = vxc = vrho = aow = None
    vref = vmat - vmat.transpose(0,2,1)
    assert abs(lib.fp(vref) - -0.236996049404) < 1e-7

    ao = mol.eval_gto('GTOval', grids.coords)
    rho = make_rho(0, ao, grids.non0tab, 'LDA')
    vxc = ni.eval_xc('LDA,', rho, 0, deriv=1)[1]
    vrho = vxc[0]

    aow = numpy.einsum('pi,p->pi', ao, grids.weights*vrho)
    giao = mol.eval_gto('GTOval_ig', grids.coords, comp=3)
    vmat = (numint._dot_ao_ao(mol, aow, giao[0], grids.non0tab, shls_slice, ao_loc),
            numint._dot_ao_ao(mol, aow, giao[1], grids.non0tab, shls_slice, ao_loc),
            numint._dot_ao_ao(mol, aow, giao[2], grids.non0tab, shls_slice, ao_loc))
    vmat = numpy.array(vmat)
    vmat = vmat - vmat.transpose(0,2,1)
    assert abs(vref - vmat).max() < 1e-9

    aow = numpy.einsum('pi,p,p,px->pxi', ao, grids.weights, vrho, grids.coords)
    vmat = lib.einsum('pxi,pj->xij', aow, ao)
    atom_coords = mol.atom_coords()
    Rx = atom_coords[:,0]
    Ry = atom_coords[:,1]
    Rz = atom_coords[:,2]
    v1 = numpy.zeros_like(vmat)
    v1[0] += (Ry[:,None]-Ry) * vmat[2]
    v1[0] -= (Rz[:,None]-Rz) * vmat[1]
    v1[1] += (Rz[:,None]-Rz) * vmat[0]
    v1[1] -= (Rx[:,None]-Rx) * vmat[2]
    v1[2] += (Rx[:,None]-Rx) * vmat[1]
    v1[2] -= (Ry[:,None]-Ry) * vmat[0]
    v1 *= -.5
    assert abs(vref - v1).max() < 1e-9
