#!/usr/bin/env python

"""
Computing methylformate infrared spectra (intensity unit in km/mol)
"""

from pyscf import gto, dft
from pyscf.prop import infrared
from pyscf.hessian import thermo


if __name__ == '__main__':
    mol = gto.Mole(atom="""
        O  2.76955048 -0.58923815 -0.81058585
        C  2.23194066  0.11774678  0.00179064
        O  2.85490749  0.89296103  0.89265344
        C  4.28541489  0.83272777  0.8226615
        H  1.15462343  0.22497014  0.12524013
        H  4.64284177  1.50263483  1.5927712
        H  4.62542789  1.15385421 -0.15597918
        H  4.62602338 -0.18082661  1.00476811
    """, basis="6-31G", verbose=0).build()

    # infrared calculation also invokes hessian, though calling hessian is hidden and wrapped
    # so this evaluation is somehow time costly
    mf = dft.RKS(mol, xc="PBE0").run()
    mf_ir = infrared.rks.Infrared(mf).run()
    mf_ir.summary()
    # To further perform thermo calculation, use the hessian object inside `mf_ir`:
    thermo.dump_thermo(mol, thermo.thermo(mf, mf_ir.vib_dict["freq_au"], 298.15, 101325))

    # Following code requires matplotlib library
    # NOTE: The vibration frequency in IR spectra plot is scaled by 0.956
    #       For more information, please consult CCCBDB recommendation
    #       https://cccbdb.nist.gov/vibscalejust.asp
    fig, ax, ax2 = mf_ir.plot_ir(w=100, scale=0.956)
    ax.set_title(r"Infrared Spectra of Methylformate Molecule ($\mathrm{CH_3COOH}$)" "\n"
                 r"PBE0/6-31G with scaling factor 0.956 and Lorentz broadening with $\mathrm{FWHW = 100 cm^{-1}}")
    fig.tight_layout()
    fig.show()
