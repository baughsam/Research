#!/usr/bin/env python

import numpy as np
import h5py
import glob

from xctph.units import ryd2ev


def parse_excitons(path_to_eigenvectors):
    """Routine to read in bse excitons computed on some q-grid.

    Parameters
    ----------
    eigenvector_fnames list(str)
        Paths to binary eigenvector files produced by BGW absorption
        calculation.
    Returns
    -------
    ex_data : dict
        Dictionary containing quantities related to exction calculation,
        including exciton energies and coefficients.
    """

    # search for eigenvectors
    fnames = np.sort(glob.glob(path_to_eigenvectors))
    nQ = len(fnames)


    # allocate space for Qpts
    exciton_Q_shift = np.zeros((nQ, 3), 'f8')

    for iQ, fname in enumerate(fnames):

      # Reads in exciton header associated with current q-point.
      with h5py.File(fname, 'r') as f:

         # if iQ=0, store header information.
         if iQ == 0:

             exciton_Q_shift[iQ, :] = f['exciton_header/kpoints/exciton_Q_shifts'][()]
             nevecs = f['exciton_header/params/nevecs'][()]
             ns = f['exciton_header/params/ns'][()]
             nv = f['exciton_header/params/nv'][()]
             nc = f['exciton_header/params/nc'][()]
             nk = f['exciton_header/kpoints/nk'][()]
             kpts = f['exciton_header/kpoints/kpts'][()]

             # some other info needed
             blat = f['mf_header/crystal/blat'][()]
             bvec = f['mf_header/crystal/bvec'][()]

             eigenvalues = np.zeros((nevecs, nQ), 'f8')
             eigenvectors = np.zeros((2, ns, nv, nc, nk, nevecs, nQ), 'f8')

         # otherwise, check values 
         else:

             exciton_Q_shift[iQ, :] = f['exciton_header/kpoints/exciton_Q_shifts'][()]
             assert nevecs == f['exciton_header/params/nevecs'][()]
             assert ns == f['exciton_header/params/ns'][()]
             assert nv == f['exciton_header/params/nv'][()]
             assert nc == f['exciton_header/params/nc'][()]
             assert nk == f['exciton_header/kpoints/nk'][()]


         # reads in energies and exciton coefficients
         eigenvalues[:, iQ] = f['exciton_data/eigenvalues'][:nevecs]
         eigenvectors[..., iQ] = f['exciton_data/eigenvectors'][0,:nevecs,...].T

    eigenvalues = eigenvalues / ryd2ev
    eigenvectors_cplx = eigenvectors[0,...] + 1j * eigenvectors[1,...]

    
    # write exciton to hdf5
    fname_out = 'xct.h5'
    f = h5py.File(fname_out, 'w')

    f.create_group('exciton_header')
    f.create_group('exciton_data')

    ex_data = {

        '/exciton_header/nevecs' : nevecs,
        '/exciton_header/ns' : ns,
        '/exciton_header/nv' : nv,
        '/exciton_header/nc' : nc,
        '/exciton_header/nk' : nk,
        '/exciton_header/nQ' : nQ,
        '/exciton_header/kpts' : kpts,
        '/exciton_header/center_of_mass_Q' : -1.0 * exciton_Q_shift,

        '/exciton_data/eigenvalues' : eigenvalues,
        '/exciton_data/eigenvectors' : eigenvectors_cplx,

        '/mf_header/crystal/blat' : blat,
        '/mf_header/crystal/bvec' : bvec,
    }

    for name, data in ex_data.items():
        f.create_dataset(name, data=data)

    f.close()

    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_eigenvectors')
    args = parser.parse_args()

    parse_excitons(**vars(args))
