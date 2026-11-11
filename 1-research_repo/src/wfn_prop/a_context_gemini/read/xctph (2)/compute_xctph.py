#!/usr/bin/env python

import numpy as np
import h5py

from xctph.kpoints import get_all_kq_maps


def compute_xctph(fname_eph_h5, fname_xct_h5, nbnd_xct):
    """ Compute the exciton-phonon matrix elements. """

    with h5py.File(fname_eph_h5, 'r') as f:
        nmodes = f['gkq_header/nmode'][()]
        nk_elph = f['gkq_header/nk'][()]
        kpts_elph = f['gkq_header/kpts'][()]
        nq = f['gkq_header/nq'][()]
        qpts = f['gkq_header/qpts'][()]
        k_plus_q_map = f['gkq_mappings/k_plus_q_map'][()]
        frequencies = f['gkq_data/frequencies'][()]
        gkq = f['gkq_data/g_nu'][()]

    with h5py.File(fname_xct_h5, 'r') as f:
        nbnd = f['/exciton_header/nevecs'][()]
        nv = f['/exciton_header/nv'][()]
        nc = f['/exciton_header/nc'][()]
        nk = f['/exciton_header/nk'][()]
        kpts = f['/exciton_header/kpts'][()]
        nQ = f['/exciton_header/nQ'][()]
        Qpts = f['/exciton_header/center_of_mass_Q'][()]
        energies = f['/exciton_data/eigenvalues'][()]
        avck = f['/exciton_data/eigenvectors'][()]


    # consistency checks:
    assert nk_elph == nk
    assert nbnd >= nbnd_xct

    # generate additional kq maps
    Q_plus_q_map = get_all_kq_maps(Qpts, qpts)
    k_minus_Q_map = get_all_kq_maps(kpts, Qpts, -1.0)

    # xct-ph matrix elements are packaged the same way as el-ph
    gQq = np.zeros((nbnd_xct, nbnd_xct, nQ, nmodes, nq), 'c16')

    cb = slice(nv, nv + nc)
    vb = slice(0, nv)

    for iQ in range(nQ):
      for iq in range(nq):
        iQ_plus_q = Q_plus_q_map[iQ, iq]

        for ik in range(nk):
          ik_plus_q = k_plus_q_map[ik, iq]
          ik_minus_Q = k_minus_Q_map[ik, iQ]

          for mb in range(nbnd_xct):
            for nb in range(nbnd_xct):

                # electron channel
                aQ_e = avck[0, :, :, ik, nb, iQ]
                aQq_e = avck[0, :, :, ik_plus_q, mb, iQ_plus_q]
                gkq_e = gkq[cb, cb, ik, :, iq]

                gQq[mb, nb, iQ, :, iq] += np.einsum('vc,cdn,vd->n', aQq_e.conj(), gkq_e, aQ_e)

                # hole channel
                aQ_h = avck[0, :, :, ik_plus_q, nb, iQ]
                aQq_h = avck[0, :, :, ik_plus_q, mb, iQ_plus_q]
                gkq_h = gkq[vb, vb, ik_minus_Q, :, iq][::-1, ::-1, :]

                gQq[mb, nb, iQ, :, iq] -= np.einsum('vc,wvn,wc->n', aQq_h.conj(), gkq_h, aQ_h)


    f = h5py.File('xctph.h5', 'w')

    xctph_dict = {
        # header information.
        'ns': 1,
        'nbndskip': 0,
        'nbnd': nbnd_xct,
        'nocc': 0,
        'nmode': nmodes,
        'nQ': nQ,
        'nq': nq,
        'Qpts': Qpts,
        'qpts': qpts,

        # Q+q mappings.
        'Q_plus_q_map': Q_plus_q_map,

        # energies, frequencies, and matrix elements.
        'energies': energies[:nbnd_xct, :],
        'frequencies': frequencies,
        'xctph' : gQq,

    }


    for name, data in xctph_dict.items():
        f.create_dataset(name, data=data)
    f.close()

    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fname_eph_h5')
    parser.add_argument('fname_xct_h5')
    parser.add_argument('nbnd_xct', type=int)
    args = parser.parse_args()

    compute_xctph(**vars(args))
