#!/usr/bin/env python

import h5py
import numpy as np

from xctph.units import ryd2ev, ryd2mev
from xctph.symmetrize_mtxel import sym_mtxel


def print_eph(fname_eph_h5):
  """Prints xctph matrix elements EPW style. """

  with h5py.File(fname_eph_h5, 'r') as f:
  
    # band information
    ns = f['gkq_header/ns'][()]
    nbndskip = f['gkq_header/nbndskip'][()]
    nbnd = f['gkq_header/nbnd'][()]
    nocc = f['gkq_header/nocc'][()]
    nmode = f['gkq_header/nmode'][()]
  
    # k/q point information
    nk = f['gkq_header/nk'][()]
    nq = f['gkq_header/nq'][()]
    kpts = f['gkq_header/kpts'][()]
    qpts = f['gkq_header/qpts'][()]
    k_plus_q_map = f['gkq_mappings/k_plus_q_map'][()]
  
    # state information information
    energies = f['gkq_data/energies'][()]
    frequencies = f['gkq_data/frequencies'][()]
    gkq = f['gkq_data/g_nu'][()]

  
  # prepare to write
  with open('eph.dat', 'w') as f:

    for ik, k in enumerate(kpts):
      for iq, q in enumerate(qpts):

        # find index for the point at k+q
        ikq = k_plus_q_map[ik, iq]

        # write k, q point information
        f.write('\n{:^20s}{k[0]:8.4f}{k[1]:8.4f}{k[2]:8.4f}\n'
                .format('kpt = ', k=k))

        f.write('{:^20s}{q[0]:8.4f}{q[1]:8.4f}{q[2]:8.4f}\n\n'
                .format('qpt = ', q=q))

        f.write('{:>6s}{:>6s}{:>7s}{:>10s}{:>11s}{:>13s}{:>13s} '
                '{:>12s}{:>11s}{:>14s}\n'
                .format('nbnd', 'mbnd', 'imode',
                        'enk (eV)', 'emkq (eV)', 'omega (meV)',
                        'Re[g] (meV)', 'Im[g] (meV)', '|g| (meV)',
                        'Sym|g| (meV)'
                        )
                )

        ek = energies[:, ik] * ryd2ev
        ekq = energies[:, ikq] * ryd2ev
        freq = frequencies[:, iq] * ryd2mev
        g = gkq[:, :, ik, :, iq] * ryd2mev
        g_sym = sym_mtxel(g, ek, ekq, freq)

        for nb in range(nbnd):
          for mb in range(nbnd):
            for imode in range(nmode):
              # Large write statement
              f.write(
                '{:>6d}{:>6d}{:>7d}{:>10.3f}{:>11.3f}'
                '{:>13.2f}{:>13.2f}{:>12.2f}{:>11.2f}'
                '{:>14.2f}\n'
                .format(nb + 1,
                        mb + 1,
                        imode + 1,
                        ek[nb],
                        ekq[mb],
                        freq[imode],
                        np.real(g[mb, nb, imode]),
                        np.imag(g[mb, nb, imode]),
                        np.abs(g[mb, nb, imode]),
                        np.abs(g_sym[mb, nb, imode])
                        )
              )


if __name__ == '__main__':

  import argparse 
  parser = argparse.ArgumentParser()
  parser.add_argument('fname_eph_h5')
  args = parser.parse_args()

  print_eph(**vars(args))
