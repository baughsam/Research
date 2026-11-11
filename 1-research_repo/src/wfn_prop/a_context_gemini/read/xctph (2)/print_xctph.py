#!/usr/bin/env python

import h5py
import numpy as np

from xctph.units import ryd2ev, ryd2mev
from xctph.symmetrize_mtxel import sym_mtxel


def print_xctph(fname_xctph_h5):
  """Prints xctph matrix elements EPW style. """

  with h5py.File(fname_xctph_h5, 'r') as f:
  
    # band information
    ns = f['ns'][()]
    nbndskip = f['nbndskip'][()]
    nbnd = f['nbnd'][()]
    nocc = f['nocc'][()]
    nmode = f['nmode'][()]
  
    # k/q point information
    nQ = f['nQ'][()]
    nq = f['nq'][()]
    Qpts = f['Qpts'][()]
    qpts = f['qpts'][()]
    Q_plus_q_map = f['Q_plus_q_map'][()]
  
    # state information information
    energies = f['energies'][()]
    frequencies = f['frequencies'][()]
    gkq = f['xctph'][()]

  
  # prepare to write
  with open('xctph.dat', 'w') as f:

    for iQ, Q in enumerate(Qpts):
      for iq, q in enumerate(qpts):

        # find index for the point at k+q
        iQq = Q_plus_q_map[iQ, iq]

        # write k, q point information
        f.write('\n{:^20s}{Q[0]:8.4f}{Q[1]:8.4f}{Q[2]:8.4f}\n'
                .format('kpt = ', Q=Q))

        f.write('{:^20s}{q[0]:8.4f}{q[1]:8.4f}{q[2]:8.4f}\n\n'
                .format('qpt = ', q=q))

        f.write('{:>6s}{:>6s}{:>7s}{:>10s}{:>11s}{:>13s}{:>13s} '
                '{:>12s}{:>11s}{:>14s}\n'
                .format('nbnd', 'mbnd', 'imode',
                        'eSQ (eV)', 'eS\'Qq (eV)', 'omega (meV)',
                        'Re[G] (meV)', 'Im[G] (meV)', '|G| (meV)',
                        'Sym|G| (meV)'
                        )
                )

        eQ = energies[:, iQ] * ryd2ev
        eQq = energies[:, iQq] * ryd2ev
        freq = frequencies[:, iq] * ryd2mev
        g = gkq[:, :, iQ, :, iq] * ryd2mev
        g_sym = sym_mtxel(g, eQ, eQq, freq)

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
                        eQ[nb],
                        eQq[mb],
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
  parser.add_argument('fname_xctph_h5')
  args = parser.parse_args()
  
  print_xctph(**vars(args))
