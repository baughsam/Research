import numpy as np
import copy


def sym_mtxel(gkk_orig, ek, ekq, freq=None, eps=1e-3):
    """Symmetrize electron phonon matrix elements"""

    gkk = copy.deepcopy(gkk_orig)
    nbnd, nbnd, nmodes = np.shape(gkk)
    gkk_sym = abs(gkk)

    # Symmetrize over phonons first
    if freq is None:
        pass
        # print "No frequencies provided, will not symmetrize over phonon modes"
    else:
        for ibnd in range(nbnd):
            for jbnd in range(nbnd):
                for nu in range(nmodes):
                    g2 = 0
                    n = 0
                    w_1 = freq[nu]
                    for mu in range(nmodes):
                        w_2 = freq[mu]
                        if abs(w_2-w_1) < eps:
                            n = n + 1
                            g2 = g2 + abs(gkk[ibnd, jbnd, mu]) ** 2
                    g2 = g2 / float(n)
                    gkk_sym[ibnd, jbnd, nu] = np.sqrt(g2)
        gkk[:] = gkk_sym

    # Symmetrize over k+q electrons
    for nu in range(nmodes):
        for jbnd in range(nbnd):
            for ibnd in range(nbnd):
                g2 = 0
                n = 0
                e_1 = ekq[ibnd]
                for pbnd in range(nbnd):
                    e_2 = ekq[pbnd]
                    if abs(e_2-e_1) < eps:
                        n = n + 1
                        g2 = g2 + abs(gkk[pbnd, jbnd, nu]) ** 2
                g2 = g2 / float(n)
                gkk_sym[ibnd, jbnd, nu] = np.sqrt(g2)
    gkk[:] = gkk_sym

    # Symmetrize over k electrons
    for nu in range(nmodes):
        for ibnd in range(nbnd):
            for jbnd in range(nbnd):
                g2 = 0
                n = 0
                e_1 = ek[jbnd]
                for pbnd in range(nbnd):
                    e_2 = ek[pbnd]
                    if abs(e_2-e_1) < eps:
                        n = n + 1
                        g2 = g2 + abs(gkk[ibnd, pbnd, nu]) ** 2
                g2 = g2 / float(n)
                gkk_sym[ibnd, jbnd, nu] = np.sqrt(g2)
    gkk[:] = gkk_sym

    return gkk_sym
