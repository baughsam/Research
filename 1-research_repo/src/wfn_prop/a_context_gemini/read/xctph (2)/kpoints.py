#!/usr/bin/env python

import numpy as np
import copy
from scipy import spatial


def unit_range(klist_in):

    tol = 1e-6
    klist = copy.copy(klist_in)

    for ik, k in enumerate(klist):
        for i, kx in enumerate(k):

            while kx < -tol:
                kx = kx + 1.0
            while kx > 1.0 -tol:
                kx = kx - 1.0

            k[i] = kx

        klist[ik,:] = k

    return klist


def find_kpt(ktargets, klist):

    klist = unit_range(klist)
    ktargets = unit_range(ktargets)

    tree = spatial.KDTree(klist)
    ik_addr = list()
    for k in ktargets:
        d, i = tree.query(k)
        if d > 1e-6:
            print('kpt not found:', k)
            i = None

        ik_addr.append(i)

    return ik_addr


def get_all_kq_maps(kpts=None, qpts=None, plus_or_minus=1.0):

    nk = np.shape(kpts)[0]
    nq = np.shape(qpts)[0]
    kq_maps = np.zeros((nk, nq), dtype=int)

    for iq, q in enumerate(qpts):
        kqpts = np.array(kpts) + plus_or_minus * np.array(q)
        kq_maps[:, iq] = find_kpt(kqpts, kpts)
  
    return kq_maps
