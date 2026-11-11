#!/usr/bin/env python

import numpy as np
import scipy.io
import h5py
import xmltodict
import glob

from xctph.kpoints import get_all_kq_maps
from xctph.units import har2ryd, amu2me


def main(dirname_epw, prefix, nq, nc, nv, fname_out=None):
    """Routine to parse electron-phonon related files.

    Parameters
    ----------
    fname_nscf_xml : str
        Path to prefix.xml file writing during a nscf calculation in QE.
    nq : int
        Number of q-points on the 
    nv : int
        Number of valence states.
    nc : int
        NUmber of conduction states.

    """

    def to_array(string):
        return np.genfromtxt(string.split())

    with open(dirname_epw + '/' + prefix + '.xml') as foo:
        data = xmltodict.parse(foo.read())

        # Determine number of pools calculation was run on.
        npool = int(data[u'qes:espresso']['parallel_info']['npool'])
        prefix = data[u'qes:espresso']['input']['control_variables']['prefix']

        # Begin parsing structural information from input.
        f = data[u'qes:espresso']['input']

        nat = int(f['atomic_structure']['@nat'])
        alat = float(f['atomic_structure']['@alat'])
        ntyp = int(f['atomic_species']['@ntyp'])

        lat = np.zeros((3, 3), 'f8')
        pos = np.zeros((nat, 3), 'f8')
        species = dict()
        typlist = list()
        masses = np.zeros(ntyp, 'f8')

        f_cell = f['atomic_structure']['cell']
        for i in range(3):
            lat[i, :] = to_array(f_cell['a{}'.format(i + 1)])

        # Also get recip lat in atomic units (a.u.)! To get cartesian coords:
        # k^{cart} = np.dot(recip_lat, k^{crys}).
        # We use this later to convert derivatives, then:
        # dE^{cart} = np.dot(inv(recip_lat.T), dE^{crys})
        recip_lat = 2 * np.pi * np.linalg.inv(lat)

        if ntyp == 1:
            species[str(f['atomic_species']['species']['@name'])] = 0
            masses[0] = float(f['atomic_species']['species']['mass'])
        else:
            f_species = f['atomic_species']['species']
            for i in range(ntyp):
                species[str(f_species[i]['@name'])] = i
                masses[i] = float(f_species[i]['mass'])

        f_atom = f['atomic_structure']['atomic_positions']['atom']
        if nat == 1:
            typlist.append(f_atom['@name'])
            pos[0, :] = to_array(f_atom['#text'])
        else:
            for i in range(nat):
                typlist.append(f_atom[i]['@name'])
                pos[i, :] = to_array(f_atom[i]['#text'])

        typlist_num = [species[typ] for typ in typlist]

        # Begin parsing output.
        f = data[u'qes:espresso']['output']['band_structure']

        if f['lsda'] == 2:
            raise NotImplemented('Spinor capability not yet implemented.')

        nbnd = int(f['nbnd'])
        nocc = int(float(f['nelec']) / 2.0)
        nk = int(f['nks'])

        # Read in kpoints and energies.
        f = f['ks_energies']

        kpts_cart = np.zeros((nk, 3), 'f8')
        energies = np.zeros((nbnd, nk), 'f8')

        for ik in range(nk):
            kpts_cart[ik, :] = to_array(f[ik]['k_point']['#text'])
            energies[:, ik] = to_array(f[ik]['eigenvalues']['#text']) * har2ryd

    def cart2crys(kk):
        return np.dot(kk, (lat / alat).T)

    kpts = cart2crys(kpts_cart)

    nmodes = 3 * nat
    nk_per_pool = int(nk / npool)

    # Now confirm that the choice of nbnd begin/end makes sense
    nbnd_begin = nocc - nv
    nbnd_end = nocc + nc
    nbnd_red = nc + nv

    if nbnd_begin > nbnd_end:
        raise Exception('Keyword "nbnd_begin" cannot be greater than "nbnd_end".')
    elif nbnd_end > nbnd:
        raise Exception('Keyword "nbnd_begin" cannot be greater than "nbnd in QE".')
    if nbnd_begin < 0:
        raise Exception('Keyword "nbnd_begin" cannot be less than 0.')

    # Arrays to be used later
    gqk_ia = np.zeros((nbnd_red, nbnd_red, nk, nmodes, nq), 'c16')
    dynmat = np.zeros((nmodes, nmodes, nq), 'c16')

    # Loop through and read epb files
    ikk = 0
    for ipool in range(npool):

        print('Reading in {}.epb{}'.format(prefix, ipool + 1))

        f_in = '{}/{}.epb{}'.format(dirname_epw, prefix, ipool + 1)
        f = scipy.io.FortranFile(f_in, mode='r')

        # Define shape of all quantities to be read in.
        nqc_dim = 'i4'
        xqc_dim = '{}f8'.format(3 * nq)
        et_dim = '{}f8'.format(nbnd * nk_per_pool)

        dynq_dim = '{}c16'.format(nmodes * nmodes * nq)
        epmatq_dim = '{}c16'.format(nbnd * nbnd * nmodes * nk_per_pool * nq)
        zstar_dim = '{}f8'.format(3 * nmodes)
        epsi_dim = '{}f8'.format(3 * 3)


        # Read in all epb file
        record = f.read_record(nqc_dim, xqc_dim, et_dim, dynq_dim, epmatq_dim, zstar_dim, epsi_dim)

        if ipool == 0:
            nq_readin = record[0][0]
            qpts_cart = record[1].reshape(3, nq, order='F').T
            ekq = record[2].reshape(nbnd, nk_per_pool, order='F')
            dynmat = record[3].reshape(nmodes, nmodes, nq, order='F')
            qpts = cart2crys(qpts_cart)
            assert nq_readin == nq

        gqk_readin = record[4].reshape(nbnd, nbnd, nk_per_pool, nmodes, nq, order='F')

        # Trim down gqk_atom
        #print('WARNING: excluding bands not included in W90, this is how EPW with qe6.3 works.')
        gqk_ia[:, :, ikk:ikk + nk_per_pool, :, :] = gqk_readin[nbnd_begin:nbnd_end, nbnd_begin:nbnd_end, ...]

        ikk = (ipool + 1) * nk_per_pool


    # Rotate epb matrix elements from atomic to mode basis
    gkq_nu = np.zeros((nbnd_red, nbnd_red, nk, nmodes, nq), 'c16')
    frequencies = np.zeros((nmodes, nq), 'f8')

    print('Begin rotating from gkq from atomic to mode basis')

    for iq in range(nq):
        print('start iq={}'.format(iq))

        dyn = dynmat[..., iq]
        dynm = np.zeros((nmodes, nmodes), 'c16')

        # Compute mass reduced dyn (which we call dynm)
        for i in range(nat):
            for j in range(nat):
                m_i = masses[typlist_num[i]]
                m_j = masses[typlist_num[j]]

                fac = 1.0 / (np.sqrt(m_i * m_j) * amu2me / 2.)

                dynm[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = dyn[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] * fac

        # Symmetrize dynm (which we call dynm_sym)
        dynm_sym = np.asarray((dynm + np.conj(dynm).T) / 2, 'c16')

        # Diagonalize dynm_sym
        w2, v = np.linalg.eigh(dynm_sym)

        # Store eigenfrequencies
        for imode in range(nmodes):

            if iq == 0 and imode < 3:
                frequencies[imode, iq] = 0.

            else:
                if w2[imode] < 0.0:
                    print('Warning: The frequency associated with mode ' \
                          'iq = {}, imode = {} is negative w ={}.' \
                          .format(iq, imode, w2[imode]) )

                frequencies[imode, iq] = np.sqrt(np.abs(w2[imode]))

        # Compute mass reduced normal modes
        vm = np.empty((3 * nat, nmodes), 'c16')

        for i in range(nat):
            m_i = masses[typlist_num[i]]
            vm[3 * i:3 * (i + 1), :] = \
                v[3 * i:3 * (i + 1), :] / np.sqrt(m_i * amu2me / 2.)

        # Loop through gkq_ia and transform into the mode basis
        gk_ai = gqk_ia[..., iq]
        gk_nu = np.zeros((nbnd_red, nbnd_red, nk, nmodes), 'c16')

        for ib in range(nbnd_red):
            for jb in range(nbnd_red):
                for ik in range(nk):
                    for imode in range(nmodes):
                        gk_nu[ib, jb, ik, imode] = np.dot(gk_ai[ib, jb, ik, :], vm[:, imode])

        for imode in range(nmodes):

            if iq == 0 and imode < 3:
                gkq_nu[..., imode, iq] = 0.

            else:
                gkq_nu[..., imode, iq] = gk_nu[..., imode] / np.sqrt(2 * frequencies[imode, iq])

    # We also compute the kq_maps in this
    k_plus_q_map = get_all_kq_maps(kpts, qpts)

    # Routines to write elph to hdf5 format
    if fname_out is None:
        fname_out = 'eph.h5'

    f = h5py.File(fname_out, 'w')

    f.create_group('gkq_header')
    f.create_group('gkq_data')

    sorting_dict = {

        # Header information.
        'gkq_header/ns': 1,
        'gkq_header/nbndskip': nbnd_begin,
        'gkq_header/nbnd': nbnd_red,
        'gkq_header/nocc': nocc,
        'gkq_header/nmode': nmodes,
        'gkq_header/nk': nk,
        'gkq_header/nq': nq,
        'gkq_header/kpts': kpts,
        'gkq_header/qpts': qpts,
        'gkq_header/recip_lat': recip_lat,


        # Information on k+q mappings.
        'gkq_mappings/k_plus_q_map': k_plus_q_map,

        # Data sets.
        'gkq_data/energies': energies[nbnd_begin:nbnd_end],
        'gkq_data/frequencies': frequencies,
        'gkq_data/g_nu': gkq_nu,

    }

    for name, data in sorting_dict.items():
        f.create_dataset(name, data=data)

    f.close()

    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname_epw')
    parser.add_argument('prefix')
    parser.add_argument('nq', type=int)
    parser.add_argument('nc', type=int)
    parser.add_argument('nv', type=int)
    args = parser.parse_args()

    main(**vars(args))
