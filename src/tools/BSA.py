"""
BSA (Buried Surface Area) computation module.

Uses the freesasa library for SASA calculations.

Example:
    >>> bsa = BSA('1AK4.pdb')
    >>> bsa.get_structure()
    >>> bsa.get_contact_residue_sasa()
    >>> bsa.sql.close()
"""

import numpy as np
from pdb2sql.interface import interface

_freesasa = None
try:
    import freesasa as _freesasa
except ImportError:
    pass


class BSA_Freesasa(object):
    """
    BSA computation using freesasa library.

    This is the original implementation, considered scientifically validated.
    """

    def __init__(self, pdb_data, sqldb=None, chainA='A', chainB='B'):
        '''Compute the buried surface area feature

        Freesasa is required for this feature.

        https://freesasa.github.io

        >>> wget http://freesasa.github.io/freesasa-2.0.2.tar.gz
        >>> tar -xvf freesasa-2.0.3.tar.gz
        >>> cd freesasa
        >>> ./configure CFLAGS=-fPIC (--prefix /home/<user>/)
        >>> make
        >>> make install

        Since release 2.0.3 the python bindings are separate module
        >>> pip install freesasa

        Args :
            pdb_data (list(byte) or str): pdb data or filename of the pdb
            sqldb (pdb2sql.interface instance or None, optional) if the sqldb is None the sqldb will be created
            chainA (str, optional): name of the first chain
            chainB (str, optional): name of the second chain

        Example :

        >>> bsa = BSA('1AK4.pdb')
        >>> bsa.get_structure()
        >>> bsa.get_contact_residue_sasa()
        >>> bsa.sql.close()

        '''
        if _freesasa is None:
            raise ImportError(
                "freesasa not found. Install with: pip install freesasa"
            )

        self.freesasa = _freesasa
        self.pdb_data = pdb_data
        if sqldb is None:
            self.sql = interface(pdb_data)
        else:
            self.sql = sqldb
        self.chains_label = [chainA, chainB]

        self.freesasa.setVerbosity(self.freesasa.nowarnings)

    def get_structure(self):
        """Prepare freesasa Structure objects for complex and chains."""
        # 1. FULL COMPLEX
        if isinstance(self.pdb_data, str):
            self.complex = self.freesasa.Structure(self.pdb_data)
        else:
            self.complex = self.freesasa.Structure()
            atomdata = self.sql.get(
                'name,resName,resSeq,chainID,x,y,z')
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName[0])
                self.complex.addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z)
        self.result_complex = self.freesasa.calc(self.complex)

        # 2. CHAINS
        self.chains = {}
        self.result_chains = {}
        for label in self.chains_label:
            self.chains[label] = self.freesasa.Structure()
            atomdata = self.sql.get(
                'name,resName,resSeq,chainID,x,y,z', chainID=label)
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName[0])
                self.chains[label].addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z)
            self.result_chains[label] = self.freesasa.calc(
                self.chains[label])

    def get_contact_residue_sasa(self, cutoff=8.5):
        """Compute BSA for contact residues."""
        self.bsa_data = {}
        self.bsa_data_xyz = {}

        res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(res.keys())
        res = res[keys[0]] + res[keys[1]]
        unique_res = set(res)

        # Pre-fetch residue centroids once (used for bsa_data_xyz).
        residue_xyz = {}
        for residue in unique_res:
            # Contact-residue tuples can include extra fields (e.g., residue name).
            # We only need chain ID and residue number for centroid lookup.
            chain_id, resseq = residue[0], residue[1]
            coords = self.sql.get('x,y,z', resSeq=resseq, chainID=chain_id)
            residue_xyz[(chain_id, resseq)] = np.mean(coords, 0) if coords else np.zeros(3)

        for r in res:
            # SAS in complex
            select_str = ('res, (resi %d) and (chain %s)' % (r[1], r[0]),)
            asa_complex = self.freesasa.selectArea(
                select_str, self.complex, self.result_complex)['res']

            # SAS when unbound
            select_str = ('res, resi %d' % r[1],)
            asa_unbound = self.freesasa.selectArea(
                select_str, self.chains[r[0]], self.result_chains[r[0]])['res']

            # BSA = area unbound - area complex
            bsa = asa_unbound - asa_complex

            chain = {'A': 0, 'B': 1}[r[0]]
            xyz = residue_xyz[(r[0], r[1])]
            xyzkey = tuple([chain] + xyz.tolist())

            self.bsa_data[r] = [bsa]
            self.bsa_data_xyz[r] = xyzkey


class BSA:
    """
    BSA computation wrapper.

    Delegates to BSA_Freesasa for buried surface area calculation.
    """

    def __new__(cls, pdb_data, sqldb=None, chainA='A', chainB='B', **kwargs):
        """
        Factory that returns a BSA_Freesasa instance.
        """
        return BSA_Freesasa(pdb_data, sqldb=sqldb, chainA=chainA, chainB=chainB)
