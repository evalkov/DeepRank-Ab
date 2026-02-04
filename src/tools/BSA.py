"""
BSA (Buried Surface Area) computation module.

Supports two backends controlled by the USE_FREESASA environment variable:
  - USE_FREESASA=1 (default): Use freesasa library (original implementation)
  - USE_FREESASA=0: Use voronota-lt via VoroContacts (faster, but may differ slightly)

Example:
    export USE_FREESASA=1   # Use freesasa (default, scientifically validated)
    export USE_FREESASA=0   # Use voronota-lt (faster)
"""

import os
import numpy as np
from pdb2sql.interface import interface


def _use_freesasa() -> bool:
    """Check if freesasa backend should be used (default: True)."""
    val = os.environ.get("USE_FREESASA", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


# Lazy import for voronota-lt backend
_VoroContacts_BSA = None


def _get_voronota_bsa():
    """Lazily import BSA from VoroContacts to avoid circular imports."""
    global _VoroContacts_BSA
    if _VoroContacts_BSA is None:
        from tools.VoroContacts import BSA as VoroBSA
        _VoroContacts_BSA = VoroBSA
    return _VoroContacts_BSA


# Import freesasa if available
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
            xyz = np.mean(self.sql.get(
                'x,y,z', resSeq=r[1], chainID=r[0]), 0)
            xyzkey = tuple([chain] + xyz.tolist())

            self.bsa_data[r] = [bsa]
            self.bsa_data_xyz[r] = xyzkey


class BSA:
    """
    BSA computation with configurable backend.

    Backend selection via USE_FREESASA environment variable:
      - USE_FREESASA=1 (default): Use freesasa library (original, validated)
      - USE_FREESASA=0: Use voronota-lt via VoroContacts (faster)

    This is a wrapper that delegates to the appropriate implementation.
    """

    def __new__(cls, pdb_data, sqldb=None, chainA='A', chainB='B', **kwargs):
        """
        Factory that returns the appropriate BSA implementation.

        Returns:
            BSA_Freesasa or VoroContacts.BSA instance depending on USE_FREESASA
        """
        if _use_freesasa():
            return BSA_Freesasa(pdb_data, sqldb=sqldb, chainA=chainA, chainB=chainB)
        else:
            VoroBSA = _get_voronota_bsa()
            return VoroBSA(pdb_data, sqldb=sqldb, chainA=chainA, chainB=chainB, **kwargs)
