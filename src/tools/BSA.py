# Re-export optimized BSA from VoroContacts
from tools.VoroContacts import BSA

# Legacy freesasa-based implementation below for fallback

import numpy as np
from pdb2sql.interface import interface


class BSA_Freesasa:
    """
    Original freesasa-based BSA implementation.
    Kept for comparison/fallback purposes.
    """

    def __init__(self, pdb_data, sqldb=None, chainA='A', chainB='B'):
        try:
            import freesasa
            self.freesasa = freesasa
        except ImportError:
            raise ImportError(
                "freesasa not found. Install with: pip install freesasa"
            )

        self.pdb_data = pdb_data
        if sqldb is None:
            self.sql = interface(pdb_data)
        else:
            self.sql = sqldb
        self.chains_label = [chainA, chainB]

        freesasa.setVerbosity(freesasa.nowarnings)

    def get_structure(self):
        """Prepare freesasa Structure objects for complex and chains."""
        # Full complex
        if isinstance(self.pdb_data, str):
            self.complex = self.freesasa.Structure(self.pdb_data)
        else:
            self.complex = self.freesasa.Structure()
            atomdata = self.sql.get('name,resName,resSeq,chainID,x,y,z')
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName)
                self.complex.addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z
                )

        self.result_complex = self.freesasa.calc(self.complex)

        # Individual chains
        self.chains = {}
        self.result_chains = {}
        for label in self.chains_label:
            self.chains[label] = self.freesasa.Structure()
            atomdata = self.sql.get(
                'name,resName,resSeq,chainID,x,y,z', chainID=label
            )
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName)
                self.chains[label].addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z
                )
            self.result_chains[label] = self.freesasa.calc(self.chains[label])

    def get_contact_residue_sasa(self, cutoff=8.5):
        """Compute BSA for contact residues using freesasa."""
        self.bsa_data = {}
        self.bsa_data_xyz = {}

        res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(res.keys())
        res = res[keys[0]] + res[keys[1]]

        for r in res:
            chain, resSeq = r[0], r[1]
            resName = r[2] if len(r) >= 3 else self.sql.get(
                'resName', resSeq=resSeq, chainID=chain
            )[0]

            # SAS in complex
            select_str = ('res, (resi %d) and (chain %s)' % (resSeq, chain),)
            asa_complex = self.freesasa.selectArea(
                select_str, self.complex, self.result_complex
            )['res']

            # SAS when unbound
            select_str = ('res, resi %d' % resSeq,)
            asa_unbound = self.freesasa.selectArea(
                select_str, self.chains[chain], self.result_chains[chain]
            )['res']

            bsa = asa_unbound - asa_complex

            chain_idx = {'A': 0, 'B': 1}.get(chain, 0)
            xyz = np.mean(
                self.sql.get('x,y,z', resSeq=resSeq, chainID=chain), axis=0
            )
            xyzkey = tuple([chain_idx] + xyz.tolist())

            key = (chain, resSeq, resName)
            self.bsa_data[key] = [bsa]
            self.bsa_data_xyz[key] = xyzkey
