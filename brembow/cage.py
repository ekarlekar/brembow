import numpy as np
from pyquaternion import Quaternion
from biopandas.pdb import PandasPdb
import glob
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R


class Cage:
    '''A cage represented by a list of locations.
    Args:
        cage_folder (``string``):
            Directory containing the PDB files for this cage.
        cage_id (``int``):
            A unique ID to refer to this cage, has to be larger than 0.
    '''

    # list of locations in 3D
    def __init__(self, cage_folder, cage_id):

        assert cage_id > 0, (
            "Cage IDs should be greater than 0 (0 is used internally as 'no "
            "cage')")

        self.cage_id = cage_id
        locations, atomic_numbers, atom_types = self.__get_atom_locations(
            cage_folder)
        self.locations = locations
        self.atomic_numbers = atomic_numbers
        self.atom_types = atom_types
        self.rotation_changed = False
        self.rotated = self.locations
        self.quaternion = None

    def get_locations(self):

        if self.rotation_changed:
            self.rotated = np.array(self.locations)

            # apply the rotation using scipy
            r = R.from_quat(self.quaternion.elements)
            self.rotated = r.apply(self.rotated)

            self.rotation_changed = False

        return self.rotated

    def get_atomic_numbers(self):
        return self.atomic_numbers

    def set_random_rotation(self):
        self.quaternion = Quaternion.random()
        self.rotation_changed = True

    def __get_atom_locations(self, cage_folder):

        # Creates list of .pdb file names
        pdb_files = glob.glob(os.path.join(cage_folder, '*.pdb'))

        list_file_df = []

        atom_type = []
        # ATOM -> 1
        # HETATM -> 2

        for file_name in pdb_files:
            ppdb = PandasPdb().read_pdb(file_name)
            list_file_df.append(ppdb.df['ATOM'])
        big_df = pd.concat(list_file_df)
        curr_len = len(big_df)
        atom_type.extend(1 for x in range(curr_len))

        for file_name in pdb_files:
            ppdb = PandasPdb().read_pdb(file_name)
            list_file_df.append(ppdb.df['HETATM'])
        big_df = pd.concat(list_file_df)
        atom_type.extend(2 for x in range(len(big_df)-curr_len))

        # atoms_df = big_df.loc[big_df['atom_name'] == atom_name]
        atom_location_matrix = big_df[["z_coord",
                                       "y_coord",
                                       "x_coord"]].to_numpy()
        # get all atomic numbers
        atomic_numbers = np.array([
            self.atom_name_to_atomic_number(atom_name)
            for atom_name in big_df['atom_name']
        ])

        # all the locations are in Ångström, convert them to nm
        atom_location_matrix /= 10.0

        # center cage at (0, 0, 0)
        center = np.mean(atom_location_matrix, axis=0)
        atom_location_matrix -= center

        return atom_location_matrix, atomic_numbers, atom_type

    def atom_name_to_atomic_number(self, atom_name):

        atomic_number = {
            '3HD1': 1,
            '2HZ': 1,
            '2HB': 1,
            'HH': 1,
            '2H': 1,
            '2HH1': 1,
            '3H': 1,
            '1HD': 1,
            '3HB': 1, # not sure if Hydrogen or Boron
            '1HD2': 1,
            '2HD2': 1,
            'HD2': 1,
            '1H': 1,
            '2HA': 1,
            '2HD': 1,
            '1HH1': 1,
            'HD1': 1,
            'H': 1,
            'HA': 1,
            'HB': 1, # not sure if Hydrogen or Boron
            '1HH2': 1,
            '1HA': 1,
            '1HZ': 1,
            '3HZ': 1,
            '1HD1': 1,
            '2HD1': 1,
            '2HH2': 1,
            '1HB': 1,
            '3HD2': 1,
            'HZ': 1,
            'HB1': 1, # not sure if Hydrogen or Boron
            'HB2': 1, # not sure if Hydrogen or Boron
            'HH21': 1,
            'HH12': 1,
            'HD13': 1,
            'H2': 1,
            'HA2': 1,
            'HH11': 1,
            'HZ1': 1,
            'HD11': 1,
            'H1': 1,
            'HZ3': 1,
            'HZ2': 1,
            'HD21': 1,
            'HB3': 1, # not sure if Hydrogen or Boron
            'HD3': 1,
            'HA1': 1,
            'HD12': 1,
            'HD22': 1,
            'HD23': 1,
            'HH22': 1,

            'HE2': 2,
            'HE1': 2,
            '3HE': 2,
            '2HE2': 2,
            '2HE': 2,
            '1HE2': 2,
            'HE': 2,
            '1HE': 2,
            'HE21': 2,
            'HE22': 2,
            'HE3': 2,

            'C': 6,
            'CD': 6,
            'CZ': 6,
            'CA': 6, # not sure if Calcium or Carbon, assuming C from context clues
            'CD2': 6,
            'CE1': 6, # not sure if Ce (58, Cerium) or Carbon
            'CG2': 6,
            'CG': 6,
            'CE': 6, # not sure if Ce (58, Cerium) or Carbon
            'CD1': 6,
            'CE2': 6, # not sure if Ce (58, Cerium) or Carbon
            'CG1': 6,
            'CB': 6,

            'N': 7,
            'NH1': 7,
            'ND2': 7,
            'NH2': 7, # not sure if Nitrogen or Hydrogen
            'NE2': 7,
            'NE': 7,
            'NZ': 7,
            'ND1': 7,

            'O': 8,
            'OE1': 8,
            'OE2': 8,
            'OG1': 8,
            'OD2': 8,
            'OG': 8,
            'OXT': 8,
            'OD1': 8,
            'O1': 8,
            'O2': 8,
            'OH': 8, # not sure if Oxygen or Hydrogen

            'SD': 16, # assuming Sulfur
            'SG': 16,

            '1HG1': 80, # not sure if Hydrogen or Mercury, assuming Hg
            '3HG1': 80,
            'HG': 80,
            '2HG1': 80,
            '1HG2': 80,
            '3HG2': 80,
            '1HG': 80,
            '2HG2': 80,
            '2HG': 80,
            'HG1': 80,
            'HG2': 80,
            'HG13': 80,
            'HG11': 80,
            'HG3': 80,
            'HG22': 80,
            'HG12': 80,
            'HG23': 80,
            'HG21': 80,

            'U': 92
            }

        try:
            return atomic_number[atom_name]
        except KeyError as e:
            print(
                f"Don't know the atomic number for {atom_name} (yet), please "
                "fill in!")
            raise
