import numpy as np
from pyquaternion import Quaternion
from biopandas.pdb import PandasPdb
import glob
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R


class Cage:
    # list of locations in 3D
    def __init__(self, cage_folder):
        self.locations = self.get_atom_locations(cage_folder, "HETATM", "U")
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

    def set_random_rotation(self):
        self.quaternion = Quaternion.random()
        self.rotation_changed = True

    def get_atom_locations(self, cage_folder, record_name, atom_name):

        # Creates list of .pdb file names
        pdb_files = glob.glob(os.path.join(cage_folder, '*.pdb'))

        list_file_df = []

        for file_name in pdb_files:
            ppdb = PandasPdb().read_pdb(file_name)
            list_file_df.append(ppdb.df[record_name])
        big_df = pd.concat(list_file_df)

        atoms_df = big_df.loc[big_df['atom_name'] == atom_name]
        atom_location_matrix = atoms_df[["z_coord",
                                         "y_coord",
                                         "x_coord"]].to_numpy()
        return atom_location_matrix
