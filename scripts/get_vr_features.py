"""This program interprets crystal reduction potential data and outputs a CSV with one-hot encodings for each element"""

import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from ase.io import read as ase_read
import math
import os
from cgcnndefect.util import ELEM_DICT

vr_data = pd.read_csv("Vr.csv")
vr_encoding_data = pd.read_csv("vr_one_hot.csv")

vr = vr_data["Vr"].tolist()
min_vr = min(vr)
max_vr = max(vr)

dataRange = max(vr)-min(vr)

num_categories = 10 # Number of bins for one-hot encoding
cat_size = dataRange/num_categories

def to_one_hot(vr):
    cat = math.floor((vr-min_vr)/cat_size)
    if(cat==10):
        cat=9 # Fix off-by-one error for maximum reduction potential
    zeros = ["0.0" for x in range(num_categories)]
    zeros[cat] = "1.0"
    return zeros

def generate_vr_lookup():
    pandas_data = []
    for Z in range(100): # Loop through all elements
        vr_entries = vr_data[vr_data["Z"]==Z]
        ox_dict = {} # Stores oxidation states and Vr encodings
        for x in range(len(vr_entries)):
            n = vr_entries.iloc[x]["n"]
            m = vr_entries.iloc[x]["m"]
            if(not m.is_integer() or not n.is_integer()):
                continue
            Vr_val = vr_entries.iloc[x]["Vr"]
            if n not in ox_dict: # Otherwise smaller oxidation jump found
                ox_dict[n] = " ".join(to_one_hot(Vr_val))
        for y in ox_dict:
            pandas_data.append([Z,y,ox_dict[y]])
    df = pd.DataFrame(pandas_data, columns=['Element', 'Oxidation State', 'Vr encoding'])
    df.to_csv("vr_one_hot.csv",index=False)

def reverse_ox_encoding(encoding):
    ind = np.argmax(encoding)
    return (ind-2) # oxidation state encoding starts at -2 and ends at 7

def generate_file_features():
    root_directory = "sp_cgcnn"
    # Loop through all cif files in the directory
    id_prop_data = pd.read_csv(os.path.join(root_directory,"id_prop.csv.all"))

    cif_ids = ["0289862-O2"]+id_prop_data["0289862-O2"].to_list()

    for idx,cif_id in enumerate(cif_ids):
        print(cif_id)
        structure = ase_read(os.path.join(root_directory, cif_id + '.cif'))
        crystal = Structure(structure.get_cell(),
                            structure.get_chemical_symbols(),
                            structure.get_positions(),
                            coords_are_cartesian=True)

        # Get atom types for crystal
        all_atom_types = [ELEM_DICT[crystal[i].specie.symbol] for i in range(len(crystal))]

        ox_data = np.loadtxt(os.path.join(root_directory,cif_id+".locals"))
        # Use oxidation states and atomic types to generate Vr encodings
        one_hots = []
        for index,ox_encoding in enumerate(ox_data):
            ox = reverse_ox_encoding(ox_encoding) # Atom oxidation state
            atomic_num = all_atom_types[index]
            vr_atom = vr_encoding_data[(vr_encoding_data["Element"] == atomic_num) & (vr_encoding_data["Oxidation State"]==ox)]

            if len(vr_atom)>0: # Metal cation : has data for Vr
                one_hot = np.array([float(x) for x in vr_atom["Vr encoding"].iloc[0].split(" ")])
            else:
                one_hot = np.zeros(10)
            one_hots.append(one_hot)
        all_encodings = np.vstack(tuple(one_hots))
        locals_encodings = np.hstack((ox_data,all_encodings))
        np.savetxt(os.path.join(root_directory,cif_id+".locals2"), locals_encodings, "%.1f",delimiter=' ')
        break

generate_file_features()