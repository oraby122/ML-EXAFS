import os
import shutil
import tarfile
import numpy as np

# Element:Atomic number dictionary
element_to_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

# Default FEFF Template
feff_template = """TITLE	{title}
EDGE	{edge}
S02	0.9
CONTROL	1	1	1	1	1	1
PRINT	0	0	0	0	0	0
EXCHANGE	0	1.	0.
SCF	5.5	0	100	0.1	1
COREHOLE	RPA
XANES	5	0.05	0.1
FMS	7	0
POTENTIALS
{potentials}
ATOMS
{atoms}
END"""

def make_potential_atoms_from_xyz(xyz, absorber=0):
    '''
    This function takes in xyz coordinates and aborber index
    and return the POTENTIALS and ATOMS cards in FEFF

    '''
    lines = xyz.split("\n")

    if len(lines) < 3:
        raise ValueError("Invalid xyz")

    n_atoms = int(lines[0])
    lines = lines[2:]

    elements = []
    coodinates = []
    for line in lines:
        if not line: # skip empty lines at the end
            continue
        element, x, y, z = line.split()
        element = element.strip()
        x, y, z = float(x), float(y), float(z)

        elements.append(element)
        coodinates.append([x, y, z])

    elements = np.array(elements)
    coodinates = np.array(coodinates)
    print("elements: ", elements)
    print("coodinates: ", coodinates)
    # Potentials
    unique_elements = np.unique(elements)
    unique_counts = np.array(
        [np.sum(elements == element) for element in unique_elements]
    )
    print("unique_elements: ", unique_elements)
    print("unique_counts: ", unique_counts)

    absorber_element = element_to_Z[elements[absorber]]
    print("elements[absorber]: ", elements[absorber])
    potentials = []
    potential_dict = {}

    potentials.append(
        f"0 {absorber_element} {elements[absorber]} -1 -1 0.001"
    )  # absorber
    print("potentials: ", potentials)
    counter = 1
    for element, count in zip(unique_elements, unique_counts):
        if (element == elements[absorber]) and (count <= 1): # skip absorber if it there is only one atom in the cluster
            continue

        potentials.append(f"{counter} {element_to_Z[element]} {element} -1 -1 {count}")
        potential_dict[element] = counter
        counter += 1
    print("potentials: ", potentials)
    potentials = "\n".join(potentials)

    # Atoms

    atoms = []

    for i, (element, coodinate) in enumerate(zip(elements, coodinates)):
        if i == absorber:
            print(i)
            atoms.append(f"{coodinate[0]} {coodinate[1]} {coodinate[2]} 0")
            continue
        atoms.append(
            f"{coodinate[0]} {coodinate[1]} {coodinate[2]} {potential_dict[element]}"
        )

    atoms = "\n".join(atoms)

    return potentials, atoms


def write_feff_dir(feff_inp, directory):
    '''
    Create a FEFF working directory for a FEFF input file
    '''
    os.makedirs(directory, exist_ok=True)
    with open(directory + "feff.inp", "w") as f:
        f.write(feff_inp)
    
def write_feff_dir_from_xyz(
    xyz, directory, absorber=0, edge="K", title="test", xmu_path=None, feff_inp_path=None, feff_template=feff_template
):
    '''
    Create a FEFF working directory for a XYZ file
    '''
    with open(xyz,'r') as f:
        xyz = f.read()
    potentials, atoms = make_potential_atoms_from_xyz(xyz, absorber=absorber)
    feff_inp = feff_template.format(
        title=title, edge=edge, potentials=potentials, atoms=atoms
    )
    write_feff_dir(feff_inp, directory)

    if xmu_path is not None:
        os.makedirs(os.path.dirname(xmu_path), exist_ok=True)
        shutil.copy(directory + "xmu.dat", xmu_path)
        
    if feff_inp_path is not None:
        os.makedirs(os.path.dirname(feff_inp_path), exist_ok=True)
        shutil.copy(directory + "feff.inp", feff_inp_path)

def create_slurm_scripts(dir_list, script_prefix, file_path):
    """
    Create SLURM batch scripts from a list of directories.

    :param dir_list: List of directories to process
    :param script_prefix: Prefix for the output SLURM script files
    """
    step_size = len(dir_list)//2
    for block_num, i in enumerate(range(0, len(dir_list), step_size), start = 1):
        block = dir_list[i:i + step_size]
        script_name = f"{script_prefix}_block{block_num}.sh"

        with open(file_path+script_name, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --job-name=feff\n")
            file.write("#SBATCH --ntasks=128\n")  
            file.write("#SBATCH --cpus-per-task=1\n") 
            file.write("#SBATCH --time=12:00:00\n")  # 24 hour time limit
            file.write("#SBATCH -o job-%j.out\n")
            file.write("#SBATCH -e job-%j.err\n")
            file.write('#SBATCH --wckey=ne_gen\n')
            file.write("\n")

            for index,dir in enumerate(block):
                if index != 0 and index % 128 == 0:
                    file.write("wait\n")
                dir_modified = dir.replace('/mnt/sdcc/', '/')
                file.write(f"srun --exclusive -N 1 -n 1 --cpus-per-task=1 --mem=1G --time=00:10:00 --chdir={dir_modified} ~/software/feff_code/feff.sh &\n")
            file.write("wait\n")
            
    print(f"Created {block_num} SLURM script(s) with prefix '{script_prefix}' in directory {file_path}.")

def create_tar_gz_of_directory(directory_path, output_filename, root_dir_name):
    """
    Create a tar.gz archive of the specified directory.

    :param directory_path: Path of the directory to be archived.
    :param output_filename: Name of the output tar.gz file.
    :param root_dir_name: Name of the root directory in the archive.
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                arcname = os.path.join(root_dir_name, os.path.relpath(filepath, directory_path))
                tar.add(filepath, arcname=arcname)
