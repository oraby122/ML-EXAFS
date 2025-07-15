#%%
# Importing Libraries and Packages
import os
import time
import random
import shutil
import subprocess
import numpy as np

from glob import glob
from scipy import stats
from ase.io import read
from feff_tools import *
from scipy.stats import norm
from multiprocessing import Pool
#%%
# Activate if execution
# time measurement is needed
# start = time.time()

# Defining Utility functions
def write_xyz_file_multiatom(filename, coordinate_array, atom_ids):
    '''
        Saves numpy array of XYZ co-ordinates to a file of xyz format

        :param filename: Desired name of the file including extension (e.g. structure.xyz)
        :param co-ordinate_array: Numpy array of XYZ co-ordinates
        :param atom_ids: Atom names/symbols corresponding to each element of the XYZ array
    '''
    with open(filename, 'w') as file:
        file.write(str(len(coordinate_array)) + "\n")
        file.write("ONNE test\n")
        for atom_i,coordinates in enumerate(coordinate_array):
            file.write(f"{atom_ids[atom_i]} {coordinates[0]} {coordinates[1]} {coordinates[2]}\n")

def normal_distro(mean, std):
    '''
        Returns a normal distribution (Continous Random Variable)
        with mu = mean and standard deviation = std
    '''
    mu, sigma = mean, std
    return stats.norm(loc=mu, scale=sigma)
#%%
#########################
# Pre Processing
#########################

# Packmol template: ID is the unique structure identifier
# block is the text for the atomic positions
# Adjust tolerance for the desired minimum interatomic distance
packmol_template = """#
# 
#

tolerance 2.4
filetype xyz 
output structure_{ID}.xyz

{block}
"""
# Block for each atom inside the packmol file
block_template = '''
structure {element}.xyz
   number 1
   inside sphere  0.0 0.0 0.0 {rho}
end structure
'''

# FEFF input file template (tailor to context and needs)
feff_template = """TITLE	{title}
EDGE	{edge}
S02	1.0
CONTROL	1 1	1 1	1 1

SCF 4 1 100 0.2 10 
COREHOLE FSR
CORRECTIONS 0 0 
REAL

EXAFS 14
RPATH 6.0

POTENTIALS
{potentials}
ATOMS
{atoms}
END"""

#########################
# Sampling for ONNE
#########################
# Define a list of means for each coordination shell
l1_means = [1.9,2.0,2.1,2.2,2.3] # Shell F
l2_means = [3.7,3.8,3.9,4.0,4.1] # Shell Na
l3_means = [3.8,3.9,4.0,4.1,4.2] # Shell Zr

random.shuffle(l1_means)
random.shuffle(l2_means)
random.shuffle(l3_means)

l_combo = list(zip(l1_means,l2_means,l3_means))

# Define nearest neighbor coordination number for each shell
# Here its identified for only one shell
F_CNs = [4,5,6,7,8,9] # First shell co-ordination

macrostates = [] # Empty list for packmol input templates by macrostate
n_frames = 100 # Number of configurations per distribution per CN
#%%
# Start Distance Sampling over each CN and different distributions
for CN in F_CNs:
    for index,combo in enumerate(l_combo):
        packmol_inputs = []
        # Generate CRVs 
        S1 = normal_distro(combo[0], 0.1) # 1st shell F distribution
        S2 = normal_distro(combo[1], 0.4) # 2nd shell Na distribution
        S3 = normal_distro(combo[2],0.3) # 2nd shell Zr distribution
        S4 = normal_distro(4.5,1) # 3rd shell F distribution
        
        # Number of F atoms in first shell
        n_F = CN
        # Total atoms in second shell
        l2_atoms = 10
        # Next Neartest Neighbor fractions (Could be obtained from MD)
        Na_frac = .72
        n_Na = int(.72 * l2_atoms)
        n_Zr = int((1-Na_frac)*l2_atoms)
        # Number of F atoms in third shell
        n_F_2 = 5

        # Begin sampling from CRVs for each configuration
        for f in range(n_frames):
            # Sampling for each atom species in each shell
            rho_f = S1.rvs(n_F) 
            rho_Na = S2.rvs(n_Na)
            rho_Zr = S3.rvs(n_Zr)
            rho_F2 = S4.rvs(n_F_2)
            
            # Create packmol blocks for each atom in configuration with distance from
            # absorber rho
            shell1 = [block_template.format(element='F',rho=rho) for rho in rho_f]
            shell2 = [block_template.format(element='Na',rho=rho) for rho in rho_Na]
            shell3 = [block_template.format(element='Zr',rho=rho) for rho in rho_Zr]
            shell4 = [block_template.format(element='F',rho=rho) for rho in rho_F2]

            # Concatenating all atom blocks
            all_blocks = shell1 + shell2 + shell3 + shell4

            # Combine blocks to form a large block
            final_block = ''
            for block in all_blocks:
                final_block += block
                
            # Insert block in packmol template and append template list for the current distributions
            file = packmol_template.format(ID=str(CN)+str(f),block=final_block)
            packmol_inputs.append(file)
        # Append general template list
        macrostates.append(packmol_inputs)
print(f"Files prepared {len(packmol_inputs)}")
#%%
#########################
# Running Packmol
#########################

'''
To make faster computations, packmol inputs were parallelized on 
SLURM
'''

original_dir = os.getcwd() # Get the current directory
working_dir = 'generated_structures' # Define a working directory
os.makedirs(working_dir,exist_ok=True) # Create directory

# Command to run packmol from CLI
command = "~/software/packmol-20.15.1/packmol < packmol.inp"
#%%
# Define a working function for structure generation that 
# python will use to generate structures using multiprocessing

def generate_structure(task_data):
    '''
        Worker function for multiprocessing tool to use for ONNE
        structure generation

        :param task_data: A Tuple that contains the task ID and packmol input file (task_id,packmol_input)
    '''
    # Unpack Task ID and input file
    task_id, packmol_input,macrostate_dir = task_data 
    # Create working directory for each task
    task_working_dir = os.path.join(working_dir,macrostate_dir, f"task_{task_id}")
    os.makedirs(task_working_dir, exist_ok=True)

    # Copy atom files in the task directory
    for file in glob(os.path.join(original_dir,"*.xyz")):
        try:
            shutil.copy(file,task_working_dir)
        except Exception as e:
            print(f"Error moving {file}: {e}")

    # Go to task directory
    os.chdir(task_working_dir)

    # Write packmol file in task directory
    with open('packmol.inp','w') as f:
        f.write(packmol_input)
        f.close()
    try:
        # Use subprocess to execute the run packmol command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Print the command's output
        print(f"Task {task_id} Output:")
        print(result.stdout)

        # Check for errors
        if result.stderr:
            print(f"Task {task_id} encountered an error: {e}")
            print(result.stderr)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Go back to original directory
        os.chdir(original_dir)

# Main function needed by multiprocessing library
if __name__ == "__main__":
    # Loop over packmol input files for each macrostate
    for n,macrostate in enumerate(macrostates):
        # Make a directory for each macrostate
        macrostate_dir = os.path.join(working_dir,f"macrostate_{n}")
        os.makedirs(macrostate_dir,exist_ok=True)

        # Prepare a list of tuples with (ID,Packmol_input) for worker function
        tasks = [(i,packmol_input,f"macrostate_{n}") for i,packmol_input in enumerate(macrostate)]

        # Use multiprocessing to run through all packmol inputs
        with Pool(processes=min(len(tasks),os.cpu_count())) as pool:
            pool.map(generate_structure,tasks)

        # Go back to original directory (For redundancy)
        os.chdir(original_dir)
#%%
#########################
# Post Processing
#########################
# Adding abosrber atom to the structure files using ASE
# Load all generated structures
files_pattern = os.path.join(working_dir,'macrostate*','task_*','structure_*.xyz')
files = glob(files_pattern)
#%%
# Adding the absorber atom at x,y,z = (0,0,0) using ASE
for file in files:
    config = read(file,format='xyz')
    xyz = config.get_positions()
    atom_ids = ['Zr'] + config.get_chemical_symbols()
    new_xyz = np.concatenate((np.array([[0,0,0]]),xyz))
    write_xyz_file_multiatom(file,
                            new_xyz,
                            atom_ids)
#%%
# Creating FEFF input files for each generated structure
feff_dir = 'feff_input_files'
for index,file in enumerate(files):
    directory = os.path.join(feff_dir,file.split('/')[1],
                             "spectra_"+file.split('/')[1].split('_')[1]+'/')
    os.makedirs(directory,exist_ok=True)
    write_feff_dir_from_xyz(file,
                            directory=directory,
                            absorber=0, 
                            title = 'NaF_ZrF4_ONNE',
                            feff_template = feff_template)
#%%
# Creating slurm scripts to run feff
dir_list = glob("feff_input_files/*/*/")
file_path = os.getcwd() + '/'
create_slurm_scripts(dir_list,'run',file_path)

# Activate if execution
# time measurement is needed
# print(f"Time elapsed: {time.time()-start}")