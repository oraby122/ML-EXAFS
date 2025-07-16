#%%
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from data_prep_tools_v2 import *
from sklearn.model_selection import train_test_split
#%%
# ONNE Data Prep
# Loading directories of produced macrostates
paths = glob("../../feff_input_files_new/macrostate_*")
all_macrostates = []
# Looping over all macrostates
for path in paths:
    exafs,rdf_F,rdf_Na, rdf_Zr = data_preparation(path)
    all_macrostates.append([exafs,rdf_F,rdf_Na, rdf_Zr])
# Directory to save the produced dataset
data_dir = "dataset12"
#%%
# Eliminating "bad" EXAFS calculations
all_macrostates = np.array([example for example in all_macrostates if np.max(example[0]) < 2],dtype=object)
# Saving the prepared macrostates in a .pkl file
file_name = "onne_macrostates"
pickle.dump(all_macrostates,open(f"../{data_dir}/{file_name}.pkl", "wb"))
print("Done with macrostates processing")
#%%
plt.figure(dpi=200)
plt.xlabel("Radial Distance ($\AA$)")
plt.ylabel("g(r)")
for i,plot in enumerate(all_macrostates):
    plt.plot(rmesh[20:50],plot[1][20:50],alpha=0.5)
    # plt.plot(rmesh,plot[3],alpha=0.5,color="blue")
plt.xlim((0,6))
plt.tight_layout()
# plt.savefig(fig_save_path + "second_shell_initial_sampling.jpg")
#%%
# Loading prepared data
all_macrostates = pickle.load(open("../dataset12/onne_macrostates.pkl","rb"))
# Synthetic Data generation for validation and testing
num_special_bois = len(all_macrostates)
exafs_examples, rdf_examples_zip = generate_examples(50000, num_special_bois, all_macrostates)
x_val, x_test, y_val, y_test = train_test_split(exafs_examples, rdf_examples_zip, test_size=0.5)

# Tailor shape to needs
x_val = x_val.reshape(x_val.shape[0], 1 ,x_val.shape[1]) 
x_test = x_test.reshape(x_test.shape[0],1, x_test.shape[1])
#%%
plt.figure(dpi=200)
plt.xlabel("Radial Distance ($\AA$)")
plt.ylabel("g(r)")
for i,plot in enumerate(y_val[:300]):
    plt.plot(rmesh[20:50],plot[:30],alpha=0.5)
    # plt.plot(rmesh,plot[3],alpha=0.5,color="blue")
plt.xlim((0,6))
plt.tight_layout()
#%%
# Saving synthetic validation and testing data
pickle.dump(x_val, open(f"../{data_dir}/x_val_full_rdf.pkl", "wb"))
pickle.dump(x_test, open(f"../{data_dir}/x_test_full_rdf.pkl", "wb"))
pickle.dump(y_val, open(f"../{data_dir}/y_val_full_rdf.pkl", "wb"))
pickle.dump(y_test, open(f"../{data_dir}/y_test_full_rdf.pkl", "wb"))

print("Done with synthetic generation")

# %%
# NNMD Data preparation for training and fine tuning
NNMD_paths = glob("/home/omar/NNMD_FEFF_NaF_ZrF4/results/*_NNMD_arranged/*")
NNMD_data = []
print(len(NNMD_paths))
#%%
i = 0
for path in NNMD_paths:
    try:
        exafs,rdf_F,rdf_Na, rdf_Zr = NNMD_data_preparation(path)
        NNMD_data.append([exafs,rdf_F,rdf_Na,rdf_Zr])
    except TypeError as e:
        i +=1
        print(f"{path}")

print(f"There was {i} empty directories")

NNMD_data = np.array([example for example in NNMD_data if np.max(example[0]) < 2],dtype=object)
pickle.dump(NNMD_data,open(f"../{data_dir}/NNMD_all_traj_all_temps.pkl", "wb"))

NNMD_data = pickle.load(open(f"../{data_dir}/NNMD_all_traj_all_temps.pkl","rb"))
#%% 
# Generating NNMD Validation and testing sets for transfer learning
num_special_bois = len(NNMD_data)
exafs_examples, rdf_examples_zip = generate_examples(50000, num_special_bois, NNMD_data)
x_val, x_test, y_val, y_test = train_test_split(exafs_examples, rdf_examples_zip, test_size=0.5)

x_val = x_val.reshape(x_val.shape[0], 1 ,x_val.shape[1])
x_test = x_test.reshape(x_test.shape[0],1, x_test.shape[1])

pickle.dump(x_val, open(f"../{data_dir}/NNMD_x_val_full_rdf.pkl", "wb"))
pickle.dump(x_test, open(f"../{data_dir}/NNMD_x_test_full_rdf.pkl", "wb"))
pickle.dump(y_val, open(f"../{data_dir}/NNMD_y_val_full_rdf.pkl", "wb"))
pickle.dump(y_test, open(f"../{data_dir}/NNMD_y_test_full_rdf.pkl", "wb"))