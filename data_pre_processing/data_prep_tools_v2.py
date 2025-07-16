#%%
import numpy as np
from glob import glob
from scipy.interpolate import interp1d
# Ovito python API is needed to process MD Data
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder

# Define the k- and r-space as global Variables
kspace = np.arange(2,13.0,.05)
rmesh = np.arange(0.03,6.06,.06)

def read_exafs(file_path):
    '''
    Function that reads a FEFF output file and returns the 
    k and chi values in a numpy array
    '''
    data = np.loadtxt(file_path, skiprows=1)[:,[2,5]]
    return data

def read_feff(file_path):
    '''
    Function that reads and returns a FEFF input file
    '''
    with open(file_path,"r") as f:
        lines = f.readlines()
    lines = [line.replace("\n", "").replace("\t", " ") for line in lines]
    return lines

def make_rdf(distances,rmesh):
    '''
    This function constructs a radial distribution function based on the definition
    g(r) = N/(4 x pi x dr x r^2)

    distances: atomic distances from an absorbing atom
    rmesh: Radial distance grid
    '''
    dr = rmesh[1]-rmesh[0]
    digitized = np.digitize(distances, rmesh) - 1
    gr = np.bincount(digitized[digitized>=0],minlength=len(rmesh))
    gr = gr/(4 * np.pi * dr * rmesh**2)
    return 10*gr

def compute_coordination_number(gr,rmesh,rrange):
    '''
    This function returns the coordination number by integrating
    g(r) = N/(4 x pi x dr x r^2) within a range

    gr: Radial Distribution Function
    rmesh: Radial distance grid
    rrange: Integration range
    '''
    gr = gr/10
    mask = (rmesh >= rrange[0]) & (rmesh <= rrange[1])
    r_selected = rmesh[mask]
    g_r_selected = gr[mask]
    dr = rmesh[1]-rmesh[0]
    # Compute coordination number using numerical integration
    CN = 4 * np.pi * np.trapz(g_r_selected * r_selected**2, r_selected,dr)
    return CN


def interpol(exafs):
    '''
    This function maps any EXAFS data to a new k-space mesh
    '''
    global kspace
    x, y = exafs.transpose()
    f1=interp1d(x, y, kind='cubic')
    test_real=f1(kspace)
    return test_real

def data_preparation(directory):
    '''
    This function prepares the data within a directory by obtaining
    the average RDF and EXAFS of a macrostate. To use with other elements,
    one must update the chemical symbols below
    '''
    xmu_files = glob(directory + "/*/xmu.dat")
    feff_files = [f.replace("xmu.dat", "feff.inp") for f in xmu_files]

    all_rdfs = []
    all_exafs = []

    if xmu_files:
        for i,spectra in enumerate(xmu_files):
            try:
                all_exafs.append(read_exafs(spectra))
                feff_data = read_feff(feff_files[i])

                element_index_map = feff_data[feff_data.index("POTENTIALS")+2:feff_data.index("ATOMS")]
                element_index_map = [(int(item.split()[0]),item.split()[2]) for item in element_index_map]
                element_index_map = np.array(element_index_map,dtype=object)

                coord_starting_index = feff_data.index("ATOMS") + 1
                coord = feff_data[coord_starting_index:-1] 
                coord = np.asarray([line.split() for line in coord])

                indexes = coord[:,3].astype(int)
                atom_coord = coord[:,0:3].astype(float)
                absorber_index = np.where(indexes == 0)[0][0]

                # Update chemical symbols as appropriate and add/remove lines 
                # depending on the number of species you have
                F_coord = atom_coord[np.where(indexes==element_index_map[element_index_map[:,1]=="F",0][0])]
                Na_coord = atom_coord[np.where(indexes==element_index_map[element_index_map[:,1]=="Na",0][0])]
                Zr_coord = atom_coord[np.where(indexes==element_index_map[element_index_map[:,1]=="Zr",0][0])]

                F_distances = np.linalg.norm(F_coord - atom_coord[absorber_index],axis=1)
                Na_distances = np.linalg.norm(Na_coord - atom_coord[absorber_index],axis=1)
                Zr_distances = np.linalg.norm(Zr_coord - atom_coord[absorber_index],axis=1)

                all_rdfs.append(np.array([make_rdf(F_distances,rmesh),
                                              make_rdf(Na_distances,rmesh),
                                              make_rdf(Zr_distances,rmesh)]))
                
            except ValueError as e:
                print(f"Error Processing '{spectra}': {e}")
                print(e.with_traceback())
                continue
    else:
        print("No xmu.dat files found")


    all_exafs_s = np.asarray(all_exafs)
    all_exafs_s = np.asarray([[l for l in s if 2<=l[0]<=13] for s in all_exafs])

    all_exafs_k2 = []
    for spectrum in all_exafs_s:
        x, y = spectrum.transpose()
        all_exafs_k2.append(np.array([x,y*x**2]).transpose())

    all_exafs_k2 = np.asarray(all_exafs_k2)

    bad_exafs_s=[i for i,l in enumerate(all_exafs_k2) if max(l.transpose()[1])>5]
    all_exafs_k2 = np.delete(all_exafs_k2, bad_exafs_s, axis=0)
    all_rdf_s = np.delete(all_rdfs, bad_exafs_s, axis=0)

    all_exafs_final=np.mean(np.asarray(all_exafs_k2),axis=0)
    all_rdf_final=np.mean(np.asarray(all_rdf_s),axis=0)

    rdf_F, rdf_Na, rdf_Zr = all_rdf_final
    return interpol(all_exafs_final),rdf_F, rdf_Na, rdf_Zr

def combinator_s(num_special_bois, special_bois):
    '''
    This function returns the randomly weighted combinations of
    EXAFS and RDFs to be used in the function generate_examples

    num_special_bois: Total number of data points
    special_bois: The rdf data
    '''
    ran_i = np.random.randint(1, 5, 1)
    ran_bois = np.random.choice(num_special_bois, ran_i, replace=False)

    weights = np.random.dirichlet(np.ones(ran_i),size=1)[0]

    bois = special_bois[ran_bois]
    exafs_weighted = np.average([boi[0] for boi in bois], axis=0, weights=weights)
    rdf_F_weighted = np.average([boi[1] for boi in bois], axis=0, weights=weights)
    rdf_Na_weighted = np.average([boi[2] for boi in bois], axis=0, weights=weights)
    rdf_Zr_weighted = np.average([boi[3] for boi in bois], axis=0, weights=weights)

    return exafs_weighted, rdf_F_weighted, rdf_Na_weighted,rdf_Zr_weighted

def generate_examples(num_examples, num_special_bois, special_bois):
    '''
    This function generates a diverse training dataset based on
    exisiting data

    num_examples: Desired number of new data
    num_special_bois: Total number of data points
    special_bois: The rdf data
    '''
    examples = np.zeros((num_examples, 4), dtype=object) # change to 2 if only using 1 rdf
    for i in range(num_examples):
        examples[i] = combinator_s(num_special_bois, special_bois)
    exafs_examples, rdf_examples_F, rdf_examples_Na,rdf_examples_Zr = zip(*examples)
    exafs_examples = np.asarray(exafs_examples)
    rdf_examples_F = np.asarray(rdf_examples_F)
    rdf_examples_Na = np.asarray(rdf_examples_Na)
    rdf_examples_Zr = np.asarray(rdf_examples_Zr)
    rdf_examples_zip = np.array([np.concatenate((rdf_examples_F[i], rdf_examples_Na[i],rdf_examples_Zr[i])) for i in range(len(exafs_examples))])
    return exafs_examples, rdf_examples_zip

def data_generator(batch_size, num_special_bois, special_bois):
    '''
    The data generator constructs training samples by randomly selecting
    three macrostates and computing their linear combination, with the contribution
    of each macrostate determined by randomly assigned weights. Random noise was then
    added to the resulting EXAFS spectra during training allowing for smoother coverage
    of the feature space, and improved the model’s robustness and generalization to 
    experimental uncertainties and structural variations. 

    batch_size: ML batch size
    num_special_bois: Total number of data points
    special_bois: The rdf data
    '''
    while True:
        x_batch_0, y_batch = generate_examples(batch_size, num_special_bois, special_bois)
        x_batch = x_batch_0.reshape(x_batch_0.shape[0], x_batch_0.shape[1], 1)
        n_level = np.random.uniform(low=0.0, high=0.6)
        noise = np.random.normal(loc=0, scale=n_level, size=x_batch.shape)
        yield x_batch + noise, y_batch

def NNMD_data_preparation(directory):
    '''
    Same as data_preparation but designed for using OVITO API for MD data
    This function accounts for periodic boundary conditions of MD data
    '''
    xmu_files = glob(directory + "/*/xmu.dat")
    structure_files = [f.replace("xmu.dat", "structure.xyz") for f in xmu_files]

    all_rdfs = []
    all_exafs = []

    if xmu_files:
        for i,spectra in enumerate(xmu_files):
            try:
                all_exafs.append(read_exafs(spectra))

                pipeline = import_file(structure_files[i])
                data = pipeline.compute()
                ptypes = data.particles.particle_types[:]
                r_max = rmesh[-1]
                finder = CutoffNeighborFinder(r_max, data)
                finder.pbc = True
                for central_index in np.where(ptypes == 3)[0]:  # Zr atoms
                    absorber_F = []
                    absorber_Na = []
                    absorber_Zr = []
                    for neigh in finder.find(central_index):
                        if ptypes[neigh.index] == 1:  # F atoms
                            absorber_F.append(neigh.distance)
                        elif ptypes[neigh.index] == 2:
                            absorber_Na.append(neigh.distance)
                        else:
                            absorber_Zr.append(neigh.distance)
                    F_distances = np.asarray(absorber_F)
                    Na_distances = np.asarray(absorber_Na)
                    Zr_distances = np.asarray(absorber_Zr)
                    all_rdfs.append(np.array([make_rdf(F_distances,rmesh),
                                                            make_rdf(Na_distances,rmesh),
                                                            make_rdf(Zr_distances,rmesh)]))

            except ValueError as e:
                print(f"Error Processing '{spectra}': {e}")
                print(e.with_traceback())
                continue
    else:
        print("No xmu.dat files found")


    all_exafs_s = np.asarray(all_exafs)
    all_exafs_s = np.asarray([[l for l in s if 2<=l[0]<=13 ] for s in all_exafs])

    all_exafs_k2 = []
    for spectrum in all_exafs_s:
        x, y = spectrum.transpose()
        all_exafs_k2.append(np.array([x,y*x**2]).transpose())

    all_exafs_k2 = np.asarray(all_exafs_k2)

    bad_exafs_s=[i for i,l in enumerate(all_exafs_k2) if max(l.transpose()[1])>5]
    all_exafs_k2 = np.delete(all_exafs_k2, bad_exafs_s, axis=0)
    all_rdf_s = np.delete(all_rdfs, bad_exafs_s, axis=0)

    all_exafs_final=np.mean(np.asarray(all_exafs_k2),axis=0)
    all_rdf_final=np.mean(np.asarray(all_rdf_s),axis=0)

    rdf_F, rdf_Na, rdf_Zr = all_rdf_final
    return interpol(all_exafs_final),rdf_F, rdf_Na, rdf_Zr
