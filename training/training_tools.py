#%%
import numpy as np
from glob import glob
from scipy.interpolate import interp1d

# Global Variables
kspace = np.arange(2,13.0,.05)
rmesh = np.arange(0.03,6.0,.06)

def compute_coordination_number(gr,rmesh,rrange):
    gr = gr/10
    mask = (rmesh >= rrange[0]) & (rmesh <= rrange[1])
    r_selected = rmesh[mask]
    g_r_selected = gr[mask]
    dr = rmesh[1]-rmesh[0]
    # Compute coordination number using numerical integration
    CN = 4 * np.pi * np.trapz(g_r_selected * r_selected**2, r_selected,dr)
    
    return CN

def interpol(exafs):
    global kspace
    x, y = exafs.transpose()
    f1=interp1d(x, y, kind='cubic')
    test_real=f1(kspace)
    return test_real

def combinator_s(num_special_bois, special_bois):
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
    while True:
        x_batch_0, y_batch = generate_examples(batch_size, num_special_bois, special_bois)
        x_batch = x_batch_0.reshape(x_batch_0.shape[0], x_batch_0.shape[1], 1)
        n_level = np.random.uniform(low=0.0, high=0.6)
        noise = np.random.normal(loc=0, scale=n_level, size=x_batch.shape)
        yield x_batch + noise, y_batch