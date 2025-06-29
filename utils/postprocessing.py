import numpy as np
import os
import mdtraj as md
import multiprocessing
import natsort
import argparse
import glob
import re
import collections
import pandas as pd
import pymbar
import pickle

temperature_re = re.compile(r"(?<=_)\d+\.\d+")
replica_re = re.compile(r"\d+(?=\.log|\.xtc)")
time_re = re.compile(r"(?<=\.)[0-9]+")
title_re = re.compile(r"equil/(.+?)/MCPU")

def calculate_n_tasks_equil(wildcards, input=None, threads=None, attempt=None):
    # calculate number of workers needed for equil run
    contact_count = count_contacts_rounded(
        "min_pdbs/{}.pdb".format(wildcards.pdbroot))
    nodes_per_temp = int((contact_count / 10) + 1)
    return 20 * nodes_per_temp

def count_contacts_rounded(pdbfile,cutoff):
    # Count number of native contacts and round to 10
    if not os.path.isfile(pdbfile):
        print("count_contacts_rounded: No file at {}".format(pdbfile))
        print("Assuming possible dry run")
        return 0
    protein = dbfold.dbfold.Protein('protein', pdbfile)
    contact_count = protein.count_contacts(d_cutoff=cutoff, min_seq_separation=8)
    contact_count = round(contact_count / 10) * 10
    return contact_count

def calculate_n_tasks_min(wildcards, input=None, threads=None, attempt=None):
    # calculate number of workers needed for minimization run
    contact_count = count_contacts_rounded(
        "in_pdbs/{}.pdb".format(wildcards.pdbroot))
    nodes_per_temp = int((contact_count / 10) + 1)
    return nodes_per_temp  # we don't really need to go to 0

def create_constraint_file(pdbroot,output,constraint):
    f = open(output+"/constraint.txt","w")
    f.write("#Res. i, Res. j, Weight\n")
    for pair in constraint:
        res_i = int(pair[0])-1
        res_j = int(pair[1])-1
        f.write(str(res_i) +"\t"+ str(res_j)+"\t"+str(1)+"\n") #Res. i, Res. j, Weight
    f.close()

def find_lowest_rung(pdbroot):
    globstr = f"MCPU_min/{pdbroot}/MCPU_run/{pdbroot}_T_0.100_*.log"
    emin_files = glob.glob(globstr)
    min_E = []
    for log_file in emin_files:
        print(log_file)
        f = open(log_file)
        for line in f.readlines():
            if line.startswith("Emin:"):
                min_E.append(float(line[5:15]))
                print(float(line[5:15]))
    min_log = emin_files[min_E.index(min(min_E))]
    replica = min_log.split("_")[-1].replace(".log","")
    print(f"Emin pdb file is {pdbroot}_0.100_{replica}_Emin.pdb")
    return f"MCPU_min/{pdbroot}/MCPU_run/{pdbroot}_0.100_{replica}_Emin.pdb"

def convert_pdb_to_xtc(zeroth_file):
    fileroot = zeroth_file[:-2]
    pdb_files = natsort.natsorted(glob.glob("{}.*0".format(fileroot)))
    to_concat = []
    for pdb_file in pdb_files:
        try:
            pdb = md.load_pdb(pdb_file)
        except UnicodeDecodeError:
            print(pdb_file, "UnicodeDecodeError")
            continue
        pdb.time = float(time_re.findall(pdb_file)[-1])
        to_concat.append(pdb)
    traj = md.join(to_concat)
    output_name = fileroot + ".xtc"
    traj.save_xtc(output_name)
    print("Saved", output_name, len(traj), flush=True)
    return

def pdbs_to_xtcs(trajdir):
    n_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if not n_cpus:
        n_cpus = 1
    pool = multiprocessing.Pool(int(n_cpus))
    print(f"Number of CPUs is {n_cpus}")
    ref_files = glob.glob("{}/*.0".format(trajdir))
    pool.map(convert_pdb_to_xtc, ref_files)
    pool.close()
    pool.join()
    return

def logfiles_to_dataframe(trajdir, outname=None, correction=0):
    temperature_re = re.compile(r"(?<=_)\d+\.\d+")
    replica_re = re.compile(r"\d+(?=\.log|\.xtc)")
    time_re = re.compile(r"(?<=\.)[0-9]+")
    title_re = re.compile(r"equil/(.+?)/MCPU")

    print("Processing", trajdir)
    if outname is None:
        outname = f'{trajdir}/logfiles_as_dataframe.pkl'
    gather = collections.defaultdict(list)
    gather_replica = collections.defaultdict(list)
    logfiles = natsort.natsorted(glob.glob(
        "{}/*log".format(trajdir)))
    for logfile in logfiles:
        print("  logfile", logfile)
        temperature = float(temperature_re.findall(logfile)[0])
        replica = int(replica_re.findall(logfile)[0])
        with open(logfile) as f:
            old_accept = 0
            old_reject = 0
            for line in f:
                if line.startswith('STEP '):
                    items = line.split()
                    gather['step'].append(float(items[1]) + correction)
                    gather['energy'].append(float(items[2]))
                    gather['contacts'].append(float(items[3]))
                    gather['rmsd'].append(float(items[4]))
                    gather['ncontacts'].append(float(items[5]))
                    gather['setpoint'].append(float(items[-2]))
                    gather['e_constraint'].append(float(items[-1]))
                    gather['temperature'].append(temperature)
                    gather['replica'].append(replica)
                elif line.startswith('RPLC '):
                    items = line.split()
                    gather_replica['step'].append(float(items[1]) + correction)
                    accept = int(items[-4][:-1])
                    reject = int(items[-1])
                    if accept + reject == 0:
                        rate = 0
                    else:
                        rate = accept / (accept + reject)
                    accept_interval = accept - old_accept
                    reject_interval = reject - old_reject
                    if accept_interval + reject_interval == 0:
                        period_rate = 0
                    else:
                        period_rate = accept_interval / (accept_interval + reject_interval)
                    old_accept = accept
                    old_reject = reject
                    gather_replica['accept'].append(accept)
                    gather_replica['reject'].append(reject)
                    gather_replica['exchangeRate'].append(rate)
                    gather_replica['periodExchangeRate'].append(period_rate)
                    gather_replica['temperature'].append(temperature)
                    gather_replica['replica'].append(replica)
    dataframe = pd.DataFrame(gather)
    dataframe2 = pd.DataFrame(gather_replica)
    dataframe = dataframe.set_index(['temperature', 'replica', 'step'])
    dataframe2 = dataframe2.set_index(['temperature', 'replica', 'step'])
    dataframe = pd.merge(dataframe, dataframe2, left_index=True, right_index=True)
    dataframe.to_pickle(outname)
    print(outname, "written to file")
    return dataframe

def initialize_mbar(log_df, k_bias, trajdir, solver_protocol='pymbar3', **mbar_kwargs):
    # Set up for mbar calculation
    conditions = []
    for cond in zip(log_df.index.get_level_values(0),log_df.index.get_level_values(1)):
        if cond not in conditions:
            conditions.extend([cond])
    # simulation conditions
    n_conditions = len(conditions)
    n_samples = len(log_df.index.get_level_values(2).unique())
    # quntities needed
    energies = np.array(log_df["energy"]).reshape((n_conditions,n_samples))
    temperatures = np.array(log_df.index.get_level_values(0)).reshape((n_conditions,n_samples))
    N_k = np.array([n_samples]*n_conditions)
    natives = np.repeat(np.array(log_df["ncontacts"]).reshape((n_conditions,n_samples))[:,np.newaxis,:],repeats=n_conditions,axis=1)
    setpoints = np.repeat(np.array(log_df["setpoint"]).reshape((n_conditions,n_samples))[np.newaxis,:,:],repeats=n_conditions,axis=0)
    bias = k_bias * (natives-setpoints)**2         
    u_kln = (energies[:,None,:])/temperatures[:,0][None,:,None] + bias
    print(u_kln.shape)
    print('Initializing MBAR')
    if solver_protocol == 'pymbar3':
        solver_options = {"maximum_iterations":10000,"verbose":True}
        solver_protocol = {"method":"adaptive","options":solver_options}
        mbar = pymbar.MBAR(u_kln, N_k, solver_protocol = (solver_protocol,), **mbar_kwargs)
    elif type(solver_protocol) == str:
        mbar = pymbar.MBAR(u_kln, N_k, solver_protocol = solver_protocol, **mbar_kwargs)
    else:
        mbar = pymbar.MBAR(u_kln, N_k, solver_protocol = (solver_protocol,), **mbar_kwargs)
    print('Initialization complete')
    # Saving
    with open(f'{trajdir}/mbar.pkl', 'wb') as pickle_file:
        pickle.dump(mbar, pickle_file)
    return mbar

def initialize_fes(log_df, k_bias, trajdir, solver_protocol='pymbar3', **fes_kwargs):
    # Set up for mbar calculation
    conditions = []
    for cond in zip(log_df.index.get_level_values(0),log_df.index.get_level_values(1)):
        if cond not in conditions:
            conditions.extend([cond])
    # simulation conditions
    n_conditions = len(conditions)
    n_samples = len(log_df.index.get_level_values(2).unique())
    # quntities needed
    energies = np.array(log_df["energy"]).reshape((n_conditions,n_samples))
    temperatures = np.array(log_df.index.get_level_values(0)).reshape((n_conditions,n_samples))
    N_k = np.array([n_samples]*n_conditions)
    natives = np.repeat(np.array(log_df["ncontacts"]).reshape((n_conditions,n_samples))[:,np.newaxis,:],repeats=n_conditions,axis=1)
    setpoints = np.repeat(np.array(log_df["setpoint"]).reshape((n_conditions,n_samples))[np.newaxis,:,:],repeats=n_conditions,axis=0)
    bias = k_bias * (natives-setpoints)**2
    u_kln = (energies[:,None,:]+bias)/temperatures[:,0][None,:,None]
    print('Initializing FES')
    if solver_protocol == 'pymbar3':
        solver_options = {"maximum_iterations":10000,"verbose":True}
        solver_protocol = {"method":"adaptive","options":solver_options}
        mbar_options = {'solver_protocol':(solver_protocol,)}
        fes = pymbar.FES(u_kln, N_k, mbar_options=mbar_options, **fes_kwargs)
    elif type(solver_protocol) == str:
        mbar_options = {'solver_protocol':solver_protocol}
        fes = pymbar.FES(u_kln, N_k, mbar_options=mbar_options, **fes_kwargs)
    else:
        mbar_options = {'solver_protocol':(solver_protocol,)}
        fes = pymbar.FES(u_kln, N_k, mbar_options=mbar_options, **fes_kwargs)
    print('Initialization complete')
    # Saving
    with open(f'{trajdir}/fes.pkl', 'wb') as pickle_file:
        pickle.dump(fes, pickle_file)
    return fes

def m_to_n_bootstrap(log_df, trajdir, m, n):
    sampled_df = log_df.sample(n=m, replace=True)
    sampled_df.to_pickle(f'{trajdir}/logfiles_as_dataframe_bootstrap.pkl')
    
    

def compute_contacts(traj, mode='distances', min_seq_separation=2, sqaureform=True, **kwargs):
    distances, pairs = md.compute_contacts(traj, scheme='ca')
    filter_idx = [i for i,pair in enumerate(pairs) if pair[1] - pair[0] > min_seq_separation]
    distances = distances[:,filter_idx]
    pairs = pairs[filter_idx]
    if mode == 'distances':
        if sqaureform:
            result = md.geometry.squareform(distances, pairs)*10
            return result
        else:
            result = {'pair':pairs, 'distances':distances*10}
            return result
    elif mode == 'contacts':
        dist_cutoff = kwargs.get('dist_cutoff')
        if dist_cutoff is None:
            raise ValueError("dist_cutoff must be provided when mode is 'contacts'")
        contacts = np.zeros(distances.shape)
        contacts[(distances > 0) & (distances < dist_cutoff/10)] = 1
        if sqaureform:
            result = md.geometry.squareform(contacts, pairs)
            return result
        else:
            result = {'pair':pairs, 'contacts':contacts}
            return result
        
def find_lowest_energy_pdb(filedir, verbose=False):
    e_min_files = glob.glob(f"{filedir}/*Emin.pdb")
    if len(e_min_files) == 0:
        print("No Emin files found in", filedir)
        return None
    elif len(e_min_files) == 1:
        return e_min_files[0]
    else:
        log_files = natsort.natsorted(glob.glob(f"{filedir}/*log"))
        min_energies = []
        for log_file in log_files:
            if verbose:
                print(os.path.basename(log_file))
            f = open(log_file)
            for line in f.readlines():
                if line.startswith("Emin:"):
                    min_energies.append(float(line[5:15]))
                    if verbose:
                        print(float(line[5:15]))
        min_log = log_files[min_energies.index(min(min_energies))]
        replica = min_log.split("_")[-1].replace(".log","")
        print(f"Emin pdb file is from {replica} replica")
        for e_min_file in e_min_files:
            if e_min_file.endswith(f"_{replica}_Emin.pdb"):
                print(f"Emin pdb file is {e_min_file}")
                return e_min_file