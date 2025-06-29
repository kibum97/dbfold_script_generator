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

def logfiles_to_dataframe(trajdir):
    temperature_re = re.compile(r"(?<=_)\d+\.\d+")
    replica_re = re.compile(r"\d+(?=\.log|\.xtc)")
    time_re = re.compile(r"(?<=\.)[0-9]+")
    title_re = re.compile(r"equil/(.+?)/MCPU")

    print("Processing", trajdir)
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
                    gather['step'].append(float(items[1]))
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
                    gather_replica['step'].append(float(items[1]))
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
    outname = '{}/logfiles_as_dataframe.pkl'.format(trajdir)
    dataframe.to_pickle(outname)
    print(outname, "written to file")
    return dataframe

def initialize_mbar(log_df, k_bias, trajdir, solver_protocol='pymbar3'):
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
    u_kln = energies[:,None,:]/temperatures[:,0][None,:,None]+bias
    print('Initializing MBAR')
    if solver_protocol == 'pymbar3':
        solver_options = {"maximum_iterations":10000,"verbose":True}
        solver_protocol = {"method":"adaptive","options":solver_options}
        mbar = pymbar.MBAR(u_kln, N_k, solver_protocol = (solver_protocol,))
    else:
        mbar = pymbar.MBAR(u_kln, N_k, solver_protocol = (solver_protocol,))
    print('Initialization complete')
    # Saving
    with open(f'{trajdir}/mbar.pkl', 'wb') as pickle_file:
        pickle.dump(mbar, pickle_file)
    return mbar

def initialize_fes(log_df, k_bias, trajdir, solver_protocol='pymbar3'):
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
    u_kln = energies[:,None,:]/temperatures[:,0][None,:,None]+bias
    print('Initializing FES')
    if solver_protocol == 'pymbar3':
        solver_options = {"maximum_iterations":10000,"verbose":True}
        solver_protocol = {"method":"adaptive","options":solver_options}
    mbar_options = {'solver_protocol':(solver_protocol,)}
    fes = pymbar.FES(u_kln, N_k, mbar_options=mbar_options)
    print('Initialization complete')
    # Saving
    with open(f'{trajdir}/fes.pkl', 'wb') as pickle_file:
        pickle.dump(fes, pickle_file)
    return fes

def generate_substructure(native, d_cutoff, min_seq_separation, contact_sep_thresh, min_clustersize,atom='CA', manual_merge=None ,labelsize = 30, fontsize = 30, max_res = None, plot=True, ax = None, native_contacts=[], verbose=False):
    if len(native_contacts)==0:
        coords, resis=read_PDB(native_file, atom)
        #we get a contact map with a min seq separation larger than usual to avoid helices

        native_distances=compute_contacts_matrix(coords, mode='distances', min_seq_separation=min_seq_separation)
        
        native_contacts=np.zeros(np.shape(native_distances))
        native_contacts[np.where((native_distances<d_cutoff) & (native_distances!=0))]=1

    
    positions=np.where(native_contacts==1) #which residues participate in contacts
    positions=np.transpose(positions)
    M=metrics.pairwise.pairwise_distances(positions, metric='manhattan')  #how far is each contact from each other contact?
    
    #To find connected components, I  use my loopCluster function by feeding in the positions ofr the contacts instead of the "files",
    #as well as above matrix M as d_contacts
    
    clusters, pairs_in_substructures, mean_intercluster, mean_intracluster=loopCluster(contact_sep_thresh, positions, M, sort_orphans=False, min_clustersize=min_clustersize, verbose=verbose)


    #pairs in substructures is a list of sublists, each of which correspodns to a given substructure
    #Within a given sublist, there are a bunch of pairs which tell you which pairs of residues belong to that substructure
    
    #The above is in a messy form, so we convert it into a form that allows for numpy indexing,
    #where we have a list of sublists, each sublist contains two arrays, the first of which gives the first indices for the interacting residues
    #pairs_in_substructures=[[np.array(C)[:,0], np.array(C)[:,1]] for C in pairs_in_substructures]
    pairs_in_substructures=[(np.array(C)[:,0], np.array(C)[:,1]) for C in pairs_in_substructures]
    
    
    
    
    nsubstructures=len(pairs_in_substructures)  #we now produce a set of matrices...the ith page tells you which contacts belong to the ith substructure
    substructures=np.zeros((np.shape(native_contacts)[0], np.shape(native_contacts)[1], nsubstructures))
    for n in range(nsubstructures):
        SS=np.zeros(np.shape(native_contacts))
        SS[pairs_in_substructures[n]]=1
        substructures[:,:,n]=SS
    if manual_merge is not None:
        del_idx = []
        for pair in manual_merge:
            substructures[:,:,pair[0]] = np.sum(substructures[:,:,pair],axis=2)
            del_idx.append(pair[1:])
        del_idx = [item for sublist in del_idx for item in sublist]
        substructures = np.delete(substructures,del_idx,axis=2)
    if plot:
        visualize_substructures(native_contacts, substructures, max_res = max_res, ax = ax, labelsize = labelsize, fontsize = fontsize)
    #print(positions)
    return native_contacts, substructures
