import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
import mdtraj as md
from sklearn import metrics

def loop_clustering(thresh, files, d_contacts, sort_orphans=False, min_clustersize=1, verbose=True):
    """
    Only keep clusters min_clustersize elements or more 
    Note, this function will (understandably) get very upset and loop forever if any node is completely disconnected from all other nodes
    """
    if d_contacts[1,0]!=d_contacts[0,1]:
        d_contacts=d_contacts+np.transpose(d_contacts)
    contacts_red=np.zeros(np.shape(d_contacts))
    for i, row in enumerate(d_contacts):
        for j, entry in enumerate(row):
            if entry<=thresh: contacts_red[i,j]=1
    
    for j,row in enumerate(contacts_red):
        if np.max(row)==0:
            print('WARNING! Node {} is not connected to any other node...this function is about to go BIZARK!!'.format(j))
    
    #We start with the 0th structure. the goal is to find all nodes in the network that can be reached starting the 0th node entirely by traversing edges whose weight is greater than thresh
    
    #First, we find all nodes that only require one such passage
    
    
    clusters=[]
    clustered_indices=[]
    unsorted_indices=np.array(range(len(files)))
    
    
    
    while len(unsorted_indices)>0:
        curr_cluster=np.unique(np.nonzero(contacts_red[unsorted_indices[0],:])[0])
        
        
        
        new_layer=curr_cluster
        curr_len=len(curr_cluster)        
        new_len=0
        
        
        while new_len != curr_len:
            curr_len=len(curr_cluster)
            for node in new_layer:
                new_layer=np.append(new_layer, np.nonzero(contacts_red[node,:])[0])
            new_layer=np.unique(new_layer)
            
            new_layer=np.array([x for x in new_layer if not np.any(curr_cluster==x)])
            
            curr_cluster=np.append(curr_cluster, new_layer)
            new_len=len(curr_cluster)
        
        
        curr_cluster=np.sort(curr_cluster)
        
        clusters.append(list(curr_cluster))
        clustered_indices.extend(curr_cluster)
        
        
        #clusters_zipped=[]
        unsorted_indices=np.array([x for x in range(len(files)) if x not in clustered_indices ])
    
    
    singlets=[x for x in clusters if len(x)==1]   #clusters with only one file--these are set aside
    clusters=[x for x in clusters if len(x)>=min_clustersize] #only keep the "fat" clusters with more than min_clustersize files
    
    clusters_tally=np.zeros((len(files), len(clusters)))   #note the assignments in a matrix: rows are files, columns are fat clusters

    for n,x in enumerate(clusters):
        for y in x:
            clusters_tally[int(y),n]=1
    
    clustered_indices=np.where(np.sum(clusters_tally, axis=1)==1)[0]
    #unclustered_indices=np.where(np.sum(clusters_tally, axis=1)==0)[0]
    
    
    if sort_orphans:
        spots_for_singlets=np.zeros((len(files), len(clusters)))
        
        for n, singlet in enumerate(singlets):
            row=d_contacts[int(singlet[0]),:]
            indices=np.where(row==np.min(row[clustered_indices]))[0]   #yields all indices in row such that row(index)=minimum distance to current file among those files that have been classified 
            best_index=[i for i in indices if i in clustered_indices][0]    # we want
            cluster_assignment=np.where(clusters_tally[best_index,:]==1)[0][0]
        
            spots_for_singlets[int(singlet[0]), cluster_assignment]=1
        
        clusters_tally=clusters_tally+spots_for_singlets
        
    
    clusters=[]
    clustered_files=[]
    for n, col in enumerate(clusters_tally.T):
        clusters.append([i for i in range(len(col)) if col[i]!=0])
        clustered_files.append([files[i] for i in clusters[n]])
        
        if verbose: print("Cluster {}: {} \n {} files total \n".format(n, clustered_files[n], sum(col)))
        
    #go through all pairs of files and figure out if two members of pair are in the same cluster or not.
    #If they are, add the distanec between these two files to the variable intracluster distances
    #Otherwise, add this distnce to intercluster_distances
    intracluster_distances=[]
    intercluster_distances=[]
    for i in range(len(d_contacts)):
        for j in range(i+1):
            if  np.size(np.where(clusters_tally[i,:])[0])!=0 and np.size(np.where(clusters_tally[j,:])[0])!=0:
                if np.where(clusters_tally[i])[0]==np.where(clusters_tally[j])[0]:  #they are in the same cluster
                    intracluster_distances.append(d_contacts[i,j])
                else:
                    intercluster_distances.append(d_contacts[i,j])
    
    mean_intracluster=np.mean(intracluster_distances)
    mean_intercluster=np.mean(intercluster_distances)
        
    if verbose: print('Mean distance within clusters: {}'.format(mean_intracluster))
    if verbose: print('Mean distance between clusters: {}'.format(mean_intercluster))
    

    return clusters, clustered_files, mean_intercluster, mean_intracluster

def read_coords(traj, atom='CA'):
    traj = traj.atom_slice(traj.top.select(f'name {atom}'))
    coords = traj.xyz[0]
    return coords

def compute_contacts(traj, mode='distances', min_seq_separation=2):
    distances, pairs = md.compute_contacts(traj, scheme='ca')
    filter_idx = [i for i,pair in enumerate(pairs) if pair[1] - pair[0] > 7]
    distances = distances[:,filter_idx]
    pairs = pairs[filter_idx]
    contacts = md.geometry.squareform(distances, pairs)*10
    return contacts[0]

def generate_substructures(native_file, d_cutoff, min_seq_separation, contact_sep_thresh, min_clustersize,atom='CA', manual_merge=None ,labelsize = 30, fontsize = 30, max_res = None, plot=False, ax = None, native_contacts=[], verbose=False):
    if len(native_contacts)==0:
        traj = md.load(native_file)
        coords = read_coords(traj, atom)
        #we get a contact map with a min seq separation larger than usual to avoid helices
   
        native_distances=compute_contacts(traj, mode='distances', min_seq_separation=min_seq_separation)
        
        native_contacts=np.zeros(np.shape(native_distances))
        native_contacts[np.where((native_distances<d_cutoff) & (native_distances!=0))]=1

    
    positions=np.where(native_contacts==1) #which residues participate in contacts
    positions=np.transpose(positions)
    M=metrics.pairwise.pairwise_distances(positions, metric='manhattan')  #how far is each contact from each other contact?
    
    #To find connected components, I  use my loopCluster function by feeding in the positions ofr the contacts instead of the "files",
    #as well as above matrix M as d_contacts
    
    clusters, pairs_in_substructures, mean_intercluster, mean_intracluster=loop_clustering(contact_sep_thresh, positions, M, sort_orphans=False, min_clustersize=min_clustersize, verbose=verbose)


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
    return native_distances, substructures

def load_scores(scores, native_distances, substructures, thresh, convert_to_binary=True):
    
    mean_substructure_distances=[]
    for i in range(np.shape(substructures)[2]):
        x=np.multiply(substructures[:,:,i], native_distances)
        mean_substructure_distances.append(np.nanmean(x[np.where(x)]))
    mean_substructure_distances=np.array(mean_substructure_distances)
    if convert_to_binary:
        data = scores/mean_substructure_distances
        barcodes=[]
        for i in range(len(data)):
            string=''
            for j in range(len(data[i,:])):
                if data[i,j]<=thresh:
                    string='{}1'.format(string)
                else:
                    string='{}0'.format(string)
            barcodes.append(string)
        scores=barcodes

    return scores

def dec2bin(decimal_number):
    return bin(decimal_number)[2:]

def bin2dec(binary_number):
    return int(binary_number, 2)