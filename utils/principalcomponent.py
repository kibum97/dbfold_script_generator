from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import mdtraj as md
import numpy as np

def compute_contacts(traj, mode='distances', min_seq_separation=2, sqaureform=True, **kwargs):
    distances, pairs = md.compute_contacts(traj, scheme='ca')
    filter_idx = [i for i,pair in enumerate(pairs) if pair[1] - pair[0] > min_seq_separation]
    distances = distances[:,filter_idx]
    pairs = pairs[filter_idx]
    if mode == 'distances':
        if sqaureform:
            result = md.geometry.squareform(distances, pairs)*10
            return result[0]
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
            return result[0]
        else:
            result = {'pair':pairs, 'contacts':contacts}
            return result

def compute_contacts_all(xtc_list, top, eq_step, mode='distances', dist_cutoff=7, min_seq_separation=6):
    """
    Compute contacts for all trajectories
    Parameters
    ----------
    xtc_list : list
        List of xtc files
    mode : str, optional
        Mode of contacts, either 'distances' or 'contacts'
    dist_cutoff : float, optional
        Distance cutoff for contacts
    min_seq_separation : int, optional
        Minimum sequence separation for contacts
    Returns
    -------
    contacts : array
        Contacts for all trajectories
    """
    contacts_list = []
    for xtc in xtc_list:
        traj = md.load(xtc, top=top)
        traj = traj[traj.time >= eq_step]
        traj = traj.atom_slice(traj.top.select('name CA'))
        contacts_list.append(compute_contacts(traj, mode=mode,
                                              min_seq_separation=min_seq_separation,
                                              dist_cutoff=dist_cutoff,
                                              sqaureform=False)[mode])
    contacts = np.vstack(contacts_list)
    return contacts

def run_pca(data, n_components=8):
    """
    Compute pairwise distance matrix and perform PCA
    Parameters
    ----------
    xtc_list : list
        List of xtc files
    n_components : int, optional
        Number of components to keep
    Returns
    -------
    pca : PCA
        PCA object
    """
    standard_scaler = StandardScaler()
    data = standard_scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    print("Explained variance ratio:", np.sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_)
    return pca

def dbscan_with_pca(pca, data, eps=1, min_samples=50):
    """
    Perform DBSCAN on PCA components and compute clustering quality metrics.
    Parameters
    ----------
    pca : PCA
        PCA object
    data : array
        Data before PCA transformation
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point
    Returns
    -------
    dbscan : DBSCAN
        DBSCAN object
    silhouette : float
        Silhouette score of the clustering
    davies_bouldin : float
        Davies-Bouldin index of the clustering
    """
    standard_scaler = StandardScaler()
    data = standard_scaler.fit_transform(data)
    transformed_data = pca.transform(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(transformed_data)
    
    labels = dbscan.labels_
    
    # Filter out noise
    clustered_data = transformed_data[labels != -1]
    clustered_labels = labels[labels != -1]
    # Compute silhouette score
    silhouette = silhouette_score(clustered_data, clustered_labels)
    # Compute Davies-Bouldin index
    davies_bouldin = davies_bouldin_score(clustered_data, clustered_labels)

    print("Number of noise points:", np.sum(labels == -1))
    print("Silhouette score:", silhouette)
    print("Davies-Bouldin index:", davies_bouldin)
    
    return dbscan, silhouette, davies_bouldin
