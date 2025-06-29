import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cccc
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap
import scipy
from scipy import linalg
import mdtraj as md
from sklearn import metrics
from sklearn.cluster import DBSCAN
from copy import deepcopy

# TODO: figure out how to pass the plot keywrod arguments to the plot function

def read_coords(traj, atom='CA'):
    traj = traj.atom_slice(traj.top.select(f'name {atom}'))
    coords = traj.xyz[0]
    return coords

def clusterings(coords, thresh, min_clustersize=1):
    db = DBSCAN(eps=thresh, min_samples=min_clustersize,metric='minkowski',p=1).fit(coords)
    return db.labels_

def compute_contacts(traj, mode='distances', min_seq_separation=2, squareform=True, **kwargs):
    distances, pairs = md.compute_contacts(traj, scheme='ca')
    filter_idx = [i for i,pair in enumerate(pairs) if pair[1] - pair[0] > min_seq_separation]
    distances = distances[:,filter_idx]
    pairs = pairs[filter_idx]
    if mode == 'distances':
        if squareform:
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
        if squareform:
            result = md.geometry.squareform(contacts, pairs)
            return result
        else:
            result = {'pair':pairs, 'contacts':contacts}
            return result
    

def generate_substructures(native_file, d_cutoff, min_seq_separation, contact_sep_thresh, min_clustersize,atom_type='CA', verbose=False):
    """
    Generates substructures based on contact distances in a trajectory.

    Parameters:
    native_file (str): Path to the native structure file.
    d_cutoff (float): Distance cutoff for contacts.
    min_seq_separation (int): Minimum sequence separation between pairs.
    contact_sep_thresh (float): Threshold for contact separation.
    min_clustersize (int): Minimum cluster size.
    atom_type (str): Atom type to consider (default is 'CA').
    verbose (bool): Whether to print verbose output (default is False).

    Returns:
    tuple: Native distances and substructures.
    """
    # Compute pairwise distances and contacts of native structure
    traj = md.load(native_file)
    native_distances = compute_contacts(traj, mode='distances', min_seq_separation = min_seq_separation)[0]
    native_contacts_dict = compute_contacts(traj, mode='contacts', min_seq_separation = min_seq_separation, dist_cutoff = d_cutoff, squareform=False)
    native_contacts = native_contacts_dict['contacts']
    native_pairs = native_contacts_dict['pair']
    positions = native_pairs[native_contacts[0] == 1]

    # Cluster contacts and define substructures
    cluster_labels = clusterings(positions, contact_sep_thresh, min_clustersize=min_clustersize)
    
    # Save substructures in corresponding variable
    substructures = []
    print(f'Number of substructures: {cluster_labels.max()+1}')
    for n in range(cluster_labels.max()+1):
        substructure_temp = np.zeros(np.shape(native_distances))
        substructure_temp[positions[cluster_labels == n,0],positions[cluster_labels == n,1]] = 1
        substructures.append(substructure_temp)
    substructures = np.stack(substructures, axis=-1)

    # Sort substructures for better visualization
    min_indices = np.amax(np.argmax(substructures == 1, axis= 1), axis= 0)
    sorted_indices = np.argsort(min_indices)
    substructures = substructures[:, :, sorted_indices]

    return native_distances, substructures

def load_substructures(native_file, substructure_dict, min_seq_separation):
    traj = md.load(native_file)
    native_distances = compute_contacts(traj, mode='distances', min_seq_separation = min_seq_separation)[0]

    # Load predefined substructures
    substructures = []
    for key, value in substructure_dict.items():
        substructure_temp = np.zeros(np.shape(native_distances))
        for i, j in value:
            substructure_temp[i, j] = 1
        substructures.append(substructure_temp)
    substructures = np.stack(substructures, axis=-1)

    return native_distances, substructures

def load_scores(scores, native_distances, substructures, thresh, convert_to_binary=True, **kwargs):
    
    mean_substructure_distances=[]
    for i in range(np.shape(substructures)[2]):
        x=np.multiply(substructures[:,:,i], native_distances)
        mean_substructure_distances.append(np.nanmean(x[np.where(x)]))
    mean_substructure_distances=np.array(mean_substructure_distances)
    mean_substructure_distances = kwargs.get('mean_substructure_distances', mean_substructure_distances)
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

def dec2bin(decimal_number,n_subs=None):
    if n_subs is not None:
        max_len = n_subs
        return bin(decimal_number)[2:].zfill(max_len)
    else:
        return bin(decimal_number)[2:]

def bin2dec(binary_number):
    return int(binary_number, 2)

def pairwise_l1_distance(binary_strings):
    # Ensure all binary strings are of the same length
    max_len = max(len(bin_str) for bin_str in binary_strings)
    padded_binary_strings = [bin_str.zfill(max_len) for bin_str in binary_strings]
    
    # Initialize the distance matrix
    n = len(binary_strings)
    distance_matrix = [[0] * n for _ in range(n)]
    
    # Compute the pairwise L1 distances
    for i in range(n):
        for j in range(i + 1, n):
            distance = sum(abs(int(bit_a) - int(bit_b)) for bit_a, bit_b in zip(padded_binary_strings[i], padded_binary_strings[j]))
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Distance is symmetric
    
    return distance_matrix

def plot_substructure_fes(fes_result, save_file, uncertainty=False, **kwargs):
    fe_dict, features = fes_result
    labels = kwargs.get('labels', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ymax = kwargs.get('ymax', float('inf'))

    # Subset fe_dict and features to only include substructures with free energy <= ymax
    subset_dict = {'f_i':[], 'df_i':[]}
    subset_features = []
    for i, feat in enumerate(features):
        if fe_dict['f_i'][i] <= ymax:
            subset_features.append(feat)
            subset_dict['f_i'].append(fe_dict['f_i'][i])
            if uncertainty:
                subset_dict['df_i'].append(fe_dict['df_i'][i])
       
    max_len = len(dec2bin(max(features)))
    bin_subs = [dec2bin(sub).zfill(max_len) for sub in subset_features]
    len_subs = [bin_sub.count('1') for bin_sub in bin_subs]
    min_len = min(len_subs)
    max_len = max(len_subs)
    n_subs = [sum(int(bit) for bit in bin_sub) for bin_sub in bin_subs]
    jitter = 0.1 * np.random.randn(len(bin_subs))

    pdist_matrix = pairwise_l1_distance(bin_subs)

    fig, ax = plt.subplots()
    for i, sub in enumerate(bin_subs):
        if uncertainty:
            ax.errorbar(
                n_subs[i]+jitter[i], subset_dict['f_i'][i], yerr=subset_dict['df_i'][i],
                fmt='o', color='tab:blue', capsize=3, elinewidth=1, markersize=5
            )
        else:
            ax.scatter(n_subs[i]+jitter[i], subset_dict['f_i'][i], color='tab:blue')
        label = ''.join(labels[i] for i in range(len(sub)) if sub[i] == '1')
        if subset_dict['f_i'][i] <= ymax:
            if label:
                ax.text(n_subs[i]+jitter[i]+0.05, subset_dict['f_i'][i]+0.01, label, va='bottom', ha='center')
            else:
                ax.text(n_subs[i]+jitter[i]+0.05, subset_dict['f_i'][i]+0.01, '\u2205', va='bottom', ha='center')

    for i in range(len(bin_subs)):
        for j in range(i + 1, len(bin_subs)):
            if pdist_matrix[i][j] == 1:
                ax.plot([n_subs[i]+jitter[i], n_subs[j]+jitter[j]], [subset_dict['f_i'][i], subset_dict['f_i'][j]], ':', lw=1., color='gray')

    for i in range(max_len+1):
        ax.axvspan(i - 0.5, i + 0.5, facecolor='grey' if i % 2 else 'white', alpha=0.1)

    plt.xticks(range(max_len+1), range(max_len+1))
    ax.tick_params(axis='x', which='both', length=0)
    plt.xlim(min_len-0.5, max_len+0.5)
    plt.xlabel('Number of substructure formed')
    plt.ylabel('Free energy (kT)')
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

def create_substructure_PML(PML_path, substructures, labels=None):
    if not labels:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
    else:
        alphabet = labels
    n_subs = substructures.shape[2]
    file = open(PML_path, 'w')
    file.write('bg white \n color gray \n')
    colors=cm.get_cmap('jet')
    
    for subs_idx in range(n_subs):
        if n_subs > 1:
            curr_color = colors(subs_idx/(n_subs-1))
        else:
            curr_color = colors(0)
        c_hex = cccc.to_hex(curr_color)
        c_hex = '0x{}'.format(c_hex.split('#')[1])
        sub = substructures[:,:,subs_idx]
        contacts = np.where(sub)   
        substr = 'sub{}'.format(alphabet[subs_idx])
        for c in range(len(contacts[0])):
            i = contacts[0][c]+1
            j = contacts[1][c]+1
            file.write(f'select aa, //resi {i}/CA \n')
            file.write(f'select bb, //resi {j}/CA \n')
            file.write(f'distance {substr}, aa, bb \n')
            file.write(f'hide labels, {substr} \n')
            file.write(f'set dash_color, {c_hex}, {substr} \n')
        file.write('\n set dash_gap, 0.5 \n  set dash_radius, 0.2 \n')
    file.close()


def visualize_substructures(native_distances, substructures, dist_cutoff, min_seq_separation, labels=None, onlylower=False, savepath=None):
    if not labels:
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    else:
        alphabet = labels
    native_contacts = (native_distances < dist_cutoff).astype(int)
    n_subs = substructures.shape[2]
    fig, ax = plt.subplots(figsize=(20, 20), facecolor='none')
    colors=cm.get_cmap('jet')
    color_map = [(1, 1, 1, 1)]

    # Assign each substructure a unique label and color    
    substructure_map = np.zeros(native_distances.shape)
    for subs_idx in range(n_subs):
        sub = substructures[:, :, subs_idx]
        contacts = np.where(sub)
        substructure_map[contacts] = subs_idx + 1
        if n_subs > 1:
            curr_color = colors(subs_idx/(n_subs-1))
        else:
            curr_color = colors(0)
        color_map.append(curr_color)
    for i in range(len(substructure_map)):
        for j in range(i+1, len(substructure_map)):
            value = max(substructure_map[i, j], substructure_map[j, i])
            substructure_map[i, j] = value
            substructure_map[j, i] = value

    # Map unassigned native contacts and add color if any
    unassigned_contacts = np.where((substructure_map == 0) & native_contacts)
    count = 0
    for i, j in zip(*unassigned_contacts):
        if abs(i-j) > min_seq_separation:
            substructure_map[i, j] = n_subs + 1
            substructure_map[j, i] = n_subs + 1
            count += 1
    if count > 0:
        color_map.append((0.5, 0.5, 0.5, 1))
    cmap = ListedColormap(color_map)

    # Plot the heatmap
    if onlylower:
        mask = np.tril(np.ones_like(substructure_map, dtype=bool))
        substructure_map = np.where(mask, substructure_map, np.nan)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    cax = ax.imshow(substructure_map, cmap=cmap)

    # Add grey border line around non-np.nan values
    for i in range(substructure_map.shape[0]):
        for j in range(substructure_map.shape[1]):
            if not np.isnan(substructure_map[i, j]):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='lightgrey', linewidth=0.)
                ax.add_patch(rect)
    for i in range(substructure_map.shape[0]):
        for j in range(substructure_map.shape[1]):
            if not np.isnan(substructure_map[i, j]):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='lightgrey', linewidth=0.)
                ax.add_patch(rect)
            if (i == j) and onlylower:
                # Add top and right border for cells where i == j
                ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color='black', linewidth=ax.spines['bottom'].get_linewidth())  # Top border
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color='black', linewidth=ax.spines['left'].get_linewidth())  # Right border

    # Add text annotations
    for subs_idx in range(n_subs):
        sub = substructures[:, :, subs_idx]
        annot_coord = np.max(np.where(sub),axis=1)
        text = ax.text(
            annot_coord[0]+5, annot_coord[1]-3, f'{alphabet[subs_idx]}', 
            ha='center', va='center', color=color_map[subs_idx+1], fontweight='bold',fontsize=24
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),
            path_effects.Normal()
        ])
    
    # Set x-ticks and y-ticks to be 1-indexed
    ax.set_xticks(np.arange(0, substructure_map.shape[1], 10))
    ax.set_yticks(np.arange(0, substructure_map.shape[0], 10))
    ax.set_xticklabels(np.arange(1, substructure_map.shape[1] + 1, 10), rotation=45, fontdict={'fontsize': 14})
    ax.set_yticklabels(np.arange(1, substructure_map.shape[0] + 1, 10), fontdict={'fontsize': 14})

    plt.xlabel('Residue Index', fontsize=18)
    plt.ylabel('Residue Index', fontsize=18)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, facecolor='none', edgecolor='none', transparent=True)
        plt.close()
    else:
        plt.show()
        plt.close()
