import numpy as np
import pandas as pd
import mdtraj as md

def compute_ess(weights):
    """
    Compute Effective Sample Size (ESS) given weights.
    Parameters
    ----------
    weights : array-like
        Array of weights.
    Returns
    -------
    ess : float
        Effective Sample Size (ESS).    
    """
    weights = np.array(weights)  # Ensure array format
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    return ess

def subsample_with_weights(log_df,sample_size=100):
    """
    Subsample a dataframe based on weights.
    Parameters
    ----------
    log_df : pandas.DataFrame
        DataFrame containing weights.
    sample_size : int
    Number of samples to draw.
    Returns
    -------
    subsampled_df : pandas.DataFrame
        Subsampled DataFrame.
    """
    weights = log_df['weights'].values
    weights = np.nan_to_num(weights, nan=0.0)  # replaces NaN with 0
    normalized_weights = weights / np.sum(weights)  # Normalize weights
    indices = np.random.choice(range(len(log_df)), size=sample_size, replace=False, p=normalized_weights)
    subsampled_df = log_df.iloc[indices]
    return subsampled_df

def m_to_n_bootstrap(log_df):
    """
    Perform m-to-n bootstrap subsampling.
    Parameters
    ----------
    log_df : pandas.DataFrame
        DataFrame containing weights.
    Returns
    -------
    subsampled_df : pandas.DataFrame
        Subsampled DataFrame.
    """
    weights = log_df['weights'].values
    normalized_weights = weights / np.sum(weights)  # Normalize weights
    indices = np.random.choice(range(len(log_df)), size=len(log_df), replace=True, p=normalized_weights)
    subsampled_df = log_df.iloc[indices]
    return subsampled_df

def extract_structures(log_df, trajdir):
    """
    Extract structures based on subsampled indices and merge into single mdtraj.Trajectory.
    Parameters
    ----------
    log_df : pandas.DataFrame
        DataFrame containing indices of structures to extract.
    trajdir : str
        Directory containing trajectory files.
    Returns
    -------
    extracted_structures : mdtraj.Trajectory
        Merged trajectory of extracted structures.
    """
