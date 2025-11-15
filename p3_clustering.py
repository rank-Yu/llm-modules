import argparse
import os
import pickle as pkl
import time

import random
import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, SpectralCoclustering
from tabulate import tabulate

EPSILON = 1e-8

def print_dict(d: dict):
    """Prints a dictionary in table format"""
    print(tabulate([(k, v) for k, v in d.items()], tablefmt="grid"))

def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="Qwen2.5-3B-Instruct") # LLM name
    parser.add_argument("--activation_matrix_dir", type=str, default="./activation_matrix/") # Directory to save activation value files
    parser.add_argument("--n_clusters", type=int, default=20) # Specify the number of clusters
    parser.add_argument("--sample_labels_dir", type=str, default="sample_labels") # Directory to store sample clustering results
    parser.add_argument("--unit_labels_dir", type=str, default="unit_labels")     # Directory to store unit clustering results
    parser.add_argument("--pca_dim", type=int, default=16) # Add PCA dimension parameter
    parser.add_argument("--seed", type=int, default=4) # Random seed
    args = parser.parse_args()

    print_dict(vars(args))
    return args

if __name__ == "__main__":
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load Data ---
    print("Loading activation matrix...")
    start_load = time.time()
    activation_matrix_filename = f"{args.llm_name}_train_activation_matrix.pkl"
    activation_matrix_path = os.path.join(args.activation_matrix_dir, activation_matrix_filename)

    activation_matrix = pkl.load(open(activation_matrix_path, "rb"))

    print(f"Loaded matrix shape: {activation_matrix.shape}")
    print(f"Loading took {time.time() - start_load:.2f} seconds.")

    # --- Create Output Directories ---
    os.makedirs(args.sample_labels_dir, exist_ok=True)
    os.makedirs(args.unit_labels_dir, exist_ok=True)

    num_samples = activation_matrix.shape[0]
    num_units = activation_matrix.shape[1]

    # --- PCA for Units ---
    print("\n--- PCA for Units ---")
    matrix_T = activation_matrix.T
    print(f"Original unit data shape (transposed): {matrix_T.shape}")

    print(f"Performing PCA on units (reducing sample dimension to {args.pca_dim})...")
    start_pca_time = time.time()
    matrix_mean = matrix_T.mean(dim=0, keepdim=True)

    U_pca, S_pca, V_pca = torch.pca_lowrank(matrix_T, q=args.pca_dim, center=True)
    unit_coords = torch.matmul(matrix_T - matrix_mean, V_pca)

    unit_coords_np = unit_coords.numpy()
    print(f"Unit PCA took {time.time() - start_pca_time:.2f} seconds.")
    print(f"Shape after PCA (unit scores): {unit_coords_np.shape}")

    # --- Preclustering ---
    n_preclusters = 10000
    
    print(f"\nPerforming preclustering {n_preclusters=}...")
    start_precluster_time = time.time()

    kmeans = KMeans(n_clusters=n_preclusters, random_state=args.seed, init="random")
    precluster_labels = kmeans.fit_predict(unit_coords_np)
    precluster_centers = kmeans.cluster_centers_
    # get precluster centers
    precluster_matrix = torch.zeros((n_preclusters, num_samples))
    for i in range(n_preclusters):
        mask = (precluster_labels == i)
        if torch.any(torch.tensor(mask)):
            precluster_matrix[i, :] = matrix_T[mask, :].mean(dim=0)
        else:
            assert False, f"Precluster {i} is empty."
    print(f"  Clustering (preclustering) took {time.time() - start_precluster_time:.2f} seconds.")

    # --- A. Coclustering ---
    for name, model in [("spectral_cocluster", SpectralCoclustering(n_clusters=args.n_clusters, random_state=args.seed))]:
        print(f"\nPerforming {name} (k={args.n_clusters}) on preclustered data...")
        start_cocluster_time = time.time()

        model.fit(precluster_matrix.numpy())

        # Get row and column cluster labels
        row_cluster_labels = model.row_labels_  # precluster labels
        col_cluster_labels = model.column_labels_  # sample labels

        print(f"{name} took {time.time() - start_cocluster_time:.2f} seconds.")

        # Map precluster labels to final unit labels
        unit_labels_np = row_cluster_labels[precluster_labels]
        sample_labels_np = col_cluster_labels

        # Convert to torch tensors
        unit_labels_torch = torch.tensor(unit_labels_np, dtype=torch.int64)
        sample_labels_torch = torch.tensor(sample_labels_np, dtype=torch.int64)

        # Save results
        sample_filename = f"{args.llm_name}_{name}_{args.n_clusters}_samples.pkl"
        sample_filepath = os.path.join(args.sample_labels_dir, sample_filename)
        print(f"  Saving derived sample labels to {sample_filepath}")
        pkl.dump(sample_labels_torch.cpu(), open(sample_filepath, "wb"))

        unit_filename = f"{args.llm_name}_{name}_{args.n_clusters}_units.pkl"
        unit_filepath = os.path.join(args.unit_labels_dir, unit_filename)
        print(f"  Saving unit labels to {unit_filepath}")
        pkl.dump(unit_labels_torch.cpu(), open(unit_filepath, "wb"))
    
    # --- B. Clustering ---
    for name in ["kmeans", "minibatch_kmeans", "agglomerative", "spectral"]:
        # 1. Cluster Units
        print(f"  Clustering units (shape after PCA: {unit_coords_np.shape})...")
        print(f"\nPerforming {name} (k={args.n_clusters})...")
        start_cluster_time = time.time()

        if name == "kmeans":
            model = KMeans(n_clusters=args.n_clusters, random_state=args.seed)
            unit_labels_np = model.fit_predict(unit_coords_np)
            unit_labels_torch = torch.tensor(unit_labels_np, dtype=torch.int64)
        elif name == "minibatch_kmeans":
            model = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed)
            unit_labels_np = model.fit_predict(unit_coords_np)
            unit_labels_torch = torch.tensor(unit_labels_np, dtype=torch.int64)
        elif name =="agglomerative":
            model = AgglomerativeClustering(n_clusters=args.n_clusters)
            unit_labels_for_precluster = model.fit_predict(precluster_centers)
            # Map the original data points' pre-cluster labels to the final cluster labels.
            unit_labels_np = unit_labels_for_precluster[precluster_labels]
            unit_labels_torch = torch.tensor(unit_labels_np, dtype=torch.int64)
        elif name == "spectral":
            model = SpectralClustering(n_clusters=args.n_clusters, random_state=args.seed, affinity='nearest_neighbors', n_neighbors=9)
            unit_labels_for_precluster = model.fit_predict(precluster_centers)
            # Map the original data points' pre-cluster labels to the final cluster labels.
            unit_labels_np = unit_labels_for_precluster[precluster_labels]
            unit_labels_torch = torch.tensor(unit_labels_np, dtype=torch.int64)

        # 2. Derive Sample Labels
        print("  Deriving sample labels...")
        # mean_activation_for_sample_in_unit_cluster: (n_clusters, num_samples)
        mean_activation_for_sample_in_unit_cluster = torch.zeros((args.n_clusters, num_samples), device=activation_matrix.device)
        for k_cluster in range(args.n_clusters):
            units_in_cluster_k_mask = (unit_labels_torch == k_cluster)
            if torch.any(units_in_cluster_k_mask):
                mean_activation_for_sample_in_unit_cluster[k_cluster, :] = activation_matrix[:, units_in_cluster_k_mask].mean(dim=1)
            else:
                print(f"  Warning: For method {name}, unit cluster {k_cluster} is empty. Samples will not be assigned to this cluster unless other clusters also have -inf mean activation.")
                mean_activation_for_sample_in_unit_cluster[k_cluster, :] = -torch.inf

        derived_sample_labels_torch = torch.argmax(mean_activation_for_sample_in_unit_cluster, dim=0)

        print(f"  Clustering ({name}) took {time.time() - start_cluster_time:.2f} seconds.")

        # 3. Save results
        sample_filename = f"{args.llm_name}_{name}_{args.n_clusters}_samples.pkl"
        sample_filepath = os.path.join(args.sample_labels_dir, sample_filename)
        print(f"  Saving derived sample labels to {sample_filepath}")
        pkl.dump(derived_sample_labels_torch.cpu(), open(sample_filepath, "wb"))

        unit_filename = f"{args.llm_name}_{name}_{args.n_clusters}_units.pkl"
        unit_filepath = os.path.join(args.unit_labels_dir, unit_filename)
        print(f"  Saving unit labels to {unit_filepath}")
        pkl.dump(unit_labels_torch.cpu(), open(unit_filepath, "wb"))

    print("\nAll clustering processes completed.")
