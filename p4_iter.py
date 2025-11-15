import argparse
import os
import pickle as pkl
import random
import time

import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from tabulate import tabulate
from tqdm import tqdm

EPSILON = 1e-8

def print_dict(d: dict):
    """Prints a dictionary in a table format"""
    print(tabulate([(k, v) for k, v in d.items()], tablefmt="grid"))

def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="Qwen2.5-3B-Instruct") # LLM name
    parser.add_argument("--activation_matrix_dir", type=str, default="./activation_matrix/") # Directory to save activation files
    parser.add_argument("--n_clusters", type=int, default=20) # Number of clusters
    parser.add_argument("--sample_labels_dir", type=str, default="sample_labels") # Directory to store sample clustering results
    parser.add_argument("--unit_labels_dir", type=str, default="unit_labels") # Directory to store unit clustering results
    parser.add_argument("--initial_clustering_method", type=str, default="balanced_kmeans",
        choices=["kmeans", "kmeans++", "minibatch_kmeans", "minibatch_kmeans++", "agglomerative", "balanced_kmeans"]) # Clustering method name for initial sample clustering
    parser.add_argument("--pca_dim", type=int, default=16) # PCA dimension for initial sample clustering
    parser.add_argument("--max_iter", type=int, default=5) # Maximum number of iterations for iterative clustering
    parser.add_argument('--seed', type=int, default=1) # Random seed for reproducibility

    args = parser.parse_args()
    print_dict(vars(args))
    return args

def calculate_metric_components(activation_matrix, sample_labels, unit_labels, n_clusters):
    """
    Calculates the components needed for the custom diagonal metric:
    total diagonal sum, total diagonal size, and individual diagonal block sizes.

    Args:
        activation_matrix (np.ndarray): Activation matrix (num_samples, num_units).
        sample_labels (np.ndarray): Sample labels (num_samples,).
        unit_labels (np.ndarray): Unit labels (num_units,).
        n_clusters (int): Number of clusters.

    Returns:
        tuple: (total_diagonal_sum, total_diagonal_size, block_sizes_array)
               block_sizes_array contains the size (count) of each diagonal block k.
    """
    total_diagonal_sum = 0.0
    total_diagonal_size = 0
    block_sizes = np.zeros(n_clusters, dtype=np.int64)  # Store block sizes

    for k in range(n_clusters):
        # Find indices for cluster k
        sample_indices_k = np.where(sample_labels == k)[0]
        unit_indices_k = np.where(unit_labels == k)[0]

        # Skip if cluster k is empty in either dimension
        if len(sample_indices_k) == 0 or len(unit_indices_k) == 0:
            block_sizes[k] = 0  # Explicitly set size to 0
            continue

        # Extract diagonal block
        block_kk = activation_matrix[np.ix_(sample_indices_k, unit_indices_k)]

        # Accumulate sum and size
        block_sum_k = np.sum(block_kk)
        block_size_k = block_kk.size

        total_diagonal_sum += block_sum_k
        total_diagonal_size += block_size_k
        block_sizes[k] = block_size_k

    return total_diagonal_sum, total_diagonal_size, block_sizes

def calculate_metric_from_components(total_diagonal_sum, total_diagonal_size, block_sizes):
    """
    Calculates the custom metric from pre-computed components.
    Metric = (Diagonal Sum / Diagonal Size) * (Harmonic Mean of Block Sizes)
    """
    if total_diagonal_size == 0:
        return 0.0

    # Part 1: Average diagonal activation
    avg_diag_activation = total_diagonal_sum / (total_diagonal_size + EPSILON)

    # Part 2: Harmonic Mean of non-zero block sizes
    non_zero_block_sizes = block_sizes[block_sizes > 0].astype(np.float64)  # Use float for division
    # Add EPSILON to denominator terms for numerical stability
    harmonic_mean = len(non_zero_block_sizes) / np.sum(1.0 / (non_zero_block_sizes + EPSILON))

    # Final Metric
    metric = avg_diag_activation * harmonic_mean
    return float(metric)  # Return as float

if __name__ == "__main__":
    args = get_args()

    # --- Set Seed for Reproducibility ---
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 1. Load Data ---
    print("Loading activation matrix...")
    start_load_time = time.time()
    activation_matrix_filename = f"{args.llm_name}_train_activation_matrix.pkl"
    activation_matrix_path = os.path.join(args.activation_matrix_dir, activation_matrix_filename)

    if not os.path.exists(activation_matrix_path):
        print(f"Error: Activation matrix file not found at {activation_matrix_path}")
        exit()
    try:
        activation_matrix_torch = pkl.load(open(activation_matrix_path, "rb"))
        # Convert to numpy array
        activation_matrix_np = np.array(activation_matrix_torch)
    except Exception as e:
        print(f"Error loading or converting activation matrix: {e}")
        exit()

    print(f"Loaded matrix shape: {activation_matrix_torch.shape}")
    print(f"Loading took {time.time() - start_load_time:.2f} seconds.")
    num_samples, num_units = activation_matrix_torch.shape

    # --- 2. Initial Label Generation (Sample-based Clustering) ---
    print("\n--- Initial Label Generation (Sample-based Clustering) ---")
    start_initial_label_time = time.time()

    # --- 2a. PCA for Samples ---
    print(f"Original sample data shape: {activation_matrix_torch.shape}")
    effective_pca_dim_s = min(args.pca_dim, num_samples - 1, num_units - 1 if num_units > 0 else 0)
    effective_pca_dim_s = max(1, effective_pca_dim_s)
    
    print(f"Performing PCA on samples (reducing unit dimension to {effective_pca_dim_s})...")
    start_pca_time = time.time()

    A_mean_s = activation_matrix_torch.mean(dim=0, keepdim=True)

    U_s, S_s, V_s = torch.pca_lowrank(activation_matrix_torch, q=effective_pca_dim_s, center=True)
    sample_scores = torch.matmul(activation_matrix_torch - A_mean_s, V_s)
    sample_scores_np = sample_scores.numpy()
    
    print(f"Sample PCA took {time.time() - start_pca_time:.2f} seconds.")
    print(f"Shape after PCA (sample scores): {sample_scores_np.shape}")

    # --- 2b. Define Clustering Methods for Initial Sample Clustering ---
    clustering_methods = {
        "kmeans": KMeans(n_clusters=args.n_clusters, init="random", random_state=args.seed, n_init='auto'),
        "kmeans++": KMeans(n_clusters=args.n_clusters, init="k-means++", random_state=args.seed, n_init='auto'),
        "minibatch_kmeans": MiniBatchKMeans(n_clusters=args.n_clusters, init="random", random_state=args.seed, n_init='auto', batch_size=4096),
        "minibatch_kmeans++": MiniBatchKMeans(n_clusters=args.n_clusters, init="k-means++", random_state=args.seed, n_init='auto', batch_size=4096),
        "agglomerative": AgglomerativeClustering(n_clusters=args.n_clusters),
        "balanced_kmeans": KMeansConstrained(
            n_clusters=args.n_clusters,
            size_min=num_samples // args.n_clusters,
            size_max=(num_samples + args.n_clusters - 1) // args.n_clusters, # Ceiling division,
            random_state=args.seed,
            n_init=10
        )
    }

    initial_model = clustering_methods[args.initial_clustering_method]

    # --- 2c. Cluster Samples ---
    print(f"Clustering samples using {args.initial_clustering_method} (shape after PCA: {sample_scores_np.shape})...")
    current_sample_labels_np = initial_model.fit_predict(sample_scores_np)

    current_sample_labels = torch.tensor(current_sample_labels_np, dtype=torch.int64)

    # --- 2d. Derive Unit Labels based on Sample Clusters ---
    print("Deriving initial unit labels based on sample clusters...")
    mean_activation_for_unit_in_sample_cluster = torch.zeros((args.n_clusters, num_units))
    for k_cluster in range(args.n_clusters):
        samples_in_cluster_k_mask = (current_sample_labels == k_cluster)
        if torch.any(samples_in_cluster_k_mask):
            mean_activation_for_unit_in_sample_cluster[k_cluster, :] = activation_matrix_torch[samples_in_cluster_k_mask, :].mean(dim=0)
        else:
            print(f"Warning: For initial method {args.initial_clustering_method}, sample cluster {k_cluster} is empty. Units will not be assigned to this cluster initially unless other clusters also have -inf mean activation.")
            mean_activation_for_unit_in_sample_cluster[k_cluster, :] = -torch.inf 

    current_unit_labels_torch = torch.argmax(mean_activation_for_unit_in_sample_cluster, dim=0)
    current_unit_labels = current_unit_labels_torch.numpy() # Convert to numpy for the rest of the script
    current_sample_labels = current_sample_labels.numpy() # Convert to numpy for the rest of the script

    print(f"Initial label generation took {time.time() - start_initial_label_time:.2f} seconds.")
    print(f"Initial sample labels shape: {current_sample_labels.shape}, unique labels: {len(np.unique(current_sample_labels))}")
    print(f"Initial unit labels shape: {current_unit_labels.shape}, unique labels: {len(np.unique(current_unit_labels))}")

    # --- 3. Iterative Clustering --- 
    print("\n--- Starting Iterative Clustering (Step 3) ---")
    total_iter_time_start = time.time()

    # Calculate initial metric components
    current_sum, current_total_size, current_block_sizes = calculate_metric_components(
        activation_matrix_np, current_sample_labels, current_unit_labels, args.n_clusters
    )
    current_metric = calculate_metric_from_components(current_sum, current_total_size, current_block_sizes)
    sample_counts = np.bincount(current_sample_labels, minlength=args.n_clusters)
    unit_counts = np.bincount(current_unit_labels, minlength=args.n_clusters)

    print(f"Initial Metric: {current_metric:.6f} (Sum: {current_sum:.2f}, Total Diag Size: {current_total_size}, Sum / Total Diag Size: {current_sum / (current_total_size + EPSILON):.6f})")
    print(f"  Initial Sample Cluster Sizes: {sample_counts.tolist()}")
    print(f"  Initial Unit Cluster Sizes:   {unit_counts.tolist()}")
    print(f"  Initial Block Sizes: {current_block_sizes.tolist()}")
    print()

    # Main iteration loop with tqdm
    for iter_num in range(args.max_iter):
        iter_start_time = time.time()
        labels_changed_in_iter = False
        units_updated = 0
        samples_updated = 0

        # --- 3a. Update Unit Labels (Fixed Sample Labels) ---
        # Pre-calculate sums and counts needed
        sum_activations_per_sample_cluster = np.zeros((args.n_clusters, num_units), dtype=np.float32)
        count_samples_per_cluster = np.zeros(args.n_clusters, dtype=np.int64)
        for k in range(args.n_clusters):
            samples_in_k = np.where(current_sample_labels == k)[0]
            count_k = len(samples_in_k)
            count_samples_per_cluster[k] = count_k
            if count_k > 0:
                sum_activations_per_sample_cluster[k, :] = np.sum(activation_matrix_np[samples_in_k], axis=0)

        # Iterate through units in random order
        unit_indices_perm = np.random.permutation(num_units)
        # Optional: Add tqdm for inner loop
        for j in tqdm(unit_indices_perm):
            original_label = current_unit_labels[j]
            best_label = original_label
            # Use current_metric as the baseline for comparison for this element
            best_metric_for_j = calculate_metric_from_components(current_sum, current_total_size, current_block_sizes)

            # Calculate contribution of unit j to the sum/size for its current cluster
            # NOTE: Size contribution *to the diagonal block k* is count_samples_per_cluster[k]
            sum_contribution_old = sum_activations_per_sample_cluster[original_label, j]
            size_contribution_old = count_samples_per_cluster[original_label]  # This many samples are in the rows for block (orig_label, orig_label)

            # Evaluate moving unit j to other clusters
            for k_prime in range(args.n_clusters):
                if k_prime == original_label:
                    continue

                # Calculate potential new state IF unit j moved to k_prime
                sum_contribution_new = sum_activations_per_sample_cluster[k_prime, j]
                size_contribution_new = count_samples_per_cluster[k_prime]  # This many samples are in the rows for block (k_prime, k_prime)

                potential_new_sum = current_sum - sum_contribution_old + sum_contribution_new
                potential_new_total_size = current_total_size - size_contribution_old + size_contribution_new

                # Calculate potential new block sizes
                potential_new_block_sizes = current_block_sizes.copy()  # Important: COPY the array
                if original_label >= 0 and original_label < args.n_clusters:  # Check bounds just in case
                    potential_new_block_sizes[original_label] = max(0, potential_new_block_sizes[original_label] - size_contribution_old)  # Decrease size of old block
                if k_prime >= 0 and k_prime < args.n_clusters:
                    potential_new_block_sizes[k_prime] += size_contribution_new  # Increase size of new block

                potential_new_metric = calculate_metric_from_components(potential_new_sum, potential_new_total_size, potential_new_block_sizes)

                if potential_new_metric > best_metric_for_j + EPSILON:  # Add tolerance for float comparison
                    best_metric_for_j = potential_new_metric
                    best_label = k_prime

            # If a better label was found, update the label and the *global* tracked state
            if best_label != original_label:
                current_unit_labels[j] = best_label
                labels_changed_in_iter = True
                units_updated += 1

                # Update the global tracked state
                sum_contribution_best_new = sum_activations_per_sample_cluster[best_label, j]
                size_contribution_best_new = count_samples_per_cluster[best_label]

                current_sum = current_sum - sum_contribution_old + sum_contribution_best_new
                current_total_size = current_total_size - size_contribution_old + size_contribution_best_new

                # Update block sizes state 
                if original_label >= 0 and original_label < args.n_clusters:
                    current_block_sizes[original_label] = max(0, current_block_sizes[original_label] - size_contribution_old)
                if best_label >= 0 and best_label < args.n_clusters:
                    current_block_sizes[best_label] += size_contribution_best_new

                # No need to update current_metric here, it's implicitly updated via components for the next element check

        # --- 3b. Update Sample Labels (Fixed Unit Labels) ---
        # Pre-calculate sums and counts needed
        sum_activations_per_unit_cluster = np.zeros((num_samples, args.n_clusters), dtype=np.float32)
        count_units_per_cluster = np.zeros(args.n_clusters, dtype=np.int64)
        for k in range(args.n_clusters):
            units_in_k = np.where(current_unit_labels == k)[0]
            count_k = len(units_in_k)
            count_units_per_cluster[k] = count_k
            if count_k > 0:
                sum_activations_per_unit_cluster[:, k] = np.sum(activation_matrix_np[:, units_in_k], axis=1)

        # Iterate through samples in random order
        sample_indices_perm = np.random.permutation(num_samples)
        for i in tqdm(sample_indices_perm):
            original_label = current_sample_labels[i]
            best_label = original_label
            # Use the metric state after unit updates as baseline
            best_metric_for_i = calculate_metric_from_components(current_sum, current_total_size, current_block_sizes)

            # Calculate contribution of sample i
            # NOTE: Size contribution *to the diagonal block k* is count_units_per_cluster[k]
            sum_contribution_old = sum_activations_per_unit_cluster[i, original_label]
            size_contribution_old = count_units_per_cluster[original_label]  # This many units are in the columns for block (orig_label, orig_label)

            # Evaluate moving sample i to other clusters
            for k_prime in range(args.n_clusters):
                if k_prime == original_label:
                    continue

                # Calculate potential new state IF sample i moved to k_prime
                sum_contribution_new = sum_activations_per_unit_cluster[i, k_prime]
                size_contribution_new = count_units_per_cluster[k_prime]

                potential_new_sum = current_sum - sum_contribution_old + sum_contribution_new
                potential_new_total_size = current_total_size - size_contribution_old + size_contribution_new

                potential_new_block_sizes = current_block_sizes.copy()
                if original_label >= 0 and original_label < args.n_clusters:
                    potential_new_block_sizes[original_label] = max(0, potential_new_block_sizes[original_label] - size_contribution_old)
                if k_prime >= 0 and k_prime < args.n_clusters:
                    potential_new_block_sizes[k_prime] += size_contribution_new

                potential_new_metric = calculate_metric_from_components(potential_new_sum, potential_new_total_size, potential_new_block_sizes)

                if potential_new_metric > best_metric_for_i + EPSILON:  # Add tolerance
                    best_metric_for_i = potential_new_metric
                    best_label = k_prime

            # If a better label was found, update the label and the global tracked state
            if best_label != original_label:
                current_sample_labels[i] = best_label
                labels_changed_in_iter = True
                samples_updated += 1

                # Update the global tracked state
                sum_contribution_best_new = sum_activations_per_unit_cluster[i, best_label]
                size_contribution_best_new = count_units_per_cluster[best_label]

                current_sum = current_sum - sum_contribution_old + sum_contribution_best_new
                current_total_size = current_total_size - size_contribution_old + size_contribution_best_new

                # Update block sizes state
                if original_label >= 0 and original_label < args.n_clusters:
                    current_block_sizes[original_label] = max(0, current_block_sizes[original_label] - size_contribution_old)
                if best_label >= 0 and best_label < args.n_clusters:
                    current_block_sizes[best_label] += size_contribution_best_new

        # --- End of Iteration ---
        iter_time = time.time() - iter_start_time
        # Calculate metric and sizes for reporting *after* both phases
        current_metric_end_iter = calculate_metric_from_components(current_sum, current_total_size, current_block_sizes)
        sample_counts = np.bincount(current_sample_labels, minlength=args.n_clusters)
        unit_counts = np.bincount(current_unit_labels, minlength=args.n_clusters)

        # Update tqdm description or print info
        tqdm.write(f"Iter {iter_num+1}/{args.max_iter} | Metric: {current_metric_end_iter:.6f} | Units changed: {units_updated} | Samples changed: {samples_updated} | Time: {iter_time:.2f}s")
        tqdm.write(f"  Sample Cluster Sizes: {sample_counts.tolist()}")
        tqdm.write(f"  Unit Cluster Sizes:   {unit_counts.tolist()}")
        tqdm.write(f"  Block Sizes:        {current_block_sizes.tolist()}")  # Show block sizes too
        tqdm.write("")

        # --- 3c. Check for Convergence ---
        if not labels_changed_in_iter:
            print(f"\nConvergence reached after {iter_num + 1} iterations.")
            break
    else:  # Loop finished without break
        print(f"\nMaximum iterations ({args.max_iter}) reached.")

    total_iter_time = time.time() - total_iter_time_start
    print("\n--- Iterative Clustering Finished ---")
    print(f"Total iterative time: {total_iter_time:.2f} seconds")
    # Recalculate final metric for verification
    final_sum, final_total_size, final_block_sizes = calculate_metric_components(activation_matrix_np, current_sample_labels, current_unit_labels, args.n_clusters)
    final_metric = calculate_metric_from_components(final_sum, final_total_size, final_block_sizes)
    print(f"Final Metric: {final_metric:.6f} (Sum: {final_sum:.2f}, Total Diag Size: {final_total_size}, Sum / Total Diag Size: {final_sum / (final_total_size + EPSILON):.6f})")    
    print(f"Final Block Sizes: {final_block_sizes.tolist()}")
    # Sanity check:
    # Re-calculate metric from tracked components at end of loop
    final_tracked_metric = calculate_metric_from_components(current_sum, current_total_size, current_block_sizes)
    if abs(final_metric - final_tracked_metric) > 1e-5:
        print(f"Warning: Final recalculated metric {final_metric:.6f} differs slightly from final tracked metric {final_tracked_metric:.6f}. Check logic if difference is large.")

    # --- 4. Save Final Clustering Results ---
    # Ensure directories exist
    os.makedirs(args.sample_labels_dir, exist_ok=True)
    os.makedirs(args.unit_labels_dir, exist_ok=True)

    # Define output filenames indicating the iterative method and metric type
    final_sample_label_filename = f"{args.llm_name}_iterative_{args.n_clusters}_samples.pkl"  # Changed filename
    final_sample_label_path = os.path.join(args.sample_labels_dir, final_sample_label_filename)

    final_unit_label_filename = f"{args.llm_name}_iterative_{args.n_clusters}_units.pkl"  # Changed filename
    final_unit_label_path = os.path.join(args.unit_labels_dir, final_unit_label_filename)

    print(f"Saving final sample labels ({current_sample_labels.shape}) to {final_sample_label_path}")
    current_sample_labels_torch = torch.tensor(current_sample_labels, dtype=torch.int64)
    pkl.dump(current_sample_labels_torch, open(final_sample_label_path, "wb"))

    print(f"Saving final unit labels ({current_unit_labels.shape}) to {final_unit_label_path}")
    current_unit_labels_torch = torch.tensor(current_unit_labels, dtype=torch.int64)
    pkl.dump(current_unit_labels_torch, open(final_unit_label_path, "wb"))

    print("\nClustering and saving process completed.")
