import argparse
import os
import pickle as pkl
import time

import numpy as np
from tabulate import tabulate

# Define a small epsilon to prevent division by zero
EPSILON = 1e-8

def print_dict(d: dict):
    """Prints a dictionary in a table format"""
    print(tabulate([(k, v) for k, v in d.items()], tablefmt="grid"))

def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="Qwen2.5-3B-Instruct") # LLM name
    parser.add_argument("--activation_matrix_dir", type=str, default="./activation_matrix/") # Directory containing activation matrix file
    parser.add_argument("--sample_labels_dir", type=str, default="sample_labels/") # Directory containing sample label files (*_samples.pkl)
    parser.add_argument("--unit_labels_dir", type=str, default="unit_labels/") # Directory containing unit label files (*_units.pkl)
    parser.add_argument("--n_clusters", type=int, default=20) # Number of clusters (K)
    args = parser.parse_args()
    print_dict(vars(args))
    return args

def calculate_xi_f(total_act_sum_F, total_size_F):
    """
    Calculates xi(F)
    """
    if total_size_F == 0:
        return 0.0
    return total_act_sum_F / (total_size_F + EPSILON)

def calculate_h_f(block_sizes_F, n_clusters):
    """
    Calculates H(F) = K / (sum_{k=1 to K} (1 / Size_k)).
    block_sizes_F is an array of Size_k for k=1 to K.
    """
    h_f = n_clusters / np.sum(1.0 / (block_sizes_F + EPSILON))

    return h_f

def calculate_l_f(act_f_val, h_f_val):
    """
    Calculates L(F) = Act(F) * H(F).
    """
    return act_f_val * h_f_val

if __name__ == "__main__":
    args = get_args()

    # --- 1. Load Activation Matrix ---
    print("\nLoading activation matrix...")
    start_load = time.time()
    activation_matrix_filename = f"{args.llm_name}_train_activation_matrix.pkl"
    activation_matrix_path = os.path.join(args.activation_matrix_dir, activation_matrix_filename)

    if not os.path.exists(activation_matrix_path):
        print(f"Error: Activation matrix file not found at {activation_matrix_path}")
        exit()
    try:
        activation_matrix = pkl.load(open(activation_matrix_path, "rb"))
        if not isinstance(activation_matrix, np.ndarray):
            if hasattr(activation_matrix, 'numpy'): # For torch tensors
                activation_matrix = activation_matrix.cpu().numpy()
            else:
                activation_matrix = np.array(activation_matrix)
        print(f"Loaded activation matrix shape: {activation_matrix.shape}, Type: {type(activation_matrix)}")
    except Exception as e:
        print(f"Error loading or converting activation matrix: {e}")
        exit()

    num_samples, num_units = activation_matrix.shape
    total_matrix_elements = num_samples * num_units
    print(f"Loading took {time.time() - start_load:.2f} seconds.")

    if total_matrix_elements == 0:
        print("Error: Activation matrix is empty (0 samples or 0 units). Cannot proceed.")
        exit()

    # --- 2. Find Label File Pairs ---
    print("\nSearching for label files...")
    label_files_map = {} # method_name_key -> {'sample_path': ..., 'unit_path': ...}
    
    llm_name_prefix = args.llm_name + "_"
    n_clusters_str_suffix_samples = f"_{args.n_clusters}_samples.pkl"
    n_clusters_str_suffix_units = f"_{args.n_clusters}_units.pkl"

    if not os.path.exists(args.sample_labels_dir):
        print(f"Error: Sample labels directory not found at {args.sample_labels_dir}")
        exit()
    if not os.path.exists(args.unit_labels_dir):
        print(f"Error: Unit labels directory not found at {args.unit_labels_dir}")
        exit()

    for sample_filename in os.listdir(args.sample_labels_dir):
        if sample_filename.startswith(llm_name_prefix) and sample_filename.endswith(n_clusters_str_suffix_samples):
            method_name_key = sample_filename[len(llm_name_prefix) : -len(n_clusters_str_suffix_samples)]
            unit_filename = f"{llm_name_prefix}{method_name_key}{n_clusters_str_suffix_units}"
            unit_filepath = os.path.join(args.unit_labels_dir, unit_filename)
            sample_filepath = os.path.join(args.sample_labels_dir, sample_filename)

            if os.path.exists(unit_filepath):
                if method_name_key not in label_files_map:
                    label_files_map[method_name_key] = {'sample_path': sample_filepath, 'unit_path': unit_filepath}
                    print(f"  Found pair for method: {method_name_key}")
            else:
                print(f"  Warning: Sample file {sample_filename} found, but corresponding unit file {unit_filename} not found in {args.unit_labels_dir}.")

    if not label_files_map:
        print(f"Error: No matching sample/unit label file pairs found for LLM '{args.llm_name}' and {args.n_clusters} clusters.")
        print(f"Searched in '{args.sample_labels_dir}' and '{args.unit_labels_dir}'.")
        exit()
    
    print(f"\nFound {len(label_files_map)} methods to evaluate.")

    # --- 3. Evaluate Each Method ---
    results = []
    
    for method_name, paths in label_files_map.items():
        print(f"\n--- Evaluating Method: {method_name} ---")
        # Load sample labels
        with open(paths['sample_path'], "rb") as f:
            sample_labels = pkl.load(f)
        sample_labels = sample_labels.numpy()

        # Load unit labels
        with open(paths['unit_path'], "rb") as f:
            unit_labels = pkl.load(f)
        unit_labels = unit_labels.numpy()

        if sample_labels.shape[0] != num_samples:
            print(f"  Error: Sample labels shape mismatch for {method_name}. Expected {num_samples}, got {sample_labels.shape[0]}. Skipping.")
            continue
        if unit_labels.shape[0] != num_units:
            print(f"  Error: Unit labels shape mismatch for {method_name}. Expected {num_units}, got {unit_labels.shape[0]}. Skipping.")
            continue

        print(f"  Loaded sample labels shape: {sample_labels.shape}, unit labels shape: {unit_labels.shape}")

        total_act_sum_F = 0.0
        total_size_F = 0
        block_sizes_F = np.zeros(args.n_clusters, dtype=np.int64)

        for k_cluster in range(args.n_clusters):
            sample_indices_k = np.where(sample_labels == k_cluster)[0]
            unit_indices_k = np.where(unit_labels == k_cluster)[0]

            num_samples_k = len(sample_indices_k)
            num_units_k = len(unit_indices_k)

            current_size_k = num_samples_k * num_units_k
            block_sizes_F[k_cluster] = current_size_k
            total_size_F += current_size_k

            if num_samples_k > 0 and num_units_k > 0:
                # Extract the block A_kk = A[S_k, U_k]
                block_kk = activation_matrix[np.ix_(sample_indices_k, unit_indices_k)]
                current_act_sum_k = np.sum(block_kk)
                total_act_sum_F += current_act_sum_k

        act_f_val = calculate_xi_f(total_act_sum_F, total_size_F)
        h_f_val = calculate_h_f(block_sizes_F, args.n_clusters)
        l_f_val = calculate_l_f(act_f_val, h_f_val)

        results.append({
            "Method": method_name,
            "L(F)": l_f_val,
            "xi(F)": act_f_val,
            "H(F)": h_f_val,
            "SumAct_diag": total_act_sum_F,
            "SumSize_diag": total_size_F,
        })
        print(f"  Metrics - xi(F): {act_f_val:.6f}, H(F): {h_f_val:.6f}, L(F): {l_f_val:.6f}")
        print(f"  Detailed - SumAct_diag: {total_act_sum_F:.2f}, SumSize_diag: {total_size_F}, BlockSizes: {block_sizes_F.tolist()}")

    # --- 4. Summarize Results ---
    headers = ["Method", "L(F)", "xi(F)", "H(F)"]

    # Corrected sorting:
    sorted_results_list = sorted(results, key=lambda item: item['Method'], reverse=True)

    # Convert list of dictionaries to list of lists for tabulate
    table_data = [[item[header_key] for header_key in headers] for item in sorted_results_list]

    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

    os.makedirs("result_objective", exist_ok=True)
    # Save results to a text file
    with open(f"result_objective/{args.llm_name}_{args.n_clusters}_result_objective.txt", "w") as f:
        f.write(f"llm_name: {args.llm_name}, n_clusters: {args.n_clusters}\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

    print("\nEvaluation complete.")
