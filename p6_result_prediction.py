import argparse
import os
import pickle as pkl
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score # Added f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Added import
from tabulate import tabulate
from tqdm import tqdm  # For progress bars

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
    parser.add_argument("--unit_labels_dir", type=str, default="unit_labels") # Directory containing unit clustering result files (*_units.pkl)
    parser.add_argument("--n_clusters", type=int, default=20) # Number of unit clusters used (must match files)
    parser.add_argument("--train_samples", type=int, default=7000) # Total number of samples in the training activation matrix
    parser.add_argument("--train_samples_per_class", type=int, default=1000) # Number of consecutive samples per class in training data
    parser.add_argument("--test_samples", type=int, default=1400) # Total number of samples in the test activation matrix
    parser.add_argument("--test_samples_per_class", type=int, default=200) # Number of consecutive samples per class in test data
    parser.add_argument("--classifier", type=str, default="svc", choices=["svc", "lr"]) # Classifier to use (more can be added)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1) # Random seed for reproducibility

    args = parser.parse_args()
    # Calculate number of sample classes for train and test sets
    if args.train_samples % args.train_samples_per_class != 0:
        raise ValueError("train_samples must be divisible by train_samples_per_class")
    if args.test_samples % args.test_samples_per_class != 0:
        raise ValueError("test_samples must be divisible by test_samples_per_class")

    args.train_num_sample_classes = args.train_samples // args.train_samples_per_class
    args.test_num_sample_classes = args.test_samples // args.test_samples_per_class

    print("Derived number of train sample classes:", args.train_num_sample_classes)
    print("Derived number of test sample classes:", args.test_num_sample_classes)

    print_dict(vars(args))
    return args

def generate_ground_truth_labels(num_samples, samples_per_class):
    """Generates ground truth labels based on sample index."""
    print(f"Generating ground truth labels for {num_samples} samples, {samples_per_class} per class...")
    labels = np.arange(num_samples) // samples_per_class
    print(f"Generated labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
    return labels

def calculate_sample_features_from_unit_clusters(activation_matrix, unit_labels, n_clusters):
    """
    Transforms activation matrix based on unit clusters.

    Args:
        activation_matrix (torch.Tensor or np.ndarray): Shape (num_samples, num_units).
        unit_labels (np.ndarray): Shape (num_units,). Cluster assignment for each unit.
        n_clusters (int): The number of unit clusters.

    Returns:
        np.ndarray: Transformed feature matrix of shape (num_samples, n_clusters).
                    Value (i, k) is the mean activation of units in cluster k for sample i.
                    Only the maximum value for each sample is kept, others are set to zero.
    """
    num_samples, num_units = activation_matrix.shape
    print(f"Calculating features for {num_samples} samples based on {n_clusters} unit clusters...")
    start_time = time.time()

    # Ensure activation_matrix is numpy for easier indexing if needed, but torch is faster
    is_torch = isinstance(activation_matrix, torch.Tensor)
    if not is_torch:
        activation_matrix = torch.from_numpy(activation_matrix).float() # Use float for mean

    # Ensure unit_labels is numpy
    if isinstance(unit_labels, torch.Tensor):
        unit_labels = unit_labels.cpu().numpy()

    # Initialize the feature matrix
    sample_features = np.zeros((num_samples, n_clusters), dtype=np.float32)

    # Calculate features efficiently using PyTorch tensor operations
    activation_matrix_gpu = activation_matrix.cuda() if torch.cuda.is_available() else activation_matrix
    unit_labels_tensor = torch.from_numpy(unit_labels).long() # Needs to be long for indexing
    unit_labels_gpu = unit_labels_tensor.cuda() if torch.cuda.is_available() else unit_labels_tensor

    for k in tqdm(range(n_clusters), desc="Calculating features per cluster"):
        # Find indices of units belonging to cluster k
        unit_indices_k = torch.where(unit_labels_gpu == k)[0]

        if len(unit_indices_k) == 0:
            # If a cluster is empty, the mean activation is 0 (or NaN, handle appropriately)
            # Setting to 0 is reasonable here.
            sample_features[:, k] = 0.0
            # print(f"Warning: Unit cluster {k} is empty.") # Optional warning
            continue

        # Select columns (units) corresponding to cluster k
        # activation_matrix_gpu is (num_samples, num_units)
        # unit_indices_k is (num_units_in_cluster_k,)
        # activations_k should be (num_samples, num_units_in_cluster_k)
        activations_k = activation_matrix_gpu[:, unit_indices_k]

        # Calculate the mean across the unit dimension (dim=1)
        mean_activations_k = torch.mean(activations_k, dim=1)

        # Store the result (move back to CPU if calculated on GPU)
        sample_features[:, k] = mean_activations_k.cpu().numpy()

    print(f"Feature calculation took {time.time() - start_time:.2f} seconds.")
    print(f"Resulting feature matrix shape: {sample_features.shape}")
    return sample_features


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.cuda.set_device(args.gpu)

    # --- 1. Load Train and Test Activation Matrices ---
    print("\nLoading activation matrices...")
    start_load = time.time()
    
    # Load training activation matrix
    train_matrix_filename = f"{args.llm_name}_train_activation_matrix.pkl"
    train_matrix_path = os.path.join(args.activation_matrix_dir, train_matrix_filename)

    if not os.path.exists(train_matrix_path):
        print(f"Error: Training activation matrix file not found at {train_matrix_path}")
        exit()

    try:
        # Load as torch tensor directly if possible for potential GPU usage
        train_activation_matrix = pkl.load(open(train_matrix_path, "rb"))
        print(f"Loaded training activation matrix shape: {train_activation_matrix.shape}, Type: {type(train_activation_matrix)}")
    except Exception as e:
        print(f"Error loading training activation matrix: {e}")
        exit()
    
    # Load test activation matrix
    test_matrix_filename = f"{args.llm_name}_test_activation_matrix.pkl"
    test_matrix_path = os.path.join(args.activation_matrix_dir, test_matrix_filename)

    if not os.path.exists(test_matrix_path):
        print(f"Error: Test activation matrix file not found at {test_matrix_path}")
        exit()
    
    try:
        # Load as torch tensor directly if possible for potential GPU usage
        test_activation_matrix = pkl.load(open(test_matrix_path, "rb"))
        if not isinstance(test_activation_matrix, torch.Tensor):
             test_activation_matrix = torch.from_numpy(np.array(test_activation_matrix)).float()
        else:
             test_activation_matrix = test_activation_matrix.float() # Ensure float type
        print(f"Loaded test activation matrix shape: {test_activation_matrix.shape}, Type: {type(test_activation_matrix)}")
    except Exception as e:
        print(f"Error loading test activation matrix: {e}")
        exit()
        
    print(f"Loading took {time.time() - start_load:.2f} seconds.")

    # --- 2. Generate Ground Truth Sample Labels ---
    y_train = generate_ground_truth_labels(args.train_samples, args.train_samples_per_class)
    y_test = generate_ground_truth_labels(args.test_samples, args.test_samples_per_class)

    # --- 3. Find Unit Label Files ---
    unit_label_files_info = [] # Stores tuples of (method_name, file_path)
    print(f"\nSearching for unit label files in {args.unit_labels_dir} matching the pattern...")

    if not os.path.exists(args.unit_labels_dir):
        print(f"Error: Unit labels directory not found at {args.unit_labels_dir}")
        exit()

    llm_name_prefix = args.llm_name + "_"
    # Filename pattern: {llm_name}_{method_name}_{n_clusters}_units.pkl
    # Example: Qwen2.5-1.5B-Instruct_iterative_5_units.pkl
    n_clusters_units_suffix = f"_{args.n_clusters}_units.pkl"

    for filename in os.listdir(args.unit_labels_dir):
        if filename.startswith(llm_name_prefix) and filename.endswith(n_clusters_units_suffix):
            # Extract method_name from the filename
            # e.g., "Qwen2.5-1.5B-Instruct_iterative_5_units.pkl"
            # prefix_len = len("Qwen2.5-1.5B-Instruct_")
            # suffix_len = len("_5_units.pkl")
            # method_name is "iterative"
            method_name = filename[len(llm_name_prefix) : -len(n_clusters_units_suffix)]
            
            if method_name:
                file_path = os.path.join(args.unit_labels_dir, filename)
                unit_label_files_info.append((method_name, file_path))
                print(f" - Found and will process: {filename} (Method: {method_name})")
            elif not method_name: # File matched pattern, but method_name doesn't meet criteria
                print(f" - Warning: File {filename} matches prefix/suffix but method name part is empty. Skipping.")

    if not unit_label_files_info:
        print(f"Error: No unit label files found in '{args.unit_labels_dir}' matching the pattern: '{llm_name_prefix}*_{args.n_clusters}_units.pkl'")
        exit()

    print(f"\nWill evaluate {len(unit_label_files_info)} unit label files:")
    for method_name, f_path in unit_label_files_info:
        print(f" - {os.path.basename(f_path)} (Method: {method_name})")

    # --- 4. Evaluate Each Unit Clustering Method ---
    results = []
    # The 'unit_label_files_info' list now contains tuples of (method_name, file_path)
    for method_name, unit_label_file_path in unit_label_files_info:
        # 'method_name' is now directly available from the tuple
        print(f"\n--- Evaluating Method: {method_name} ---")

        # --- 4a. Load Unit Labels ---
        try:
            print(f"Loading unit labels from: {unit_label_file_path}")
            unit_labels = pkl.load(open(unit_label_file_path, "rb"))
            unit_labels = unit_labels.cpu().numpy()

            if unit_labels.shape[0] != train_activation_matrix.shape[1]:
                 print(f"Error: Number of unit labels ({unit_labels.shape[0]}) in {os.path.basename(unit_label_file_path)} does not match number of units in activation matrix ({train_activation_matrix.shape[1]}). Skipping.")
                 continue
            print(f"Loaded unit labels shape: {unit_labels.shape}")
        except Exception as e:
            print(f"Error loading unit labels from {os.path.basename(unit_label_file_path)}: {e}. Skipping.")
            continue

        # --- 4b. Calculate Features for Train and Test Sets ---
        X_train = calculate_sample_features_from_unit_clusters(train_activation_matrix, unit_labels, args.n_clusters)
        X_test = calculate_sample_features_from_unit_clusters(test_activation_matrix, unit_labels, args.n_clusters)

        print(f"Train features shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test features shape:  X={X_test.shape}, y={y_test.shape}")

        # --- 4c. Scale Features ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) # Use transform, not fit_transform on test data

        # --- 4d. Train Classifier ---
        print(f"Training {args.classifier}...")
        start_train = time.time()
        if args.classifier == "lr":
            # Updated to use OneVsRestClassifier instead of deprecated multi_class parameter
            base_model = LogisticRegression(random_state=args.seed, solver='liblinear', max_iter=1000)
            model = OneVsOneClassifier(base_model)
        elif args.classifier == "svc":  # Added SVC option
            model = SVC(random_state=args.seed, max_iter=10000)  # Increased max_iter for SVC
        else:
             print(f"Error: Classifier '{args.classifier}' not implemented.")
             continue # Skip to next method

        try:
            model.fit(X_train, y_train)
            print(f"Training took {time.time() - start_train:.2f} seconds.")
        except Exception as e:
            print(f"Error during model training: {e}. Skipping method {method_name}.")
            continue

        # --- 4e. Predict and Evaluate ---
        print("Predicting on test set...")
        start_predict = time.time()
        try:
            y_pred = model.predict(X_test)
            print(f"Prediction took {time.time() - start_predict:.2f} seconds.")

            print("\nEvaluation Results:")
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro') # Calculate macro-F1
            report = classification_report(y_test, y_pred, digits=4) # Increased digits for precision
            conf_matrix = confusion_matrix(y_test, y_pred)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro-F1 Score: {macro_f1:.4f}") # Print macro-F1
            print("Classification Report:")
            print(report)
            print("Confusion Matrix:")
            print(conf_matrix)

            # Store results
            results.append({
                "Method": method_name,
                "Accuracy": accuracy,
                "Macro_F1": macro_f1, # Store macro-F1
                "Classification_report": report,
                "Confusion_matrix": conf_matrix
            })
        except Exception as e:
            print(f"Error during prediction or evaluation: {e}. Skipping method {method_name}.")
            continue

    # --- 5. Summarize Results ---
    print("\n--- Summary of Evaluation Results ---")
    
    # Sort results by accuracy in descending order
    # Each item in 'results' is a dictionary.
    sorted_results_list = sorted(results, key=lambda item: item['Method'], reverse=True)

    # Prepare data for tabulate: list of lists/tuples
    table_data = [(item['Method'], item['Accuracy'], item['Macro_F1']) for item in sorted_results_list]
    headers = ["Method", "Accuracy", "Macro-F1"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))


    os.makedirs("result_prediction", exist_ok=True)
    # Save results to a text file
    with open(f"result_prediction/{args.llm_name}_{args.classifier}_{args.n_clusters}_result_prediction.txt", "w") as f:
        f.write(f"llm_name: {args.llm_name}, classifier: {args.classifier}, n_clusters: {args.n_clusters}\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

    print("\nEvaluation complete.")
