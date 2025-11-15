import argparse
import os
import pickle as pkl
import warnings

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

EPSILON = 1e-8

# Ignore warning messages
warnings.filterwarnings("ignore")

def print_dict(d: dict):
    """Prints a dictionary in table format"""
    print(tabulate([(k, v) for k, v in d.items()], tablefmt="grid"))

def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="Qwen2.5-3B-Instruct") # LLM name
    parser.add_argument("--dataset_dir", type=str, default="dataset/") # Directory containing train.csv and test.csv"
    parser.add_argument("--activation_matrix_dir", type=str, default="./activation_matrix/") # Directory to save activation matrix files
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"]) # Select which dataset to process ('train' or 'test')
    parser.add_argument("--gpu", type=int, default=0) # Specify the GPU index to use
    args = parser.parse_args()

    print_dict(vars(args))
    return args

class ActivationCollector:
    """Activation collector, used to get activations from intermediate layers of the model"""

    def __init__(self):
        self.activation = []  # List to store activation values
        self.hook_handles = []  # List to store hook handles

    def _activation_hook(self, module, inputs, output):
        """
        Forward propagation hook function.
        This example gets the *input* activation of the mlp.down_proj layer.
        Dimension of input inputs[0]: (batch_size, seq_len, hidden_size)
        """
        # inputs is a tuple, the first element is the tensor passed to down_proj
        input_tensor = inputs[0]

        # --- Calculate activation value ---
        # Here we calculate the average absolute activation value along the sequence length dimension for the first sample (batch_size=1)
        assert input_tensor.shape[0] == 1, "Input batch size != 1, activation calculation may not be as expected, using only the first sample."
        # Take the input of the first batch
        embedding = input_tensor[0] # (seq_len, hidden_size)
        # Calculate the mean along the sequence dimension
        activation = embedding.abs().mean(dim=0) # (hidden_size,)

        self.activation.append(activation.cpu().detach())

    def register_hooks(self, model):
        """Registers hooks for the mlp.down_proj submodules of all Transformer layers in the model"""
        for layer in model.model.layers:
            # Ensure the layer has an mlp attribute, and mlp has a down_proj attribute
            target_module = layer.mlp.down_proj
            # Register forward hook
            handle = target_module.register_forward_hook(self._activation_hook)
            self.hook_handles.append(handle)

    def clear_hooks(self):
        """Removes all registered hooks, freeing resources"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def clear_activation(self):
        """Clears the currently stored activation values, preparing for the next sample"""
        self.activation = []

if __name__ == "__main__":
    args = get_args()
    torch.cuda.set_device(args.gpu) # Set the GPU device

    # --- Initialize model and Tokenizer ---
    llm_path = os.path.join(os.getenv("LLM_PATH_BASE"), args.llm_name)
    print(f"Loading model: {args.llm_name} from {llm_path}")
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        # torch_dtype=torch.float16 # Consider using half-precision to save memory and speed up, uncomment as needed
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    model.eval() # Set to evaluation mode
    print("Model loaded successfully")


    # --- Load and preprocess dataset ---
    csv_filename = f"{args.mode}.csv" # Determine filename based on mode: train.csv or test.csv
    csv_path = os.path.join(args.dataset_dir, csv_filename)
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    dataset = Dataset.from_pandas(df)  # Convert to Hugging Face Dataset
    print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")

    # --- Initialize activation collector and register hooks ---
    collector = ActivationCollector()
    collector.register_hooks(model)

    # --- Iterate through the dataset and extract activations ---
    all_activations = []  # Used to store activations for all samples

    # Use tqdm to display progress bar
    for i in tqdm(range(len(dataset)), desc=f"Processing {args.mode} dataset samples"):
        item = dataset[i]
        instruction = item["instruction"] # Get instruction text from the dataset

        # Prepare model input format (using Qwen's chat template)
        # Note: The system prompt here can be set as needed
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]

        # Apply chat template and tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to("cuda")

        # Clear activations from the previous round
        collector.clear_activation()

        # Execute forward propagation to trigger hooks
        with torch.no_grad(): # In evaluation mode, do not calculate gradients to save resources
            _ = model(**model_inputs)

        # Collect activations for the current sample
        # collector.activation is a list, each element is an activation tensor from a layer (hidden_size,)
        # We need to stack them
        sample_activations = torch.stack(collector.activation)  # Shape becomes (num_layers, hidden_size)
        all_activations.append(sample_activations.cpu()) # Move to CPU and store

    # Ensure hooks are cleared
    collector.clear_hooks()

    # Stack all sample activation tensors in the list into one large tensor
    # all_activations is a list, each element is (num_layers, hidden_size)
    activations = torch.stack(all_activations)  # Shape becomes (num_samples, num_layers, hidden_size)
    print(f"Final activation tensor A shape: {activations.shape}") # (num_samples, num_layers, hidden_size)
    # --- Adjust as needed ---
    # If the original example's (total_units, num_samples) shape is needed, reshape and transpose
    num_samples, num_layers, hidden_size = activations.shape
    activation_matrix = activations.reshape(num_samples, num_layers * hidden_size) # (num_samples, total_units)
    print(f"Adjusted activation matrix (activation_matrix) shape: {activation_matrix.shape}")

    # Ensure activation_matrix_dir exists
    os.makedirs(args.activation_matrix_dir, exist_ok=True)

    if args.mode == "train":
        # Z-Score normalization for training: calculate, apply, and save mean/std
        activation_matrix_mean = activation_matrix.mean(dim=0, keepdim=True)
        activation_matrix_std = activation_matrix.std(dim=0, keepdim=True)

        # Save mean and std for later use in test mode
        mean_filename = f"{args.llm_name}_train_mean.pkl"
        mean_path = os.path.join(args.activation_matrix_dir, mean_filename)
        print(f"Saving training mean to: {mean_path}")
        pkl.dump(activation_matrix_mean, open(mean_path, "wb"))

        std_filename = f"{args.llm_name}_train_std.pkl"
        std_path = os.path.join(args.activation_matrix_dir, std_filename)
        print(f"Saving training std to: {std_path}")
        pkl.dump(activation_matrix_std, open(std_path, "wb"))
        print("Training mean and std saved successfully.")

    elif args.mode == "test":
        # Load mean and std from training phase for test mode normalization
        mean_filename = f"{args.llm_name}_train_mean.pkl"
        mean_path = os.path.join(args.activation_matrix_dir, mean_filename)
        print(f"Loading training mean from: {mean_path}")
        activation_matrix_mean = pkl.load(open(mean_path, "rb"))

        std_filename = f"{args.llm_name}_train_std.pkl"
        std_path = os.path.join(args.activation_matrix_dir, std_filename)
        print(f"Loading training std from: {std_path}")
        activation_matrix_std = pkl.load(open(std_path, "rb"))
        print("Training mean and std loaded successfully.")

    # Avoid NaN in std
    activation_matrix_std[activation_matrix_std < EPSILON] = 1.0
    # Apply normalization
    activation_matrix = (activation_matrix - activation_matrix_mean) / activation_matrix_std
    print("Activation matrix normalized successfully.")

    # --- Clear hooks ---
    collector.clear_hooks()

    # --- Save results ---
    activation_matrix_filename = f"{args.llm_name}_{args.mode}_activation_matrix.pkl"
    activation_matrix_path = os.path.join(args.activation_matrix_dir, activation_matrix_filename)
    print(f"Saving activation matrix to: {activation_matrix_path}")
    pkl.dump(activation_matrix, open(activation_matrix_path, "wb"))
    print("Activation matrix saved successfully")
