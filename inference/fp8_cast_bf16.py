import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file


def weight_dequant(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor.
        s (torch.Tensor): The scale tensor.

    Returns:
        torch.Tensor: The dequantized weight tensor.
    """
    return x.to(torch.bfloat16) * s.to(torch.bfloat16)


def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights and their scale factors from the specified directory,
    converts them to BF16 using CPU operations, and saves the converted weights to another
    directory. It also updates the model index file to remove references to scale factors.

    Args:
        fp8_path (str): The path to the directory containing the FP8 weights and model index file.
        bf16_path (str): The path to the directory where the converted BF16 weights will be saved.
    """
    os.makedirs(bf16_path, exist_ok=True)
    
    # Load the model index file
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    # Process each safetensor file
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files, desc="Processing weights"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                # Skip scale inverse tensors
                continue
            elif weight.dtype == torch.uint8:  # Assuming FP8 is stored as uint8
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    scale_inv = get_tensor(scale_inv_name)
                    dequantized_weight = weight_dequant(weight, scale_inv)
                    new_state_dict[weight_name] = dequantized_weight
                    fp8_weight_names.append(weight_name)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
        
        # Save the new state dict
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
    
    # Update the model index file
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    
