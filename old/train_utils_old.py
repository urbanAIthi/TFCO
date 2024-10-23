import torch


from typing import Dict


def merge_sequence(data: dict, key: str, sequence_length: int, tens=True) -> Dict[str, torch.Tensor]:
    reverse_indices = list(reversed(range(1, sequence_length + 1)))

    def generate_key(i):
        return f'{key}-{i}'

    keys_to_merge = list(map(generate_key, reverse_indices)) + [key]

    merged_dict = {d: {} for d in data}  # Pre-allocate memory for merged_dict

    for d in data:
        tensors_to_merge = [data[d][k] for k in keys_to_merge]
        if tens:
            merged_tensor = torch.stack(tensors_to_merge, dim=0)
            merged_tensor = merged_tensor.unsqueeze(1)
            merged_dict[d] = {f'merged_{key}': merged_tensor}
        else:
            merged_dict[d] = {f'merged_{key}': tensors_to_merge}

    return merged_dict