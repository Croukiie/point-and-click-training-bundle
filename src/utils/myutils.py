import yaml
import pickle
import time, os
try:
    import torch
except ModuleNotFoundError:
    torch = None
import numpy as np
from box import Box
from datetime import datetime
import json
import hashlib
from collections.abc import Mapping, Sequence, Set

HAS_BOX = True

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config


def save_dict_to_yaml(data, filename):
    if type(data) is Box:
        data = data.to_dict()
    
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    yaml_data = yaml.dump(data, allow_unicode=True, sort_keys=False)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(yaml_data)


def pickle_save(filename, data, try_multiple_save=100, verbose=False):
    if not filename.endswith('.pkl'): filename += '.pkl'
    if try_multiple_save <= 0: try_multiple_save = 1

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for _ in range(try_multiple_save):
        try:
            with open(filename, "wb") as fp:
                pickle.dump(data, fp)
            return
        except Exception as e:
            if verbose: print(f"Save attempt failed with error: {e}. Retrying...")
            time.sleep(0.5)
            continue
    raise ValueError("Save failed after multiple attempts. Check file directory and permissions.")


def pickle_load(file, verbose=False):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    if verbose:
        print(f"Pickle file {file} loaded; datatype {str(type(file))}")
    return data


def npz_save(filename, **data_kwargs):
    if not filename.endswith('.npz'): filename += '.npz'
    np.savez_compressed(filename, **data_kwargs)


def npz_load(filename):
    return np.load(filename, allow_pickle=True)


def get_current_time_digit():
    now = datetime.now()
    return now.strftime("%m%d%H%M")


def get_compact_timestamp_str(omit_year=False):
    now = datetime.now()
    year = now.year % 100
    day_of_year = now.timetuple().tm_yday
    total_seconds = now.hour * 3600 + now.minute * 60 + now.second
    if omit_year:
        session_name = f"{day_of_year:03d}{int_to_base26_char(total_seconds)}"
    else:
        session_name = f"{year:02d}{day_of_year:03d}{int_to_base26_char(total_seconds)}"
    return session_name 


def int_to_base26_char(n):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    for i in range(3, -1, -1):
        # Calculate value for the current place
        value = (n // (26 ** i)) % 26
        result += chars[value]
    return result


# Configuration file hashing
def _to_plain(obj):
    """
    Convert wrapper types (e.g., Box) into plain Python types.
    Add more conversions here if your project uses other custom config wrappers.
    """
    if HAS_BOX and isinstance(obj, Box):
        return obj.to_dict()
    return obj


def _canonicalize(obj, float_rounding=6):
    """
    Recursively convert any config object (dict, list, set, Box...)
    into a canonical JSON-serializable representation with deterministic ordering.
    This ensures 'logically identical' configs produce the same hash.
    """
    obj = _to_plain(obj)

    # Mapping (dict-like): sort keys so ordering differences do not matter
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}

    # List or tuple: preserve natural order but canonicalize each element
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]

    # Set / frozenset: convert to a sorted list because sets are unordered
    if isinstance(obj, Set) and not isinstance(obj, (str, bytes)):
        items = [_canonicalize(v) for v in obj]
        # Sorting by repr ensures consistent ordering even across mixed types
        return sorted(items, key=lambda x: repr(x))

    # Numpy scalar conversion (optional but useful)
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            obj = obj.item()
    except ImportError:
        pass

    # Float normalization to reduce insignificant differences
    if isinstance(obj, float):
        return round(obj, float_rounding)

    # Everything else: return as-is
    return obj


def config_hash(config, algo="sha256", length=16):
    """
    Convert an arbitrary config (dict/Box/nested structure) into a stable hash string.

    Steps:
    1. Canonicalize structure (remove ordering/nondeterminism)
    2. Serialize using JSON with fixed formatting
    3. Hash the resulting JSON string
    """
    canon = _canonicalize(config)
    serialized = json.dumps(
        canon,
        sort_keys=True,
        separators=(",", ":"),   # Remove whitespace → consistent output
        ensure_ascii=False,
    )
    h = hashlib.new(algo)
    h.update(serialized.encode("utf-8"))
    return h.hexdigest()[:length]


########################################################
### https://github.com/hsmoon121/amortized-inference-hci
def sort_and_pad_traj_data(stat_data, traj_data, value=0):
    """
    Sort and pad trajectory data based on their lengths.

    stat_data (ndarray): static data with a shape (num_data, stat_feature_dim).
    traj_data (list): List of trajectory data, each item should have a shape (traj_length, traj_feature_dim).
    value (int, optional): Padding value, default is 0.
    ---
    outputs (tuple): Tuple containing sorted static data (torch.Tensor), sorted padded trajectory data (torch.Tensor),
               padding lengths (torch.Tensor), and sorted indices (torch.Tensor).
    """
    # Get trajectory lengths
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)

    # Pad trajectory data
    padded_data = []
    for traj in traj_data:
        padded_data.append(np.pad(
            traj,
            ((0, max_len), (0, 0)),
            "constant",
            constant_values=value
        )[:max_len])

    # Sort trajectory data based on length
    lens = torch.LongTensor(traj_lens)
    lens, sorted_idx = lens.sort(descending=True)
    padded = max_len - lens

    # Get sorted static and trajectory data
    sorted_trajs = torch.FloatTensor(np.array(padded_data))[sorted_idx]
    sorted_stats = torch.FloatTensor(np.array(stat_data))[sorted_idx]
    return sorted_stats, sorted_trajs, padded, sorted_idx


def mask_and_pad_traj_data(traj_data, value=0):
    """
    Create a mask and pad trajectory data based on their lengths.

    traj_data (list): List of trajectory data, each item should have a shape (traj_length, traj_feature_dim).
    value (int, optional): Padding value, default is 0.
    ---
    outputs (tuple): Tuple containing padded trajectory data (torch.Tensor) and mask (torch.BoolTensor).
    """
    # Get trajectory lengths
    traj_lens = [traj.shape[0] for traj in traj_data]
    max_len = max(traj_lens)
    mask = torch.zeros((len(traj_data), max_len))

    # Pad trajectory data and create mask
    padded_data = []
    for i, traj in enumerate(traj_data):
        padded_data.append(np.pad(
            traj,
            ((0, max_len), (0, 0)),
            "constant",
            constant_values=value
        )[:max_len])
        mask[i, :traj_lens[i]] = 1

    # Get padded data and mask to tensors
    padded_trajs = torch.FloatTensor(np.array(padded_data))
    return padded_trajs, mask.bool()


def fourier_encode(x, max_freq, num_bands = 4):
    """
    Fourier feature postiion encodings
    reference: https://github.com/lucidrains/perceiver-pytorch
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[((None,) * (len(x.shape) - 1) + (...,))]

    x = x * scales * np.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


def get_auto_device():
    if torch is None:
        raise ModuleNotFoundError("torch is required for neural network utilities.")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


if __name__ == "__main__":
    print(get_current_time_digit())
