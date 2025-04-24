import yaml
import os

from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader, build_fasterrcnn_dataset


def load_args(args):
    """
    Load arguments from a YAML file and convert them to a namespace object.
    
    Args:
        args (str or dict): Path to the YAML file or a dictionary of arguments.
    
    Returns:
        ArgsNamespace: A namespace object containing the arguments.
    """
    if isinstance(args, str):
        with open(args, "r") as f:
            args = yaml.safe_load(f)
    elif not isinstance(args, dict):
        raise ValueError("args must be a path to a YAML file or a dictionary.")
    
    return ArgsNamespace(args)

def get_dataloader(args, mode="train"):
    assert mode in {"train", "val", "test"}, f"Invalid mode: {mode}"
    data = yaml.safe_load(open(args.data, "r"))
    img_path = os.path.join(data["path"], data[mode])
    if mode == "test":
        mode = "val"
    with torch_distributed_zero_first(0):
        dataset = build_fasterrcnn_dataset(args, img_path, args.batch, data, mode=mode, rect=(mode == "val"))
    shuffle = (mode == "train")
    workers = min(args.workers, os.cpu_count())
    return build_dataloader(dataset, args.batch, workers, shuffle, rank=0)



class ArgsNamespace:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ArgsNamespace(v)
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return self.__dict__