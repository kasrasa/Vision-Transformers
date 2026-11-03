from torchvision.datasets import Imagenette

def download_prepare_dataset(root_dir: str, split: str = "train"):
    """
    Download the Imagenette dataset.

    Args:
        root_dir (str): The root directory where the dataset will be downloaded.
        split (str): The split of the dataset to download. Can be 'train' or 'val'.
    """
    if split not in ["train", "val"]:
        raise ValueError("split must be either 'train' or 'val'")

    # Download the dataset
    dataset = Imagenette(root=root_dir, split=split, download=True)

    return dataset
