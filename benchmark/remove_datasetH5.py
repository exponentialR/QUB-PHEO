from utils import delete_dataset_hf5
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

__author__ = "Samuel Adebayo"

def remove_dataset_h5 (h5py_directory, dataset_name):
    """
    Remove a dataset from an HDF5 file.
    Parameters:
    h5Directory (str): Path to the HDF5 file.
    datasetName (str): Name of the dataset to remove.
    """
    subtaskList = sorted(list(h5py_directory.rglob('*.h5')))
    removed_count = 0
    not_found_count = 0
    for subtask_h5 in tqdm(subtaskList, desc="Removing datasets", unit="file"):
        success = delete_dataset_hf5(subtask_h5, dataset_name)
        if success:
            removed_count += 1
        else:
            not_found_count += 1

    summary = [
        ["Total Files", len(subtaskList)],
        ["Removed Datasets", removed_count],
        ["Not Found Datasets", not_found_count]
    ]
    print(tabulate(summary, headers=["Description", "Count"], tablefmt="grid"))
    print(f"✅ Removed {removed_count} datasets from {len(subtaskList)} files.")
    print(f"❌ Could not find {dataset_name} in {not_found_count} files.")

if __name__ == '__main__':
    h5Directory = Path('/home/samuel/ml_projects/QUBPHEO/benchmark/landmarks')
    remove_dataset_h5(h5Directory, dataset_name='bboxes')