"""
    Merge matching HDF5 files from one project directory into another,
    log missing sources, record shape‐mismatches, and summarise per-file stats.

    Author
    ------
    Samuel Adebayo

    Date
    ----
    2025-05-13

    Parameters
    ----------
    project1_dir : str
        Path to the source project directory containing HDF5 subfolders.
    project2_dir : str
        Path to the target project directory where datasets will be merged.
    missing_log : str, optional
        TSV path for logging files present in project2 but missing in project1.
        Defaults to 'missing_files.csv'.
    metadata_log : str, optional
        CSV path for saving the final summary of dataset names, shapes,
        dtypes and file‐level metadata. Defaults to 'metadata_summary.csv'.

    Behaviour
    ---------
    1. Finds all `.h5` files under `project2_dir` subfolders.
    2. Logs any whose counterpart in `project1_dir` is absent.
    3. Calls `mergeHdf5Datasets` for each pair, counting successes and skips.
    4. After each merge (or skip), opens the destination file to gather:
       - Number of datasets
       - Dataset names, shapes and dtypes
       - File‐level HDF5 attributes
    5. Prints a summary table via `tabulate` and writes `metadata_log`.
    """
__author__ = "Samuel Adebayo"


import os
from tqdm import tqdm
from utils import mergeHdf5Datasets
import pandas as pd
from tabulate import tabulate
import h5py

def mergeH5Dataset(project1_dir, project2_dir):
    subtasks2 = sorted([os.path.join(project2_dir, subtask) for subtask in os.listdir(project2_dir) if os.path.isdir(os.path.join(project2_dir, subtask))])
    subtask_h5_files2= sorted([os.path.join(subtask, file) for subtask in subtasks2 for file in os.listdir(subtask) if file.lower().endswith('.h5')])

    missing_files = [h5_files_2 for h5_files_2 in subtask_h5_files2 if not os.path.exists(h5_files_2.replace(project2_dir, project1_dir))]
    missing_files_df = pd.DataFrame(missing_files, columns=['missing_file'])
    missing_files_df.to_csv('missing_files.csv', index=False, sep='\t', mode='w', header=True)

    not_exist = 0
    merged = 0
    stats = []

    for h5_2 in tqdm(subtask_h5_files2, desc='MERGING H5 FILES'):
        h5_1 = h5_2.replace(project2_dir, project1_dir)
        entry = {'file':h5_2}
        if os.path.exists(h5_1):
            success = mergeHdf5Datasets(h5_1, h5_2)
            if success:
                merged += 1
                entry['status'] = 'merged'
                print(f'✅ Dataset Merged: {h5_1} into {h5_2}')
            else:
                entry['status'] = 'skipped_or_mismatch'
                print(f'❌ Failed to merge: {h5_1} and {h5_2} Check logs for details.')
            with h5py.File(h5_2, 'r') as f2:
                ds_names = list(f2.keys())
                shapes = [f2[n].shape for n in ds_names]
                dtypes = [str(f2[n].dtype) for n in ds_names]
                attrs = dict(f2.attrs)  # file-level metadata

            entry.update({
                'num_datasets': len(ds_names),
                'dataset_names': ','.join(ds_names),
                'shapes': ','.join(map(str, shapes)),
                'dtypes': ','.join(dtypes),
                'metadata_attrs': ','.join(f"{k}={v}" for k, v in attrs.items())
            })

            stats.append(entry)


        else:
            entry.update({
                'status': 'missing_source',
                'num_datasets': None,
                'dataset_names': None,
                'shapes': None,
                'dtypes': None,
                'metadata_attrs': None
            })
            not_exist+=1
            print(f'File does not exist: {h5_1}')
    print(f'\nNot exist count: {not_exist}')
    print(f'Successful merges: {merged} / {len(subtask_h5_files2)}\n')

    print(tabulate(
        stats,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        missingval="-"
    ))

    pd.DataFrame(stats).to_csv('metadata_summary.csv', index=False)






if __name__ == "__main__":
    project1_dir = '/home/samuel/Downloads/CAM_AV'
    project2_dir = '/home/samuel/ml_projects/QUBPHEO/benchmark/landmarks'
    # project1_dir = '/home/samuel/ml_projects/QUBPHEO/benchmark/test_run/p1'
    # poject2_dir = '/home/samuel/ml_projects/QUBPHEO/benchmark/test_run/p2'
    mergeH5Dataset(project1_dir, project2_dir)

