import os
import h5py
import csv
import datetime


def mergeHdf5Datasets(hdf5_1, hdf5_2,
                      datasets_to_merge=None,
                      rename_map=None,
                      log_path='hdf5_merge_log.csv'):
    """
    Merge selected datasets from one HDF5 file into another, with optional renaming,
    log any shape mismatches, and return a success flag.

    Parameters
    ----------
    hdf5_1 : str
        Path to the source HDF5 file (read-only).
    hdf5_2 : str
        Path to the destination HDF5 file (append mode).
    datasets_to_merge : list of str, optional
        List of dataset names (or full paths) in `hdf5_1` to copy.
        Defaults to ['bounding_boxes', 'normalized_gaze'].
    rename_map : dict, optional
        Mapping from original dataset names to new names in `hdf5_2`.
        E.g. {'bounding_boxes': 'bboxes', 'normalized_gaze': 'norm_gaze'}.
    log_path : str, optional
        CSV file path where shape‐mismatch events are appended.
        Defaults to 'hdf5_merge_log.csv'.

    Returns
    -------
    bool
        True if *all* datasets in `datasets_to_merge` were found and merged
        (with no shape mismatches); False if *any* were missing or mismatched.
    """
    # defaults
    if datasets_to_merge is None:
        datasets_to_merge = ['bounding_boxes', 'normalized_gaze']
    if rename_map is None:
        rename_map = {
            'bounding_boxes': 'bboxes',
            'normalized_gaze': 'norm_gaze'
        }

    # ensure log header exists once
    header = ['timestamp', 'source_file', 'src_dataset', 'src_shape',
              'dest_file', 'dest_dataset', 'dest_shape']
    if not os.path.isfile(log_path):
        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    all_success = True

    with h5py.File(hdf5_1, 'r') as f1, h5py.File(hdf5_2, 'a') as f2:
        for src_name in datasets_to_merge:
            if src_name not in f1:
                print(f"Warning: '{src_name}' not found in {hdf5_1}. Skipping.")
                all_success = False
                continue

            dest_name = rename_map.get(src_name, src_name)

            # if target exists, compare shapes
            if dest_name in f2:
                src_shape = f1[src_name].shape
                dest_shape = f2[dest_name].shape

                if src_shape != dest_shape:
                    print(f"Shape mismatch for {src_name} → {dest_name}: "
                          f"source {src_shape} vs destination {dest_shape}. Skipping.")
                    # log mismatch
                    with open(log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            datetime.datetime.now().isoformat(),
                            hdf5_1, src_name, src_shape,
                            hdf5_2, dest_name, dest_shape
                        ])
                    all_success = False
                    continue
                else:
                    # safe to overwrite
                    del f2[dest_name]

            # perform the copy
            f1.copy(src_name, f2, name=dest_name)

    return all_success


def delete_dataset_hf5(hdf5_file, dataset_name):
    """
    Deletes a dataset from an HDF5 file if it exists.
    :param hdf5_file: Path to the HDF5 file.
    :param dataset_name: Name of the dataset to delete.
    """
    with h5py.File(hdf5_file, 'a') as f:
        if dataset_name in f:
            del f[dataset_name]
            print(f"✅  Deleted dataset '{dataset_name}' from '{hdf5_file}'.")
            return True
        else:
            print(f"❌ Dataset '{dataset_name}' not found in '{hdf5_file}'.")
            return False


if __name__ == '__main__':
    """=======================TESTING UTILS OF DELETING DATASET IN HDF5 ========================================"""
    hd5_file = 'sample_data/delete_dataset/p01-CAM_AV-BIAH_RB-BHO-25.420644039568998_26.674916343042494.h5'
    dataset_name = 'bboxes'
    delete_dataset_hf5(hd5_file, dataset_name)


    """=======================TESTING UTILS OF MERGING DATASET ========================================"""
    h5_1 = '/home/samuel/ml_projects/QUBPHEO/benchmark/test_run/p1/BHO/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.h5'
    h5_2 = '/home/samuel/ml_projects/QUBPHEO/benchmark/test_run/p2/BHO/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.h5'

    success = mergeHdf5Datasets(
        h5_1,
        h5_2,
        datasets_to_merge=['bounding_boxes', 'normalized_gaze'],
        rename_map={'bounding_boxes': 'bboxes', 'normalized_gaze': 'norm_gaze'}
    )
    if success:
        print("Merge completed without issues.")
    else:
        print("Merge completed with issues. Check logs for details.")

