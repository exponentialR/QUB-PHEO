import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class QUBPHEOAerialDataset(Dataset):
    """
    Pytorch Dataset for QUBPHEO aerial-view hand-motion forecasting.
    Each sample is an (obs, pred) pair,
    where obs is the input sequence of hand landmarks and pred is the target sequence of hand landmarks.
        - obs: tensor of shape (obs_len, 42 *2)
        - pred: tensor of shape (pred_len, 42*2)
    """

    def __init__ (self, csv_path:str, h5_dir: str, split: str='train',
                    obs_len:int=60, pred_len:int=60, stride:int=15,
                  include_gaze:bool=True, include_obj_bbox:bool=True, include_surrogate_bbox:bool=True):
        """
        Args:
            csv_path (str): Path to subtasks_byTask_time.csv; the CSV file containing HDF5 files.
            h5_dir (str): Directory containing the HDF5 files.
            split (str): Split type ('train', 'val', 'test').
            obs_len (int): Number of past frames (observation window).
            pred_len (int): Number of future frames to forecast.
            stride (int): Step size between windows.
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.include_gaze = include_gaze
        self.include_obj_bbox = include_obj_bbox
        self.include_surrogate_bbox = include_surrogate_bbox

        df = pd.read_csv(csv_path, sep='\t')
        df = df[df['camera'] == 'CAM_AV']
        # df = df[df['filename'].str.contains(r'-CAM_AV-', regex=True)]
        n_subject = 70

        train_ids = set(range(1, 49))
        val_ids = set(range(49, 60))
        test_ids = set(range(60, n_subject + 1))

        split_map = {'train': train_ids, 'val': val_ids, 'test': test_ids}
        if split not in split_map:
            raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', or 'test'.")

        df['pid'] = df['filename'].str.extract(r'p(\d+)-').astype(int)
        df = df[df['pid'].isin(split_map[split])]

        self.subtasks = sorted(df['subtask'].unique())
        self.subtask2idx = {subtask: i for i, subtask in enumerate(self.subtasks)}

        self.samples = []
        for _, row in df.iterrows():
            subtask = row['subtask']
            fname = row['filename']

            fpath = os.path.join(h5_dir, subtask, fname).replace('.mp4', '.h5')
            if fpath in self.open_missing_files():
                continue
            # print(f'Loading {fpath}')
            if not os.path.exists(fpath):
                print(f'File not found: {fpath}')
                continue
            if not os.path.isfile(fpath):
                continue
            with h5py.File(fpath, 'r') as f:
                F = f['left_landmarks'].shape[0]
            max_start = F - (obs_len + pred_len)
            for start in range(0, max_start + 1, stride):
                self.samples.append((fpath, start, self.subtask2idx[subtask]))

    def __len__(self):
        return len(self.samples)

    def open_missing_files(self):
        with open('missing_files.csv', 'r') as f:
            lines = f.readlines()
        missing_files = [line.strip() for line in lines]
        return missing_files


    def __getitem__(self, idx):
        fpath, start, subtask_idx = self.samples[idx]
        # print(f'Loading {fpath}')
        with h5py.File(fpath, 'r') as f:
            left = f['left_landmarks'][()]
            right = f['right_landmarks'][()]
            gaze = f['norm_gaze'][()]
            obj_bbox = f['rec_bboxes'][()]
            surrogate_bbox = f['surrogate_hands'][()]

        left_xy = left[..., :2]  # (F,21,2)
        right_xy = right[..., :2]  # (F,21,2)
        hands = np.concatenate([left_xy, right_xy], axis=1)  # (F,42,2)
        wc = 0.5 * (hands[:, 0, :] + hands[:, 21, :])
        hands_centered = (hands - wc[:, None, :]) / 0.5

        obs_h = hands_centered[start:start + self.obs_len] # (obs, 42, 2)
        pred_h = hands_centered[start + self.obs_len:start + self.obs_len + self.pred_len]

        sample = {
            'obs_h': torch.from_numpy(obs_h.reshape(self.obs_len, -1)).float(),
            'pred_h': torch.from_numpy(pred_h.reshape(self.pred_len, -1)).float(),
            'subtask':torch.tensor(subtask_idx, dtype=torch.long)
        }

        # Gaze Context
        if self.include_gaze:
            obs_g = gaze[start:start + self.obs_len]
            pred_g = gaze[start + self.obs_len:start + self.obs_len + self.pred_len]
            sample['obs_gaze'] = torch.from_numpy(obs_g).float()
            sample['pred_gaze'] = torch.from_numpy(pred_g).float()

        # Object BBoxes context
        if self.include_obj_bbox:
            obs_b = obj_bbox[start:start + self.obs_len].reshape(self.obs_len, -1)
            pred_b = obj_bbox[start + self.obs_len:start + self.obs_len + self.pred_len].reshape(self.pred_len, -1)
            sample['obs_obj_box'] = torch.from_numpy(obs_b).float()
            sample['pred_obj_box'] = torch.from_numpy(pred_b).float()

        # Surrogate BBoxes context
        if self.include_surrogate_bbox:
            obs_s = surrogate_bbox[start:start+self.obs_len].reshape(self.obs_len, -1)
            pred_s = surrogate_bbox[start+self.obs_len:start+self.obs_len+self.pred_len].reshape(self.pred_len, -1)
            sample['obs_sur_box'] = torch.from_numpy(obs_s).float()
            sample['pred_sur_box'] = torch.from_numpy(pred_s).float()

        return sample

if __name__ == '__main__':
    file_dir = os.path.dirname(__file__)
    csv_path = os.path.join(file_dir, 'subtasks_byTask_time.csv')
    for split in ['train', 'val', 'test']:

        dataset = QUBPHEOAerialDataset(
            csv_path=csv_path,
            h5_dir='/home/samuel/ml_projects/QUBPHEO/benchmark/landmarks',
            split=split,
            obs_len=60,
            pred_len=60,
            stride=15,
            include_gaze=True,
            include_obj_bbox=True,
            include_surrogate_bbox=True
        )
        print(f'Split: {split}: {len(dataset)} samples')
    print(f'Number of samples: {len(dataset)}')
    # csv_path = 'subtasks_byTask_time.csv'
    # import pandas as pd
    #
    # df = pd.read_csv(csv_path, sep='\t')
    # print(df.columns)

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #
    # for batch in dataloader:
    #     print(batch)

    import pandas as pd, re

    df = pd.read_csv(csv_path, sep='\t')
    df['pid'] = df['filename'].str.extract(r'p(\d+)-').astype(int)

    splits = {'train': range(1, 49), 'val': range(49, 60), 'test': range(60, 71)}
    for split, ids in splits.items():
        n_clips = df[df['pid'].isin(ids)].shape[0]
        print(split, n_clips)
