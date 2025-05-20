import numpy as np
import torch

def build_hand_adjacency():
    # Skeleton connectivity for one hand (MediaPipe 21‐point format):
    # 0: wrist
    # thumb: 1→2→3→4
    # index: 5→6→7→8
    # middle: 9→10→11→12
    # ring: 13→14→15→16
    # pinky: 17→18→19→20
    finger_chains = [
        [0, 1, 2, 3, 4],    # thumb
        [0, 5, 6, 7, 8],    # index
        [0, 9, 10,11,12],   # middle
        [0,13,14,15,16],    # ring
        [0,17,18,19,20],    # pinky
    ]

    # build undirected edges for a single hand
    left_edges = []
    for chain in finger_chains:
        for u, v in zip(chain, chain[1:]):
            left_edges.append((u, v))

    # mirror for right hand (indices 21–41)
    right_edges = [(u+21, v+21) for (u, v) in left_edges]

    # combine
    edges = left_edges + right_edges

    # build adjacency
    A = np.zeros((42, 42), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # add self-loops
    for i in range(42):
        A[i, i] = 1

    return torch.from_numpy(A)

if __name__ == "__main__":
    # Example usage
    A = build_hand_adjacency()   # shape (42,42), float32
    print("Adjacency matrix shape:", A.shape)
    print("Sample row (joint 0 connections):", A[0].nonzero().flatten().tolist())