# QUB-PHEO Motion-Prediction Benchmark

This repository accompanies the **submitted** IFAC J3C 2025 paper  
**“Establishing Baselines for Dyadic Visual Motion Prediction Using the QUB-PHEO Dataset.”**  
It provides code, configs, pre-trained weights and result logs for five temporal backbones on the QUB-PHEO collaborative-assembly dataset.

> **Status** – Paper under review (IFAC Joint Conference on Cyber-Physical & Human-Systems 2025).  
> Please cite the *submission* until the camera-ready version is available.

---

## Directory overview

```text
benchmark/
├─ cfg/                YAML configs.yaml
├─ datasets/           QUB-PHEO aerial-view dataset loader and preprocessing utils
├─ models/             BiLSTM, BiGRU, TCN, Transformer, ST-GCN inside models.py and utils scripts
├─ utils/              metric functions, plotting, helpers
├─ plots/              auto-generated learning-curve PNGs
├─ weights/            pre-trained *.pt files (120 epochs)
├─ results_*.md        markdown logs per backbone
├─ train.py            single-GPU training / evaluation script
