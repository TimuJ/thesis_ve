# Baseline Datasets

## Download

```bash
bash experiments/baselines/data/download_dove.sh UDM10
```

Requires `gdown`: `pip install gdown`

## Manual Download (if gdown hits quota)

1. Go to [DOVE Google Drive](https://drive.google.com/drive/folders/1yNKG6rtTNtZQY8qL74GoQwA0jgjBUEby)
2. Download the UDM10 folder
3. Place at `experiments/baselines/data/UDM10/`

Expected structure:
```
UDM10/
├── GT/          # per-clip subdirs with HR frames
└── LQ/          # per-clip subdirs with LQ frames
```
