# Image Authenticity Detector

A production-ready system for detecting AI-generated images using Vision Transformers and CNNs.

## Project Structure

```
imgauthenticator/
├── data/              # Dataset storage
├── src/               # Source code
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks
└── tests/             # Unit tests
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure paths in `configs/train_config.yaml`

3. Collect datasets (see `src/data/`)

4. Train model:
```bash
python src/training/train.py
```

5. Run API:
```bash
uvicorn src.api.main:app --reload
```

## Documentation

- [Dataset Curation Guide](docs/dataset_curation.md)
- [Training Guide](docs/training.md)
- [API Documentation](http://localhost:8000/docs)
