## Data

The MovieLens 32M dataset is required to run the experiments.
Due to its size, it is not included in the repository.

See `data/README.md` for download and setup instructions.

## Experiments

All experiments were originally conducted in the Jupyter notebook
notebooks/experiments_log.ipynb
which serves as an experiment log and records the full research process, including intermediate trials and parameter tuning.

Core components of the system (data preparation, evaluation metrics, and matrix factorization model) were subsequently refactored into reusable Python modules located in the src/ directory.

The notebook remains in the repository for transparency and reproducibility.