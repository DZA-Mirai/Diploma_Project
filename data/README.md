## Dataset

This project uses the **MovieLens 32M** dataset provided by GroupLens.

Due to its large size (~856 MB), the dataset is not included in this repository.

### Download instructions

1. Download the MovieLens 32M dataset from:
   https://grouplens.org/datasets/movielens/32m/

2. Extract the archive.

3. Place the files in the following structure:
data/
└── ml-32m/
├── ratings.csv
├── movies.csv
└── README.txt

The code expects the dataset to be located at `data/ml-32m/`.