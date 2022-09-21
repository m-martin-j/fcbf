
import pathlib

import pandas as pd


_lung_cancer_path = pathlib.Path(__file__).parent.joinpath('lung-cancer.data')
lung_cancer = pd.read_csv(_lung_cancer_path, header=None, na_values='?')
