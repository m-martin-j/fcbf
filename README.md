# Fast Correlation-Based Filter
A categorical feature selection approach based on information theoretical considerations.

Implementation of the fast correlation-based filter (FCBF) proposed by Yu and Liu:

```bibtex
@inproceedings{inproceedings,
author = {Yu, Lei and Liu, Huan},
year = {2003},
month = {01},
pages = {856-863},
title = {Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution},
volume = {2},
journal = {Proceedings, Twentieth International Conference on Machine Learning}
}
``` 

Data for testing is taken from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml)

## Example

```py
from fcbf import fcbf, data


dataset = data.lung_cancer
X = dataset[dataset.columns[1:]]
y = dataset[dataset.columns[0]].astype(int)
print(X)
print(y)

relevant_features, irrelevant_features, correlations = fcbf(X, y, su_threshold=0.1, base=2)
print('relevant_features:', relevant_features, '(count:', len(relevant_features), ')')
print('irrelevant_features:', irrelevant_features, '(count:', len(irrelevant_features), ')')
print('correlations:', correlations)
```

## Setup
Using pip, execute the following

```sh
pip install fcbf
```

## Development
TODO

## Contributing
TODO

## License
Code is released under the [MIT License](LICENSE).
All dependencies are copyright to the respective authors and released under the respective licenses.
