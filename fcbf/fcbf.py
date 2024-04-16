
from typing import Tuple
import logging

import numpy as np
import pandas as pd
from scipy.stats import entropy


logger = logging.getLogger(__name__)


def prior(X: pd.Series) -> pd.Series:
    """Calculates the prior of a feature.

    Args:
        X (pd.Series): The series of a feature's values.

    Returns:
        pd.Series: The prior.
    """
    n = X.size
    return X.value_counts()/n

def cond_proba(X: pd.Series, y: pd.Series) -> pd.Series:
    """Calculates the conditional probability of a feature given the class.
    # TODO: accept a nx2 df containing X and y

    Args:
        X (pd.Series): The series of a feature's values.
        y (pd.Series): The class values. Need to correspond to X.

    Returns:
        pd.Series: The conditional probability.
    """
    sample = pd.concat([X, y], axis=1)
    return sample.groupby([X.name, y.name]).size().div(len(sample.index)).div(prior(y), axis=0, level=y.name)

def cond_entropy(X: pd.Series, y: pd.Series, base: float =np.e) -> float:
    """Calculates the conditional entropy of a feature given the class.

    Args:
        X (pd.Series): The series of a feature's values.
        y (pd.Series): The class values. Need to correspond to X.
        base (float, optional): The logarithm base to apply. Defaults to np.e.

    Returns:
        float: The conditional entropy.
    """
    cond_proba_ = cond_proba(X, y)
    logged_cond_proba_ = np.log(cond_proba_) / np.log(base)
    prod = cond_proba_ * logged_cond_proba_
    return -1 * prod.sum(axis=0, level=y.name).mul(prior(y)).sum(axis=0)

def information_gain(X: pd.Series, y: pd.Series, base: float=np.e) -> float:
    """Calculates the information gain IG of a feature regarding the class.
    Formula: IG(X|y) = entropy(X) - cond_entropy(X|y)

    Args:
        X (pd.Series): The series of a feature's values.
        y (pd.Series): The class values. Need to correspond to X.
        base (float, optional): The logarithm base to apply. Defaults to np.e.

    Returns:
        float: The information gain.
    """
    entropy_ = entropy(prior(X), base=base)
    cond_entropy_ = cond_entropy(X, y, base=base)

    if entropy_ < 0:
        raise RuntimeError('entropy < 0 detected.')
    if entropy_ < cond_entropy_:
        raise RuntimeError('entropy < associated conditional entropy detected.')

    return entropy_ - cond_entropy_

def symmetrical_uncertainty(X: pd.Series, y: pd.Series, base: float=np.e) -> float:
    """Calculates the symmetrical uncertainty SU of a feature regarding the class.
    Formula: SU(X,y) = 2 * IG(X|y) / ( entropy(X) + entropy(y) )

    Args:
        X (pd.Series): The series of a feature's values.
        y (pd.Series): The class values. Need to correspond to X.
        base (float, optional): The logarithm base to apply. Defaults to np.e.

    Returns:
        float: The symmetrical uncertainty.
    """
    entropy_X = entropy(prior(X), base=base)
    entropy_y = entropy(prior(y), base=base)
    information_gain_ = information_gain(X, y, base=base)

    return 2 * information_gain_ / (entropy_X + entropy_y)

def fcbf(X: pd.DataFrame, y: pd.Series,
         su_threshold: float=0.0, base: float=np.e) -> Tuple[list, list, dict]:
    """Fast correlation-based filter algorithm introduced by Yu and Liu.
    @inproceedings{inproceedings,
    author = {Yu, Lei and Liu, Huan},
    year = {2003},
    month = {01},
    pages = {856-863},
    title = {Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution},
    volume = {2},
    journal = {Proceedings, Twentieth International Conference on Machine Learning}
    }

    Args:
        X (pd.DataFrame): The data.
        y (pd.Series): The class values. Need to correspond to X.
        su_threshold (float, optional): Lower bound for symmetrical uncertainty. Defaults to 0.0.
        base (float, optional): The logarithm base to apply. Defaults to np.e.

    Returns:
        Tuple[list, list, dict]: The relevant features' names, sorted by decreasing relevance, the
            irrelevant features' names and the correlation values by feature names.
    """

    # identify relevant features
    S = {}
    S_removed = []
    for feature_name in X.columns:
        f_i = X[feature_name]
        su_ic = symmetrical_uncertainty(X=f_i, y=y, base=base)

        if su_ic >= su_threshold:
            S[feature_name] = su_ic
        else:
            S_removed.append((feature_name, su_ic))
    if not S:
        raise ValueError('Found no features with symmetrical uncertainty, given the class, above'
                         ' threshold. Try a lower threshold.')

    # sort features by decreasing symmetrical_uncertainty
    S_ord = sorted(S.items(), key=lambda elem: elem[1], reverse=True)
    logger.debug(f'features sorted by decreasing symmetrical_uncertainty: {S_ord}')

    # remove redundant features
    removed_features = {}
    removed_features_indices = []
    for idx_p in range(len(S_ord)):
        if idx_p in removed_features_indices:
            continue
        feature_name_p, _ = S_ord[idx_p]
        logger.debug(f'\tf_p: {feature_name_p}')
        f_p = X[feature_name_p]

        idx_q = idx_p + 1
        if idx_q < len(S_ord):
            su_pq = []
            for i in range(idx_q, len(S_ord)):
                if i in removed_features_indices:
                    continue
                feature_name, su_ic = S_ord[i]
                f_i = X[feature_name]
                su_ip = symmetrical_uncertainty(X=f_p, y=f_i, base=base)
                logger.debug(f'\t\tf_q: {feature_name}')

                if su_ip >= su_ic:
                    removed_features_indices.append(i)
                    su_pq.append({feature_name: su_ip})
                    logger.debug(f'\t\t\tRedundant: su_pq = {su_ip}, su_qc = {su_ic}')

            removed_features.update({feature_name_p: su_pq})
        else:
            break

    S_ord_removed = []
    for i in sorted(removed_features_indices, reverse=True):
        S_ord_removed.append(S_ord.pop(i))

    logger.debug(f'removed features by causing feature: {removed_features}')
    logger.debug(f'best features: {S_ord}\n')

    # return results
    correlation_values = dict(S_ord + S_ord_removed)
    relevant_feature_names_sorted = [f_name for (f_name, _) in S_ord]
    irrelevant_feature_names = [f_name for (f_name, _) in S_ord_removed + S_removed]

    return relevant_feature_names_sorted, irrelevant_feature_names, correlation_values


if __name__ == '__main__':

    from fcbf import data


    dataset = data.lung_cancer
    X = dataset[dataset.columns[1:]]
    y = dataset[dataset.columns[0]].astype(int)
    print(X)
    print(y)

    relevant_features, irrelevant_features, correlations = fcbf(X, y, su_threshold=0.1, base=2)
    print('relevant_features:', relevant_features, '(count:', len(relevant_features), ')')
    print('irrelevant_features:', irrelevant_features, '(count:', len(irrelevant_features), ')')
    print('correlations:', correlations)
