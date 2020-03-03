import numpy as np
import pandas as pd
import statsmodels.api as sm


def calc_propensity(y, X):
    """
    Calculate propensity score from logistic regression
    
    Parameters
    ----------
    y : Pandas Series
    X : Pandas DataFrame
    
    Returns
    -------
    Numpy array of propensity scores
    
    """
    model = sm.Logit(y, X)
    res = model.fit()
    pscores = np.zeros(X.shape[0])
    pscores = res.predict(X)
    return pscores

def logit_ip_weights(y, X):
    """
    Create IP weights from logistic regression
    
    Parameters
    ----------
    y : Pandas Series
    X : Pandas DataFrame
    
    Returns
    -------
    Numpy array of IP weights
    
    """
    model = sm.Logit(y, X)
    res = model.fit()
    weights = np.zeros(X.shape[0])
    weights[y == 1] = res.predict(X.loc[y == 1])
    weights[y == 0] = 1 - res.predict(X.loc[y == 0])
    return weights


def calc_ip_weights(data_df, TX_col, ps_df, pscore_col):
    """
    Calculate Inverse-Propensity weights
    
    Parameters
    ----------
    data_df: DataFrame, contains the treatment column of cohort of interest
    TX_col: str, name of the treatment column in the data_df
    ps_df: DataFrame, contatins the propensity scores
    pscoer_col: str, name of the propensity score column in the ps_df
    
    Returns
    -------
    ps_df: DataFrame of the propensity score, with a new column named 'ip_number'
    """
    prob_cs = data_df[TX_col].mean()
    ps_df['ip_number'] = 0.0
    TX_idx = ps_df[TX_col] == 1
    ps_df.loc[TX_idx, 'ip_number'] = 1 - prob_cs
    ps_df.loc[~TX_idx, 'ip_number'] = prob_cs
    ps_df['ip_number'] = ps_df['ip_number'] / ps_df[pscore_col]
    
    return ps_df