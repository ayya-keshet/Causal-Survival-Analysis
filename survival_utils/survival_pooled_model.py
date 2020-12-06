import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def fit_pooled_lr(pooled_df, formula, weights_col=None):
    """
    Fit model to a person-time format data frame
    
    Parameters
    ----------
    pooled_df : DataFrame,  contains columns Y_col and X_cols
    formula: str, formula to use to fit the Logit to data
    weights_col: str, name of the weights columns. In None data will not be weighted
    
    Returns
    -------
    fit_model: a model object fitted to the pooled_df using the given formula
    """
    if weights_col is not None:
        model = sm.GLM.from_formula(formula=formula, data=pooled_df, 
                                      freq_weights=pooled_df[weights_col], family=sm.families.Binomial()) 
    else:
        model = sm.GLM.from_formula(formula=formula, data=pooled_df, family=sm.families.Binomial()) 
    fit_model = model.fit()
    return(fit_model)


def survival_curve(hazard):
    """
    Calculate survival curve from hazard. Survival at 𝑘 equals the product of one minus the hazard at all previous times.
    
    Parameters
    ----------
    hazard: Series, hazard values, mostly predicted from an LR model
    
    Returns
    -------
    survival: list, with same length as the given hazard. Survival values at all times calculated by multiplying 
              at time k : (1-hazard[k])*survival[k-1]
    """
    survival = [1 - hazard[0]]
    for i in range(1, len(hazard)):
        survival.append((1 - hazard[i]) * survival[i - 1])
    return survival


def get_survival_curves_from_lr(model, TX_col, TX_labels_dict, followup_max_time_from_index):
    """
    Predict binary treatment survival curves from a pooled Logistic Regression model (without standardization)
    
    Parameters
    ----------
    model : estimtor Object, fitted on a pearson-time data
    TX_col : str, name of the treatment column in data
    TX_labels_dict: dict, keys are TX values, values are lables of the treated and un-treated plots
    followup_max_time_from_index: int, max follow up time from index. e.g. if we want until 10 years old we can put 10*DAYS_IN_YEAR 
    """
    A0_pred = model.predict(pd.DataFrame({'k': list(range(followup_max_time_from_index)), TX_col: [0] * followup_max_time_from_index}))
    A1_pred = model.predict(pd.DataFrame({'k': list(range(followup_max_time_from_index)), TX_col: [1] * followup_max_time_from_index}))
    model_surv_0 = survival_curve(A0_pred)
    model_surv_1 = survival_curve(A1_pred)
    
    return model_surv_0, model_surv_1, A0_pred, A1_pred

