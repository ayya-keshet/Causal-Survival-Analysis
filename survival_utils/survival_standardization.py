## NOT DONE YET #####

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import math

from survival_preprocessing import make_pooled_survival_data, prepare_pooled_lr_data, DAYS_IN_YEAR
from survival_pooled_model import fit_pooled_lr



def add_interatcions_to_pooled(pooled_df, TX_col):
    """
    Add interaction terms to a pooled person-time format data frame
    
    Parameters
    ----------
    pooled_df: DataFrame, person-time format data frame, must contain the columns 'k' and TX_col
    TX_col: str, name of the treatment column in pooled_df
    
    Returns
    -------
    pooled_df: altered data frame with additional interaction columns of k^2, TX_col*k and TX_col*k^2
    interaction_cols: list of strings, names of the columns added to the data frame
    """
    pooled_df[f'{TX_col}_:_k'] = pooled_df[TX_col] * pooled_df['k']
    pooled_df[f'{TX_col}_:_k^2'] = pooled_df[TX_col] * pooled_df['k'] * pooled_df['k']
    pooled_df['k^2'] =  pooled_df['k'] * pooled_df['k']
    interaction_cols = [f'{TX_col}_:_k', f'{TX_col}_:_k^2', 'k^2']
    return pooled_df, interaction_cols


def fit_std_lr(pooled_df, model, TX_col, ps_cols):
    """
    Fit a given model to a pooled perason-time data frame with treatment-time interactions
    
    Parameters
    ----------
    pooled_df: DataFrame, person-time format data frame, must contain the columns 'k' and TX_col
    model: estimator object, must implement the function "fit"
    TX_col: str, name of the treatment column in pooled_df
    ps_cols: list of strings, names of the columns to include in the model
    
    Returns
    -------
    model: estimator object after fitting it to the data
    """
    
    pooled_df, interaction_cols = add_interatcions_to_pooled(pooled_df, TX_col)
    tmp_X = pooled_df[ps_cols + interaction_cols + [TX_col, 'k']]
    tmp_y = pooled_df['Y_k']
    model.fit(tmp_X, tmp_y)
    return model


def plot_survival_from_standardized_lr(model, survival_df, TX_col, TX_labels_dict, id_col, event_col='E', event_time_col='T', 
                                       index_time_col='preg_start', followup_max_time_from_index=10*12,
                                       preprocessor=None, process_cols=[], outcome_name=''):
    """
    Calculate and plot average survival curves from a Logistic Regression model fitted on a pearson-time dataframe with
    standartization
    
    Parameters
    ----------
    model : estimtor Object, fitted on the pearson-time data, with covariates. Implements "predict" method
    survival_df : DataFrame, contains the same covariates the model was trained on, and columns event_col and event_time_col
    TX_col : str, name of the treatment column in data
    TX_labels_dict: dict, keys are TX values, values are lables of the treated and un-treated plots
    id_col: str, name of the id columns in data
    event_col: str, name of the boolean column which indicates whether the outcome even occured or not
    event_time_col: str, name of the time column which indicates when the event occured\end of follow up if the event
                    did not occur
    index_time_col: str, name of the index time columns. Default is 'preg_start'
    followup_max_time_from_index: int, max time allowed from index. e.g. if we want until 10 years old we can 
                                  put 10*DAYS_IN_YEAR
    preprocessor: ColumnTransformer object, implements method "fit_transform". Default is None
    process_cols: list, names of columns to pass to the preprocessor. Default is an empty list
    outcome_name: str, name of the outcome plotted
    """
    survivals_TX0 = np.zeros((survival_df.shape[0], followup_max_time_from_index))
    survivals_TX1 = np.zeros((survival_df.shape[0], followup_max_time_from_index))
    
    data = survival_df.copy()
    data[event_col] = 0
    # Do we need to put the maximum followup time or the child's age?
    data[event_time_col] = followup_max_time_from_index # - data[index_time_col]
    
    pooled_df = make_pooled_survival_data(data, id_col, event_col, event_time_col)
    if preprocessor is not None:
        pooled_df[process_cols] = pd.DataFrame(preprocessor.fit_transform(pooled_df[process_cols]), columns=process_cols)
    
    # TX_col = 0 
    pooled_df[TX_col] = 0
    res_df = pooled_df[[id_col, TX_col, 'k']]
    res_df['hazard0'] = model.predict(pooled_df.drop(['Y_k'], axis=1))
    res_df['one_minus_hazard0'] = 1- res_df['hazard0']
    res_df['survival_TX0'] = res_df.groupby([id_col])['one_minus_hazard0'].cumprod()
    
    # TX_col = 1
    pooled_df[TX_col] = 1
    res_df['hazard1'] = model.predict(pooled_df.drop(['Y_k'], axis=1))
    res_df['one_minus_hazard1'] = 1- res_df['hazard1']
    res_df['survival_TX1'] = res_df.groupby([id_col])['one_minus_hazard1'].cumprod()

    surv_gform_0 = res_df.groupby(['k'])['survival_TX0'].mean()
    surv_gform_1 = res_df.groupby(['k'])['survival_TX1'].mean()
    
    min_0 = min(surv_gform_0)
    min_1 = min(surv_gform_1)
    
    min_val = min(min_0, min_1)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(surv_gform_0)
    ax.plot(surv_gform_1)
    ax.set_ylim(0.9*min_val, 1)
    ax.set_xlabel('Months of follow-up', fontsize=14)
    ax.set_ylabel('Survival probability', fontsize=14)
    ax.set_title('{}\nPooled LR - standardized'.format(outcome_name))
    ax.text(0.85*followup_max_time_from_index, min_0 - 0.001, '{}'.format(TX_labels_dict[0]))
    ax.text(0.85*followup_max_time_from_index, min_1 + 0.001, '{}'.format(TX_labels_dict[1]))
    

def plot_outcome_pooled_lr_standardized(data, outcome_df, outcome_time_col, outcome_name, id_col, index_time_col, TX_col, TX_labels_dict,
                                        followup_end_time, followup_max_time_from_index, ps_df, ps_cols, months_bins, 
                                        preprocessor, process_cols):
    """
    Fit a pooled LR model using standardization and plots the survival curves obtained from it.
    
    Parameters
    ----------
    data : DataFrame, contains the population of interest with its features. 
           Must contain the columns passed in id_col and index_time_col
    outcome_name: str, name of the outcome
    outcome_df : DataFrame, contains the outcome of interest
    outcome_time_col: str, name of the time column of the outcome_df which contains the time of outcome event
    outcome_name: str, name of the outcome
    id_col: str, name of the id column in the outcome_df
    index_time_col: str, name of the index time column in the outcome df
    TX_col: str, name of the treatment column in the outcome df
    TX_labels_dict: dict, keys are TX values, values are lables of the treated and un-treated plots
    followup_end_time: int, the time at which adminiastrative censoring is applied. e.g. in Clalit 1/1/2018 = 18*DAYS_IN_YEAR
    followup_max_time_from_index: int, max time allowed from index. e.g. if we want until 10 years old we can put 10*DAYS_IN_YEAR
    ps_df: DataFrame, contains the propensity scores and the columns ps_cols
    ps_cols: list of strings, names of the columns to add to the LR model
    months_bins: int, number of months to take as an interval to create person-time format data
    preprocessor: ColumnTransformer object, implements method "fit_transform". Default is None
    process_cols: list, names of columns to pass to the preprocessor. Default is an empty list
    
    Returns
    ---------
    Doesn't return value, plots the survival plot
    """
    
    pooled_df, survival_df = prepare_pooled_lr_data(data, outcome_df, outcome_time_col, id_col, index_time_col, TX_col,
                                                    followup_end_time, followup_max_time_from_index, months_bins, ps_df, ps_cols)
    if preprocessor is not None:
        pooled_df[process_cols] = pd.DataFrame(preprocessor.fit_transform(pooled_df[process_cols]), columns=process_cols)
        
    formula = 'Y_k ~ {} + {}:k + {}:I(k ** 2) + k + I(k ** 2)'.format(TX_col, TX_col, TX_col)
    
    for col in ps_cols:
        formula = formula + ' + {}'.format(col)
    fit_model = fit_pooled_lr(pooled_df, formula=formula) 
    
    followup_max_time_from_index_binned = math.ceil((followup_max_time_from_index*(12/months_bins))/DAYS_IN_YEAR)
    plot_survival_from_standardized_lr(model=fit_model, survival_df=survival_df, TX_col=TX_col, TX_labels_dict=TX_labels_dict, 
                                       id_col=id_col, event_col='E', event_time_col='T', index_time_col=index_time_col, 
                                       followup_max_time_from_index=followup_max_time_from_index_binned, 
                                       preprocessor=preprocessor, process_cols=process_cols, outcome_name=outcome_name)
    
    
def get_survival_curves_from_standartization(model, survival_df, id_col, TX_col, event_col, event_time_col, followup_max_time_from_index, 
                                             preprocessor, process_cols, ps_cols):
    """
    Calculate average survival curves from a model fitted on a pearson-time dataframe with standartization
    
    Parameters
    ----------
    model : estimtor Object, fitted on the pearson-time data, with covariates. Implements "predict" method
    survival_df : DataFrame, contains the same covariates the model was trained on, and columns event_col and event_time_col
    id_col: str, name of the id columns in data
    TX_col : str, name of the treatment column in data
    event_col: str, name of the boolean column which indicates whether the outcome even occured or not
    event_time_col: str, name of the time column which indicates when the event occured\end of follow up if the event
                    did not occur
    followup_max_time_from_index: int, max time allowed from index. e.g. if we want until 10 years old we can 
                                  put 10*DAYS_IN_YEAR
    preprocessor: ColumnTransformer object, implements method "fit_transform". Default is None
    process_cols: list, names of columns to pass to the preprocessor. Default is an empty list
    
    """
    
    survivals_TX0 = np.zeros((survival_df.shape[0], followup_max_time_from_index))
    survivals_TX1 = np.zeros((survival_df.shape[0], followup_max_time_from_index))
       
    data = survival_df.copy()
    data[event_col] = 0
    # Do we need to put the maximum followup time or the child's age?
    data[event_time_col] = followup_max_time_from_index # - data[index_time_col]
    
    pooled_df = make_pooled_survival_data(data, id_col, event_col, event_time_col)
    
    if preprocessor is not None:
        pooled_df[process_cols] = pd.DataFrame(preprocessor.fit_transform(pooled_df[process_cols]), columns=process_cols)
    
    # TX_col = 0 
    pooled_df[TX_col] = 0
    res_df = pooled_df[[id_col, TX_col, 'k']].copy()
    
    pooled_df, interaction_cols = add_interatcions_to_pooled(pooled_df, TX_col)
    tmp_X = pooled_df[ps_cols + interaction_cols + [TX_col, 'k']]
    
    res_df['hazard0'] = model.predict_proba(tmp_X)[:,1]
    res_df['one_minus_hazard0'] = 1- res_df['hazard0']
    res_df['survival_TX0'] = res_df.groupby([id_col])['one_minus_hazard0'].cumprod()
    
    # TX_col = 1
    pooled_df[TX_col] = 1 
    pooled_df, interaction_cols = add_interatcions_to_pooled(pooled_df, TX_col)
    tmp_X = pooled_df[ps_cols + interaction_cols + [TX_col, 'k']]
    
    res_df['hazard1'] = model.predict_proba(tmp_X)[:,1]
    res_df['one_minus_hazard1'] = 1- res_df['hazard1']
    res_df['survival_TX1'] = res_df.groupby([id_col])['one_minus_hazard1'].cumprod()

    surv_gform_0 = res_df.groupby(['k'])['survival_TX0'].mean()
    surv_gform_1 = res_df.groupby(['k'])['survival_TX1'].mean()
    
    return surv_gform_0, surv_gform_1, res_df['hazard0'], res_df['hazard1']

def plot_survival_from_standardized_sklearn_lr(model, survival_df, TX_col, TX_labels_dict, id_col, event_col='E', 
                                               event_time_col='T', index_time_col='preg_start', 
                                               followup_max_time_from_index=10*12,
                                               preprocessor=None, process_cols=[], ps_cols=[], outcome_name=''):
    """
    Calculate and plot average survival curves from a Logistic Regression model fitted on a pearson-time dataframe with
    standartization
    
    Parameters
    ----------
    model : estimtor Object, fitted on the pearson-time data, with covariates. Implements "predict" method
    survival_df : DataFrame, contains the same covariates the model was trained on, and columns event_col and event_time_col
    TX_col : str, name of the treatment column in data
    TX_labels_dict: dict, keys are TX values, values are lables of the treated and un-treated plots
    id_col: str, name of the id columns in data
    event_col: str, name of the boolean column which indicates whether the outcome even occured or not
    event_time_col: str, name of the time column which indicates when the event occured\end of follow up if the event
                    did not occur
    index_time_col: str, name of the index time columns. Default is 'preg_start'
    followup_max_time_from_index: int, max time allowed from index. e.g. if we want until 10 years old we can 
                                  put 10*DAYS_IN_YEAR
    preprocessor: ColumnTransformer object, implements method "fit_transform". Default is None
    process_cols: list, names of columns to pass to the preprocessor. Default is an empty list
    outcome_name: str, name of the outcome plotted
    """
    
    surv_gform_0, surv_gform_1 = get_survival_curves_from_standartization(model, survival_df, id_col, TX_col, event_col, event_time_col, 
                                                                          followup_max_time_from_index, preprocessor, process_cols, 
                                                                          ps_cols)
    
    min_0 = min(surv_gform_0)
    min_1 = min(surv_gform_1)
    
    min_val = min(min_0, min_1)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(surv_gform_0)
    ax.plot(surv_gform_1)
    ax.set_ylim(min_val - 0.005, 1)
    ax.set_xlabel('Months of follow-up', fontsize=14)
    ax.set_ylabel('Survival probability', fontsize=14)
    ax.set_title('{}\nPooled LR - standardized'.format(outcome_name))
    ax.text(0.85*followup_max_time_from_index, min_0 - 0.001, '{}'.format(TX_labels_dict[0]))
    ax.text(0.85*followup_max_time_from_index, min_1 + 0.001, '{}'.format(TX_labels_dict[1]))    

    
def plot_outcome_pooled_sklearn_lr_standardized(data, outcome_df, outcome_time_col, outcome_name, id_col, index_time_col, 
                                                TX_col, TX_labels_dict, followup_end_time, followup_max_time_from_index, 
                                                ps_df, ps_cols, months_bins, preprocessor, process_cols):
    """
    Fit a pooled sklearn model using standartization and plots the survival curves obtained from it.
    
    Parameters
    ----------
    data : DataFrame, contains the population of interest with its features. 
           Must contain the columns passed in id_col and index_time_col
    outcome_name: str, name of the outcome
    outcome_df : DataFrame, contains the outcome of interest
    outcome_time_col: str, name of the time column of the outcome_df which contains the time of outcome event
    id_col: str, name of the id column in the outcome_df
    index_time_col: str, name of the index time column in the outcome df
    TX_col: str, name of the treatment column in the outcome df
    TX_labels_dict: dict, keys are TX values, values are lables of the treated and un-treated plots
    followup_end_time: int, the time at which adminiastrative censoring is applied. e.g. in Clalit 1/1/2018 = 18*DAYS_IN_YEAR
    followup_max_time_from_index: int, max time allowed from index. e.g. if we want until 10 years old we can put 10*DAYS_IN_YEAR
    ps_df: DataFrame, contains the propensity scores and the columns ps_cols
    ps_cols: list of strings, names of the columns to add to the LR model
    months_bins: int, number of months to take as an interval to create person-time format data
    preprocessor: ColumnTransformer object, implements method "fit_transform". Default is None
    process_cols: list, names of columns to pass to the preprocessor. Default is an empty list
    
    Returns
    ---------
    Doesn't return value, plots the survival plot
    """
    
    pooled_df, survival_df = prepare_pooled_lr_data(data, outcome_df, outcome_time_col, id_col, index_time_col, TX_col,
                                                    followup_end_time, followup_max_time_from_index, months_bins, ps_df, ps_cols)
    if preprocessor is not None:
        pooled_df[process_cols] = pd.DataFrame(preprocessor.fit_transform(pooled_df[process_cols]), columns=process_cols)

    pooled_df, interaction_cols = add_interatcions_to_pooled(pooled_df, TX_col)
    tmp_X = pooled_df[ps_cols + interaction_cols + [TX_col, 'k']]
    tmp_y = pooled_df['Y_k']
    
    fit_model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=5000)
    fit_model.fit(tmp_X, tmp_y) 
    
    followup_max_time_from_index_binned = math.ceil((followup_max_time_from_index*(12/months_bins))/DAYS_IN_YEAR)
    plot_survival_from_standardized_sklearn_lr(model=fit_model, survival_df=survival_df, TX_col=TX_col, TX_labels_dict=TX_labels_dict, 
                                       id_col=id_col, event_col='E', event_time_col='T', index_time_col=index_time_col, 
                                       followup_max_time_from_index=followup_max_time_from_index_binned, 
                                       preprocessor=preprocessor, process_cols=process_cols, ps_cols=ps_cols, outcome_name=outcome_name)
