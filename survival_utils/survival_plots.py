import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from survival_pooled_model import get_survival_curves_from_lr, fit_pooled_lr
from survival_preprocessing import prepare_pooled_lr_data, DAYS_IN_YEAR# , get_bootstrap_pooled_df


def _prettify_survival_plot(ax, min_0, min_1, followup_max_time_from_index, TX_labels_dict, outcome_name, weights=None):
    """
    Adjust survival plots for labels and limits
    
    Parameters
    ----------
    XXX
    """
    min_val = min(min_0, min_1)
    ax.set_ylim(min_val - 0.005, 1)
    ax.text(0.85*followup_max_time_from_index, min_0 - 0.001, '{}'.format(TX_labels_dict[0]))
    ax.text(0.85*followup_max_time_from_index, min_1 + 0.001, '{}'.format(TX_labels_dict[1]))
    ax.set_xlabel('Months of follow-up', fontsize=14)
    ax.set_ylabel('Survival probability', fontsize=14)
    ax.set_title('{}\nPooled LR {}'.format(outcome_name, weights))
        

def plot_survival_from_lr(model, TX_col, TX_labels_dict, followup_max_time_from_index, outcome_name, weights=None, ax=None):
    """
    Plot binary treatment survival curves from a pooled Logistic Regression model (without standardization)
    
    Parameters
    ----------
    model : estimtor Object, fitted on a pearson-time data
    TX_col : str, name of the treatment column in data
    TX_labels_dict: dict, keys are TX values, values are lables of the treated and un-treated plots
    followup_max_time_from_index: int, max follow up time from index. e.g. if we want until 10 years old we can put 10*DAYS_IN_YEAR 
    outcome_name: str, name of the outcome plotted
    weights: str, name of the weights used, or None if no weighing
    """
    model_surv_0, model_surv_1 = get_survival_curves_from_lr(model, TX_col, TX_labels_dict, followup_max_time_from_index)
    
    min_0 = min(model_surv_0)
    min_1 = min(model_surv_1)
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(model_surv_0)
    ax.plot(model_surv_1)
    _prettify_survival_plot(ax, min_0, min_1, followup_max_time_from_index, TX_labels_dict, outcome_name, weights)

    
# def plot_bootstrap_survival(pooled_df, formula, id_col, TX_col, TX_labels_dict, TX_colors, followup_max_time_from_index,
#                             outcome_name,  weights=None, ax=None, n_bootstraps=10):
#     """
#     XXX
    
#     Parameters
#     ----------
#     XXX
#     """        
#     surv_curves_0_list = []
#     surv_curves_1_list = []
    
#     for i in tqdm(range(n_bootstraps)):
#         boot_pooled_df = get_bootstrap_pooled_df(pooled_df, id_col)
#         fit_model = fit_pooled_lr(boot_pooled_df, formula) 
#         model_surv_0, model_surv_1 = get_survival_curves_from_lr(fit_model, TX_col, TX_labels_dict, followup_max_time_from_index)
#         surv_curves_0_list.append(model_surv_0)
#         surv_curves_1_list.append(model_surv_1)
    
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))

#     surv_curves_A0 = pd.DataFrame(surv_curves_0_list)
#     surv_curves_A1 = pd.DataFrame(surv_curves_1_list)

#     def _plot_bootstrap_survival_curve(surv_curves, color):
#         ax.plot(surv_curves.columns.values, surv_curves.quantile(0.5), color=color)
#         ax.fill_between(x=surv_curves.columns.values, y1=surv_curves.quantile(0.025), y2=surv_curves.quantile(0.975),
#                         color=color, alpha=0.2)

#     _plot_bootstrap_survival_curve(surv_curves_A0, color=TX_colors[0])
#     _plot_bootstrap_survival_curve(surv_curves_A1, color=TX_colors[1])

#     min_0 = surv_curves_A0.min().min()
#     min_1 = surv_curves_A1.min().min()
#     _prettify_survival_plot(ax, min_0, min_1, followup_max_time_from_index, TX_labels_dict, outcome_name, weights)
    

def plot_outcome_pooled_lr(data, outcome_df, outcome_time_col, outcome_name, id_col, index_time_col, TX_col, TX_labels_dict, TX_colors,
                           followup_end_time, followup_max_time_from_index, ps_df, pscore_col, ps_cols,
                           months_bins, weights_col=None, weights=None, n_bootstraps=10):
    """
    Fit a pooled LR model and plots the survival curves obtained from it.

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
    ps_df: DataFrame, contains the propensity scores, columns ps_cols and weights_col if not None
    pscore_col: str, name of the propensity score columns in ps_df
    ps_cols: list of strings, names of the columns to add to the LR model
    months_bins: int, number of months to take as an interval to create person-time format data
    weights_col: str, name of the weights columns in ps_df. In None there will be no weighing
    weights: str, name of the weights used, or None if no weighing
    n_bootstraps: number of bootstraps for 95% confidence intervals

    Returns
    ---------
    Doesn't return value, plots the survival plot
    """
    pooled_df, _ = prepare_pooled_lr_data(data, outcome_df, outcome_time_col, id_col, index_time_col, TX_col,
                                          followup_end_time, followup_max_time_from_index, months_bins, ps_df, ps_cols)
    formula = 'Y_k ~ {} + {}:k + {}:np.power(k, 2) + k + np.power(k, 2)'.format(TX_col, TX_col, TX_col)
    fit_model = fit_pooled_lr(pooled_df, formula) 
    followup_max_time_from_index_binned = math.ceil((followup_max_time_from_index*(12/months_bins))/DAYS_IN_YEAR)
    
    
    if n_bootstraps is None:
        plot_survival_from_lr(fit_model, TX_col, TX_labels_dict, followup_max_time_from_index_binned, outcome_name, weights=None)
    else:
        plot_bootstrap_survival(pooled_df, formula, id_col, TX_col, TX_labels_dict, TX_colors, followup_max_time_from_index_binned,
                                outcome_name, weights=None, n_bootstraps=n_bootstraps)
    
    # also plot IPW if needed
    if weights_col is not None:
        pooled_df = pooled_df.merge(ps_df[[id_col, weights_col]], on=id_col, how='left')
        if n_bootstraps is None:
            fit_model = fit_pooled_lr(pooled_df, formula, weights_col) 
            plot_survival_from_lr(fit_model, TX_col, TX_labels_dict, followup_max_time_from_index_binned, outcome_name, weights)
        else:
            plot_bootstrap_survival(pooled_df, formula, id_col, TX_col, TX_labels_dict, TX_colors, followup_max_time_from_index_binned,
                                    outcome_name,  weights=weights, n_bootstraps=n_bootstraps)