import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from survival_preprocessing import (create_survival_data, make_pooled_survival_data, prepare_pooled_lr_data,
                                                   get_bootstrap_dfs)
from survival_pooled_model import fit_pooled_lr, get_survival_curves_from_lr
from survival_standardization import get_survival_curves_from_standartization, fit_std_lr
from survival_plots import _prettify_survival_plot

DAYS_IN_YEAR = 365.25

class Survival_Analysis:
    """
    Survival analysis class. 
    
    Parameters
    ----------
    data: DataFrame, contains the population of interest with its featuers. Must contain the columns passed in id_col and index_time_col
    outcome: DataFrame, contains the outcome of interest
    outcome_time_col: str, name of the time column of the outcome_df which contains the time of outcome event
    outcome_name: str, name of the outcome of interest
    id_col: str, name of the id column in the outcome DataFrame
    index_time_col: str, name of the index time column in the data DataFrame
    TX_col: str, name of the treatment column in the data DataFrame
    TX_labels: dictionary, contains the lables for the different treatment values. For example {0:'Vaginal', 1:'CS'}
    TX_colors: dictionary, contatins the colors for the different treatment values for plots. For example {0:'royalblue', 1:'darkorange'}
    followup_end_time: int, the time at which adminiastrative censoring is applied. e.g. in Clalit 1/1/2018 = 18*DAYS_IN_YEAR
    followup_max_time_from_index: int, max time allowed from index. e.g. if we want until 10 years old we can put 10*DAYS_IN_YEAR
    months_bins: int, number of months to take as an interval to create person-time format data
    ps_df: DataFrame, contains the propensity scores, columns used to train the propensity score model and weights column
    pscore_col: str, name of the propensity score column in ps_df
    ps_cols: list of strings, names of the columns to add to the LR model, used to train the propensity model
    weights: str, name of the weights in ps_df, one of the following: 'IPW', 'OW'
    n_bootstraps: int, number of bootstraps iteration to perform. If None no bootsrapping will be performed
    output_dir: str, path to an output directory to which results are saved. If None output will not be saved
    
    """
    
    def __init__(self, data, outcome, outcome_time_col, outcome_name, id_col, index_time_col, TX_col, TX_labels, TX_colors,
                 followup_end_time, followup_max_time_from_index, months_bins, ps_df, pscore_col, ps_cols, weights=None, 
                 n_bootstraps=None, output_dir=None):
        # Public 
        self.survival_curve0 = None
        self.survival_curve1 = None
        self.survival_df0 = pd.DataFrame()
        self.survival_df1 = pd.DataFrame()
        self.survival_data = pd.DataFrame()
        self.pooled_data = pd.DataFrame()
        
        # Protected
        self._data = data
        self._outcome = outcome
        self._outcome_time_col = outcome_time_col
        self._competing_risks_time_cols = []
        self._competing_risks_actions = []
        self._outcome_name = outcome_name
        self._id_col = id_col
        self._index_time_col = index_time_col
        self._TX_col = TX_col
        self._TX_labels = TX_labels
        self._TX_colors = TX_colors
        self._followup_end_time = followup_end_time
        self._followup_max_time_from_index = followup_max_time_from_index
        self._months_bins = months_bins
        self._ps_df = ps_df
        self._pscore_col = pscore_col
        self._ps_cols = ps_cols
        self._weights = weights
        self._n_bootstraps = n_bootstraps
        self._output_dir = output_dir

    def _days2intervals(self, s):
        return((s//(30.4*self._months_bins)).astype(int))
    
    def create_pooled_data_from_outcome(self):
        outcome_df = self._outcome.merge(self._data[[self._id_col, self._index_time_col]], on=self._id_col)
        self.survival_data = create_survival_data(outcome_df, self._id_col, self._index_time_col, self._outcome_time_col,
                                                  self._followup_end_time, self._followup_max_time_from_index)
        self.survival_data = self.survival_data.merge(self._data[[self._id_col, self._TX_col]], on=self._id_col)
        self.survival_data['T'] = self._days2intervals(self.survival_data['T'])
        self.survival_data = \
        self.survival_data[['T', 'E', self._id_col, self._TX_col]].merge(self._ps_df[self._ps_cols + 
                                                                                     [self._id_col]], on=self._id_col, 
                                                                         how='left')

        self.pooled_data = make_pooled_survival_data(self.survival_data, self._id_col)
        
    def _estimate_survival_lr(self):
        fit_model = fit_pooled_lr(self.pooled_data, self._formula)
        return get_survival_curves_from_lr(fit_model, self._TX_col, self._TX_labels, self._followup_max_time_from_index_binned)
        
    def _estimate_survival_lr_weighted(self):
        pooled_df = self.pooled_data.merge(self._ps_df[[self._id_col, self._weights]], on=self._id_col, how='left')
        fit_model = fit_pooled_lr(pooled_df, self._formula, self._weights) 
        return get_survival_curves_from_lr(fit_model, self._TX_col, self._TX_labels, self._followup_max_time_from_index_binned)
    
    def _estimate_survival_std(self, model, preprocessor, process_cols):
        if preprocessor is not None:
                self.pooled_data[process_cols] = pd.DataFrame(preprocessor.fit_transform(self.pooled_data[process_cols]), 
                                                              columns=process_cols)
        model = fit_std_lr(self.pooled_data, model, self._TX_col, self._ps_cols)

        return get_survival_curves_from_standartization(model, self.survival_data, self._id_col, self._TX_col, 
                                                        'E', 'T', self._followup_max_time_from_index_binned, 
                                                        preprocessor, process_cols, self._ps_cols)
    
    def _estimate_survival_lr_bootstrap(self):
        pooled_df = self.pooled_data if self._method=='LR' else self.pooled_data.merge(self._ps_df[[self._id_col, self._weights]], 
                                                                                 on=self._id_col, how='left')
        weights = None if self._method=='LR' else self._weights
        
        surv_curves_0_list = []
        surv_curves_1_list = []
        for i in tqdm(range(self._n_bootstraps)):
            boot_pooled_df, _ = get_bootstrap_dfs(pooled_df, self._id_col)
            fit_model = fit_pooled_lr(boot_pooled_df, self._formula, weights) 
            model_surv_0, model_surv_1 = get_survival_curves_from_lr(fit_model, self._TX_col, self._TX_labels, 
                                                                     self._followup_max_time_from_index_binned)
            surv_curves_0_list.append(model_surv_0)
            surv_curves_1_list.append(model_surv_1)
            
        self.survival_df0 = pd.DataFrame(surv_curves_0_list)
        self.survival_df1 = pd.DataFrame(surv_curves_1_list)
        self.survival_curve0 = self.survival_df0.quantile(0.5)
        self.survival_curve1 = self.survival_df1.quantile(0.5)
        
    def _estimate_survival_std_bootstrap(self, model, preprocessor, process_cols):
        if preprocessor is not None:
            self.pooled_data[process_cols] = pd.DataFrame(preprocessor.fit_transform(self.pooled_data[process_cols]), 
                                                          columns=process_cols)
        surv_curves_0_list = []
        surv_curves_1_list = []    
        for i in tqdm(range(self._n_bootstraps)):
            boot_pooled_df, boot_survival_df = get_bootstrap_dfs(self.pooled_data, self._id_col, self.survival_data)
            model = fit_std_lr(boot_pooled_df, model, self._TX_col, self._ps_cols)
            model_surv_0, model_surv_1 = \
            get_survival_curves_from_standartization(model, boot_survival_df, 'Boot_ID', self._TX_col, 
                                                     'E', 'T', self._followup_max_time_from_index_binned, 
                                                     preprocessor, process_cols, self._ps_cols)
            surv_curves_0_list.append(model_surv_0)
            surv_curves_1_list.append(model_surv_1)
            
        self.survival_df0 = pd.DataFrame(surv_curves_0_list)
        self.survival_df0.reset_index(inplace=True, drop=True)
        self.survival_df1 = pd.DataFrame(surv_curves_1_list)
        self.survival_df1.reset_index(inplace=True, drop=True)
        self.survival_curve0 = self.survival_df0.quantile(0.5)
        self.survival_curve1 = self.survival_df1.quantile(0.5)
    
    def estimate_survival(self, method, model=None, preprocessor=None, process_cols=[]):
        if method not in ['LR', 'IPW', 'OW', 'STANDARDIZE']:
            print('ERROR! Passed method unrecognized')
            return
        
        self._method = method
        self.create_pooled_data_from_outcome()
        self._followup_max_time_from_index_binned = int(np.ceil((self._followup_max_time_from_index*(12/self._months_bins))/DAYS_IN_YEAR))
        self._formula = 'Y_k ~ {} + {}:k + {}:np.power(k, 2) + k + np.power(k, 2)'.format(self._TX_col, self._TX_col, self._TX_col)
        
        if self._n_bootstraps is None:
            if self._method == 'LR':
                self.survival_curve0, self.survival_curve1 = self._estimate_survival_lr()
            elif self._method in ['IPW', 'OW']:
                self.survival_curve0, self.survival_curve1 = self._estimate_survival_lr_weighted()
            elif self._method == 'STANDARDIZE':
                self.survival_curve0, self.survival_curve1 = self._estimate_survival_std(model, preprocessor, process_cols)
        else:
            if self._method in ['LR', 'IPW', 'OW']:
                self._estimate_survival_lr_bootstrap()
            else:
                self._estimate_survival_std_bootstrap(model, preprocessor, process_cols)

    def plot_survival(self):
        followup_max_time_from_index_binned = \
        np.ceil((self._followup_max_time_from_index*(12/self._months_bins))/DAYS_IN_YEAR)
        
        min_0 = min(self.survival_curve0)
        min_1 = min(self.survival_curve1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.survival_curve0)
        ax.plot(self.survival_curve1)
        
        if self._n_bootstraps is not None:
            ax.fill_between(x=self.survival_df0.columns.values, color=self._TX_colors[0], alpha=0.2,
                            y1=self.survival_df0.quantile(0.025), y2=self.survival_df0.quantile(0.975))
            ax.fill_between(x=self.survival_df1.columns.values, color=self._TX_colors[1], alpha=0.2,
                            y1=self.survival_df1.quantile(0.025), y2=self.survival_df1.quantile(0.975))
        
        weights = '' if self._method=='LR' else self._weights if self._method in ['IPW', 'OW'] else 'Standardized'
        _prettify_survival_plot(ax, min_0, min_1, followup_max_time_from_index_binned, self._TX_labels, 
                                self._outcome_name, weights)
        if self._output_dir is not None:
            fig.savefig(os.path.join(self._output_dir, 'survival_plot_{}_{}_{}'.format(self._method, self._n_bootstraps, self._outcome_name)))
            self.survival_df0.to_csv(os.path.join(self._output_dir, 'survival_df0_{}_{}_{}.csv'.format(self._method, self._n_bootstraps,
                                                                                                       self._outcome_name)))
            self.survival_df1.to_csv(os.path.join(self._output_dir, 'survival_df1_{}_{}_{}.csv'.format(self._method, self._n_bootstraps,
                                                                                                       self._outcome_name)))
        
    def survival_difference(self, time_after_index):
        survival_diffs = self.survival_df0[time_after_index] - self.survival_df1[time_after_index]
        diff_median = np.median(survival_diffs)*100
        diff_low, diff_high = np.percentile(survival_diffs, q=[2.5, 97.5])*100
        difference_txt = 'Difference in survival probability at time {}: {:>0.1f}% ({:>0.1f}, {:>0.1f})'.format(time_after_index,
                                                                                                                diff_median,
                                                                                                                diff_low, diff_high)
        if self._output_dir is not None:
            print(difference_txt, file=open(os.path.join(self._output_dir, 'survival_diff_{}_{}_{}.txt'.format(self._method, self._n_bootstraps,
                                                                                                               self._outcome_name)),
                                            'w'))
        print(difference_txt)