import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from functools import reduce
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts


def get_kmf(survival_df, idx=None, label=None, weights=None):
    kmf = KaplanMeierFitter()
    if idx is not None:
        survival_df = survival_df[idx]
    if weights is not None:
        weights = weights[idx]

    kmf = kmf.fit(survival_df['T'], survival_df['E'], label=label, weights=weights)
    return kmf


def plot_category_kmf(survival_df, cat_col, cat_list=None, labels=None,
                      title='', xlabel='', ax=None, q=None, SZ=14, weights=None, xticks=None, return_data=False):
    if (q is not None):
        survival_df[cat_col + '_cat'] = pd.qcut(survival_df[cat_col], q, precision=1)
        cat_col = cat_col + '_cat'
        cat_list = survival_df[cat_col].cat.categories
        if labels is None:
            labels=[str(x) for x in cat_list]
        categories = list(zip(cat_list, labels))
    else:
        if labels is not None:
            categories = list(zip(cat_list, labels))
        else:
            categories = list(zip(cat_list, cat_list.astype(str)))
    
    if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,5))
    
    kmfs = []
    survival_plot_dfs = []
    survival_ci_dfs = []
    for cat, label in categories:
        idx = (survival_df[cat_col] == cat)
        #label = '{} (N={})'.format(label, len(survival_df[idx]))
        cat_kmf = get_kmf(survival_df, idx, label, weights)
        kmfs.append(cat_kmf)
        cat_kmf.plot(ax=ax)
        if return_data:
            survival_plot_dfs.append(cat_kmf.survival_function_)
            survival_ci_dfs.append(cat_kmf.confidence_interval_survival_function_)
        
        ax.set_title(title, size=SZ+2)
        ax.set_xlabel(xlabel, size=SZ)
        ax.set_ylabel('Survival Probability', size=SZ)
        
    if xticks is None:
        add_at_risk_counts(*kmfs,ax=ax)
    else:
        ax.set_xticks(xticks);
        add_at_risk_counts(*kmfs,ax=ax)
    
    if return_data:
        survival_plot_df = reduce(lambda df1, df2: pd.merge(df1, df2, on='timeline'), survival_plot_dfs)
        return survival_plot_df, survival_ci_dfs

    return None, None