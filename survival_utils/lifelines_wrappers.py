import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
                      title='', xlabel='', ax=None, q=None, SZ=14, weights=None, xticks=None):
    
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
    for cat, label in categories:
        idx = (survival_df[cat_col] == cat)
        label = '{} ({})'.format(label, len(survival_df[idx]))
        cat_kmf = get_kmf(survival_df, idx, label, weights)
        kmfs.append(cat_kmf)
        cat_kmf.plot(ax=ax)
        
        ax.set_title(title, size=SZ+2)
        ax.set_xlabel(xlabel, size=SZ)
        ax.set_ylabel('Survival Probability', size=SZ)
        
    if xticks is None:
        add_at_risk_counts(*kmfs,ax=ax)
    else:
        ax.set_xticks(xticks);
        add_at_risk_counts(*kmfs,ax=ax)