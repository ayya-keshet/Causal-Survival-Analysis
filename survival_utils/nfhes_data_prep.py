import numpy as np
import pandas as pd

from propensity_utils import calc_propensity, logit_ip_weights


def prepare_nhfes_surv_data(data_path):
    nhefs_all = pd.read_csv(data_path).reset_index().rename(columns={'index':'pid'})
    
    ## Add interactions, dummies, propensity score and weights to the full data
    for col in ['age', 'wt71', 'smokeintensity', 'smokeyrs']:
        nhefs_all['{}_sqr'.format(col)] = nhefs_all[col] * nhefs_all[col]

    nhefs_all['one'] = 1
    edu_dummies = pd.get_dummies(nhefs_all.education, prefix='edu')
    exercise_dummies = pd.get_dummies(nhefs_all.exercise, prefix='exercise')
    active_dummies = pd.get_dummies(nhefs_all.active, prefix='active')

    nhefs_all = pd.concat(
        [nhefs_all, edu_dummies, exercise_dummies, active_dummies],
        axis=1
    )

    X = nhefs_all[['one', 'sex', 'race', 'edu_2', 'edu_3', 'edu_4', 'edu_5', 'exercise_1', 'exercise_2', 'active_1', 'active_2',
                   'age', 'age_sqr', 'wt71', 'wt71_sqr','smokeintensity', 'smokeintensity_sqr', 'smokeyrs', 'smokeyrs_sqr']]

    nhefs_all['pscore'] = calc_propensity(nhefs_all.qsmk, X)
    ip_denom = logit_ip_weights(nhefs_all.qsmk, X)

    pr_qsmk = nhefs_all.qsmk.mean()

    ip_numer = np.zeros(ip_denom.shape[0])
    ip_numer[nhefs_all.qsmk == 0] = 1 - pr_qsmk
    ip_numer[nhefs_all.qsmk == 1] = pr_qsmk

    ip_weights = ip_numer / ip_denom
    nhefs_all['IPW'] = ip_weights

    return nhefs_all