import numpy as np
import pandas as pd

DAYS_IN_YEAR = 365.25


def create_survival_data(data_df, id_col, index_time_col, outcome_time_col,
                         followup_end_time, followup_max_time_from_index,
                         censoring_time_col=None):
    
    """
    Create survival DataFrame with T, E columns from data_df that contains columns of index_time and outcome_time
    Followup time is used for censoring purposes.
    #TODO: Competing risks 
    
    Parameters
    ----------
    data_df : Pandas DataFrame that contains columns of index_time and outcome_time
    id_col : str, name of id columns
    index_time_col : str, name of the index time column
    outcome_time_col: str, name of the outcome time column
    followup_end_time: int, the time at which adminiastrative censoring is applied. e.g. in Clalit 1/1/2018 = 18*DAYS_IN_YEAR
    followup_max_time_from_index: int, max time allowed from index. e.g. if we want until 10 years old we can put 10*DAYS_IN_YEAR
    censoring_time_col: TODO
    
    Returns
    -------
    survival_df: a Pandas DataFrame with id_col, T, E columns
    
    For testing purposes
    --------------------
    # Lets say followup_end_time=50, followup_max_time_from_index=30
    # person 1: started at t=0 , died at t=20. Should have E=1, T=20
    # person 2: started at t=10 , did not die but censored at t=30. Should have E=0, T=20
    # person 3: started at t=0 , did not die and not censored. Should be cendored via followup_max_time_from_index. Should have E=0, T=30
    # person 4: started at t=40 , did not die and not censored. Should be cendored via followup_end_time. Should have E=0, T=10
    # person 5: started at t=0 , died at t=60. Should be cendored via followup_end_time. Should have E=0, T=50

    data_df = pd.DataFrame(data={'id': [1,2,3,4,5],
                            'index_date': [0,10,20,40,0],
                            'outcome_date':[20,np.nan,np.nan,np.nan, 60],
                            'censoring_date':[np.nan,30,np.nan,np.nan,np.nan]})

    survival_df = create_survival_data(data_df=data_df,
                                        id_col='id',
                                        index_time_col='index_date',
                                        outcome_time_col='outcome_date',
                                        followup_end_time=50,
                                        followup_max_time_from_index=30,
                                        censoring_time_col='censoring_date')
    survival_df.set_index('id', inplace=True)
    survival_df

    # should return following survival_df:   
        E	T
    id		
    4	0	10
    1	1	20
    2	0	20
    3	0	30
    5	0	50
    
    """
    tmp_data_df = data_df.copy()
    # add NaN censoring time col if non is given
    if censoring_time_col is None:
        censoring_time_col = 'censoring_time_col'
        tmp_data_df[censoring_time_col] = np.nan
        
    survival_df = tmp_data_df[[id_col, index_time_col, outcome_time_col, censoring_time_col]].copy()

    survival_df['event_time_from_index'] = (survival_df[outcome_time_col] - survival_df[index_time_col])
    survival_df['censoring_time_from_index'] = (survival_df[censoring_time_col] - survival_df[index_time_col])
    survival_df['followup_end_time_from_index'] = (followup_end_time - survival_df[index_time_col])
    survival_df['earliest_censoring_time_from_index'] = survival_df[['censoring_time_from_index', 'followup_end_time_from_index']].min(axis=1)
                         
    # Seperate into 2 types of people:
    # (1) those who have an event before followup_max_time_from_index, followup_end_time_from_index and censoring_time_from_index
    idx_event = ( (survival_df[outcome_time_col].notna()) &  \
                  (survival_df['event_time_from_index']<=followup_max_time_from_index) & \
                  (survival_df['event_time_from_index']<=survival_df['earliest_censoring_time_from_index']) )
    survival_df.loc[idx_event, 'E'] = 1
    survival_df.loc[idx_event, 'T'] = survival_df['event_time_from_index']
    
    # (2) those who do not
    survival_df.loc[~idx_event, 'E'] = 0
    survival_df.loc[~idx_event, 'T'] = survival_df['earliest_censoring_time_from_index'] 
    
    # make integer dtypes
    survival_df['E'] = survival_df['E'].astype(int)
    survival_df['T'] = survival_df['T'].astype(int)
    
    # sort by T
    survival_df.sort_values(['T'], inplace=True)
    

    return survival_df[[id_col, 'E', 'T']]


def get_bootstrap_dfs(pooled_df, id_col, survival_df=None, ids=None):
    """
    Sample bootstrap pooled df and survival_df
    
    Parameters
    ----------
    pooled_df : Pandas DataFrame of person-time format
    id_col : str, name of id column
    survival_df: Pandas DataFrame of survival form. Defualt is None
    ids: numpy array, ids to take into the bootstrap sample. If None - ids will be sampled randomly with replacement. Default is None.
    
    Returns
    -------
    boot_pooled_df:  bootstrap pooled df
    boot_survival_df: bootstrap survival df, or None if it was not passed
    """
    all_ids = pooled_df[id_col].unique()
    n_samples = len(all_ids)

    # Draw n_samples bootsrtap ids
    boot_ids = np.random.choice(all_ids, n_samples, replace=True) if ids==None else ids
    boot_ids_df = pd.DataFrame(boot_ids, columns=[id_col])
    boot_ids_df.index.rename('Boot_ID', inplace=True)
    boot_ids_df.reset_index(inplace=True)
    
    # Sample pooled_df
    boot_pooled_df = pooled_df.merge(boot_ids_df, on=id_col)
    
    # Sample survival_df
    if survival_df is None:
        boot_survival_df = None
    else:
        boot_survival_df = survival_df.merge(boot_ids_df, on=id_col)
    
    return boot_pooled_df, boot_survival_df


def make_pooled_survival_data(df, id_col, event_col='E', event_time_col='T'):
    
    """
    Make an "expanded" pooled df with all columns as inputed df and event_col, event_time_col replaced with k and Y_k cols
    
    Parameters
    ----------
    df : Pandas DataFrame that contains columns of id_col, event_col, event_time_col
    id_col : str, name of id column
    event_col : str, name of the event column (that contains 0 or 1) 
    event_time_col: str, name of the event time column
    
    Returns
    -------
    df: a Pandas DataFrame that expands the original df. Contains all columns as inputed df and event_col, event_time_col replaced with k
        (column of time stamp) and Y_k (indicates whether event occured until time k)
        
    For testing purposes
    --------------------
    example_df = pd.DataFrame(data={'id': [1,2,3],
                                'L0': [0,1,1],
                                'T':[2,2,3],
                                'E':[1,0,0]})

    make_pooled_survival_data(example_df, id_col='id', event_col='E', event_time_col='T')
    
    # should return:
        id	L0	k  Y_k
    0	1	0	1	0
    1	1	0	2	1
    2	2	1	1	0
    3	2	1	2	0
    4	3	1	1	0
    5	3	1	2	0
    6	3	1	3	0
    """
    
    # repeat rows depending on last Time
    df = df.reindex(df.index.repeat(df[event_time_col]))
    df.reset_index(drop=True, inplace=True)

    # make timestep column in units of k
    df['k'] = 1
    df['k'] = df.groupby(id_col)['k'].transform(lambda x: x.cumsum())

    # make Y_k column
    df['Y_k'] = 0
    df.loc[df.groupby(id_col)['Y_k'].tail(1).index, 'Y_k'] = df.loc[df.groupby(id_col)['Y_k'].tail(1).index, event_col] 
    
    df.drop([event_col, event_time_col], axis=1, inplace=True)

#     print(df.k.max())
    return df


def days2intervals(s, months_bins):
    return((s//(30.4*months_bins)).astype(int))


def prepare_pooled_lr_data(data, outcome, outcome_time_col, id_col, index_time_col, TX_col,
                          followup_end_time, followup_max_time_from_index, months_bins, ps_df, ps_cols):
    outcome_df = outcome.merge(data[[id_col, index_time_col]], on=id_col)
    survival_data = create_survival_data(outcome_df, id_col, index_time_col, outcome_time_col,
                                         followup_end_time, followup_max_time_from_index)
    survival_data = survival_data.merge(data[[id_col, TX_col]], on=id_col)
    survival_data['T'] = days2intervals(survival_data['T'], months_bins)
    survival_data = survival_data[['T', 'E', id_col, TX_col]].merge(ps_df[ps_cols + [id_col]], on=id_col, how='left')

    pooled_data = make_pooled_survival_data(survival_data, id_col)
    
    return pooled_data, survival_data