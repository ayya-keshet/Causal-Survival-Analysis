import pandas as pd

from causallib.estimation import IPW


class OW(IPW):
    """
    Causal model implementing overlap weighing
    w_i = 1 - Pr[A=a_i|X_i]
    """
    
    def __init__(self, learner):
        """

        Args:
            learner: Initialized sklearn model.
        """
        super(OW, self).__init__(learner, False)
        self.truncate_eps = None
    
    def compute_weights(self, X, a, treatment_values=None, truncate_eps=None, use_stabilized=False):
        """
        Compute individual weight given the individual's treatment assignment.
        w_i = 1 - Pr[A=a_i|X_i] for each individual i.

        Args:
            X: (pd.DataFrame): Covariate matrix of size (num_subjects, num_features).
            a: (pd.Series): Treatment assignment of size (num_subjects,).
            treatment_values (None): Not relevant for the OW class and will always be None.
           truncate_eps (None): Not relevant for the OW class and will always be None.
           use_stabilized (bool): Not relevant for the OW class and will always be False.


        Returns:
           pd.Series | pd.DataFrame: A vector of n_samples with a weight for each sample.

        """
        weights_df= pd.DataFrame(columns=['treatment'])
        weights_df['treatment'] = a
        weights_df['probabilities'] = self.compute_propensity(X, a)

        weights_df['weights'] = 0.0
        weights_df.loc[weights_df.treatment == 1, 'weights'] = 1 - weights_df.loc[weights_df.treatment == 1, 'probabilities']
        weights_df.loc[weights_df.treatment == 0, 'weights'] = weights_df.loc[weights_df.treatment == 0, 'probabilities']

        return weights_df['weights']