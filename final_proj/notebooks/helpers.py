from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np

def cleanly_read_parquet(save_loc, names=None):
    '''
    The parquets we saved are in a weird format -- retrieve the actual PCA factors.
    '''
    df = pd.read_parquet(save_loc)

    pca_arr = []
    for row in df.values:
        pca_arr.append(row[0]['values'])

    if not names:
        names = [f'pc{i}' for i in range(len(row[0]['values']))]

    df = pd.DataFrame(pca_arr, columns=names)
    
    return df

def pick_relevant_features(data: pd.DataFrame, treatment_var: str, 
                                add_to_drop: list=[], max_iter=10000) -> pd.DataFrame:
    '''
    Use a logistic regression to find the 10 most important features indicating
    that a track received treatment (treatment_var = 1) and the 10 other
    most important indicating that it did not (treatment_var = -1)

    Inputs:
      data (pd.DataFrame): The dataset, containing treatment_var and all columns
        in add_to_drop. It is assumed that all values in the dataset are floats.
      treatment_var (str): The name of the treatment var. Should only
        contain values that are -1 or 1.
      add_to_drop (list[str]): list of additional columns to drop when fitting
        the binary model.

    Returns: pd.Dataframe whose one columns is coefficients and whose index
      are the relevant feature names.
    '''

    # Let's do some feature checking!!
    data = data.copy().dropna(subset=[treatment_var])
    data[treatment_var] = data[treatment_var].astype(int)

    # Train model
    lg = LogisticRegressionCV(max_iter=max_iter)
    lg.fit(data.drop([treatment_var] + add_to_drop, axis=1), data[treatment_var])

    # Coeffs!!
    coefs = pd.DataFrame(lg.coef_[0], index=data.drop([treatment_var] + add_to_drop, axis=1).columns)

    # Select Top 10 most positive + most negative features
    coefs = coefs.sort_values(0, ascending=False)
    coefs = coefs.iloc[[i for i in range(-10, 10)], :]
    return coefs

def match_treatment_control(treatment: pd.DataFrame, 
               control: pd.DataFrame) -> tuple[pd.Series]:
    '''
    Find best matches and match distance in control group for treatment group variables.
    '''
    sim = cosine_similarity(treatment, control)
    best_matches = np.argmax(sim, axis=1)
    match_values = np.max(sim, axis=1)

    return best_matches, match_values

def match_rows(data: pd.DataFrame, features: list[str], 
               treatment_var: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Given a DataFrame of data, the features to match on, and the relevant treatment
    variable, return a DataFrame containing matches between treatment variables 
    (treatment_var=1) and control variables (treatment_var=-1) and their similarity
    metrics. Uses cosine_distance.

    Inputs:
      data (pd.Dataframe): DataFrame containing the relevant data. Should have a
        column for all features in features and treatment_var
      features (list[str]): list of strings of the relevant features.
      treatment_var (str): name of treatment variable column. Should be +1 or -1 
        everywhere.

    Returns: tuple[pd.DataFrame, pd.DataFrame]. First DataFrame is the treatment 
      variables from data, with two additional features: 'matches', the index of 
      the best match in the control, and 'similarity', or the similarity for 
      those two matches. Second DataFrame is the control data.  
    '''
    matching = data.loc[:, features]
    treatment = matching.copy()[data[treatment_var] == 1] # A and B grade tracts
    control = matching.copy()[data[treatment_var] == -1] # C and D grade tracts

    metrics, values = match_treatment_control(treatment, control)

    treatment['control_match_index'] = metrics
    treatment['similarity'] = values

    return treatment, control


def assign_race_grade_treatment(row):
    '''
    Assigns tracks to two groups: one if they were A, B graded and held no Black population,
    and another if they were C, D graded and had reported Black population.

    All other tracts are not considered.
    '''
    
    if row.grade in ['A', 'B'] and row.nyn_agg == '0':
        return 1
    
    if row.grade in ['C', 'D'] and row.nyn_agg == '1':
        return -1

    return pd.NA

def assign_cd_white(row):
    '''
    Returns -1 for C/D lined tracts which had reported no Black presence,
    and 1 for C/D lined tracts which had reported black presence.
    '''
    
    if row.grade in ['A', 'B']:
        return pd.NA

    if row.nyn_agg == '1':
        return -1
    
    if row.nyn_agg == '0':
        return 1

    raise ValueError(f'Encountered poorly labeled row {row}')
    

def do_all_matching(data: pd.DataFrame, to_exclude: list=[], max_iter: int=10000,
                    save_loc: str=None) -> tuple[tuple[pd.DataFrame, pd.DataFrame], 
                               tuple[pd.DataFrame, pd.DataFrame], 
                               tuple[pd.DataFrame, pd.DataFrame]]:
    '''
    Find similarity scores, display a graph of similarity scores, and return all
    relevant treatment/control matching groups. 
    '''
    # Retrieve features, get their names
    coefs_grade = pick_relevant_features(data.drop(to_exclude, axis=1), 'treatment_grade', add_to_drop=['treatment_grade_race', 'cd_white'], max_iter=max_iter)
    coefs_grade_race = pick_relevant_features(data.drop(to_exclude, axis=1), 'treatment_grade_race', add_to_drop=['treatment_grade', 'cd_white'], max_iter=max_iter)
    coefs_cd = pick_relevant_features(data.drop(to_exclude, axis=1), 'cd_white', add_to_drop=['treatment_grade', 'treatment_grade_race'], max_iter=max_iter)

    features_grade = coefs_grade.index
    features_grade_race = coefs_grade_race.index
    features_cd = coefs_cd.index

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    fig.set_tight_layout(True)

    treatment_grade, control_grade = match_rows(data, features_grade, 'treatment_grade')
    treatment_grade_race, control_grade_race = match_rows(data, features_grade_race, 'treatment_grade_race')
    treatment_cd, control_cd = match_rows(data, features_cd, 'cd_white')

    sns.kdeplot(treatment_grade.similarity, 
                color='r', fill=True, ax=axes[0],
                common_norm=True)
    sns.kdeplot(treatment_grade_race.similarity, 
                color='orange', fill=True, ax=axes[2],
                common_norm=True)
    sns.kdeplot(treatment_cd.similarity, 
                color='y', fill=True, ax=axes[1],
                common_norm=True)

    for ax in axes:
        ax.set_xlim([0, 1])

    axes[0].set_title('AB Graded vs CD Graded')
    axes[2].set_title('AB Graded Non-Black vs CD Graded Black')
    axes[1].set_title('CD Graded Non-Black vs CD Graded Black')

    axes[0].set_xlabel('Cosine Similarity of \nAB Tracts to CD Tracts')
    axes[2].set_xlabel('Cosine Similarity of \nAB Non-Black Tracts to CD Black Tracts')
    axes[1].set_xlabel('Cosine Similarity of \nAB Non-Black Tracts to CD Black Tracts')

    
    if save_loc:
        plt.savefig(save_loc)

    plt.show()

    return (treatment_grade, control_grade), (treatment_grade_race, control_grade_race), (treatment_cd, control_cd)

def print_return_dict(rd: dict={}):
    '''
    Given a dictionary mapping variable names to a list of relevant stuff,
    print the list in a nice format.
    '''
    to_print = ""
    for var in rd.keys():
        internal = rd[var]
        to_print = to_print + \
        f"""----------------------------------------------
        {var}:
        \tTreatment:
        \t  Mean Treatment: {internal['mean_treat']:.2f}
        \t  Median Treatment: {internal['median_treat']:.2f},
        \t  STD Treatment: {internal['std_treat']:.2f}
        \t  n treatment: {internal['n_treat']:.2f}
        \tControl:
        \t  Mean Control: {internal['mean_control']:.2f}
        \t  Median Control: {internal['median_control']:.2f},
        \t  STD Control: {internal['std_control']:.2f}
        \t  n control: {internal['n_control']:.2f}
        \tDifferences:
        \t  Mean Difference: {internal['mean_diff']:.2f}
        \t  Median Difference: {internal['mean_diff']:.2f}
        ----------------------------------------------
        """
    print(to_print)

def make_comparisons(treatment: pd.DataFrame, control: pd.DataFrame, 
                        cutoff: float, nyc_data, comparison_vars=['numCrashes'],
                        return_dictionary: bool=False) -> tuple[float, float]:
    '''
    Take treatment and control dataframes and return the difference in car crashes between
    matched tracts with similarity greater than the cutoff.
    '''
    treatment = treatment[treatment.similarity > cutoff]
    control_obs = control.iloc[treatment[treatment.similarity > cutoff].control_match_index, :].index

    print(f'treatment shape: {treatment.shape}')
    print(f'treatment index shape: {treatment[treatment.similarity > cutoff].control_match_index.shape}')
    print(f'control_obs shape {control_obs.shape}')

    return_dict = {}
    for var in comparison_vars:
        # Calculate info for crashes
        
        mean_treat = nyc_data.loc[treatment.index, :][var].mean()
        mean_control = nyc_data.loc[control_obs, :][var].mean() 
        
        med_treat = nyc_data.loc[treatment.index, :][var].median()
        med_control = nyc_data.loc[control_obs, :][var].median() 
        
        mean_diff   = mean_control - mean_treat
        median_diff = med_control - med_treat

        std_treat = nyc_data.loc[treatment.index, :][var].std()
        std_control = nyc_data.loc[control_obs, :][var].std()

        n_treat, n_control = treatment.shape[0], control_obs.shape[0]

        return_dict[var] = {
            'mean_treat': mean_treat,
            'mean_control': mean_control,
            'median_treat': med_treat,
            'median_control': med_control,
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'std_treat': std_treat,
            'std_control': std_control,
            'n_treat': n_treat,
            'n_control': n_control
        }

    print_return_dict(return_dict)

    # highly specific use case lol
    if 'injCrashes' in comparison_vars and 'numCrashes' in comparison_vars:
        calc_and_print_odds_ratios(treatment.index, control_obs, nyc_data, 
                                   'injCrashes', 'numCrashes')

    if 'deathsCrashes' in comparison_vars and 'numCrashes' in comparison_vars:
        calc_and_print_odds_ratios(treatment.index, control_obs, nyc_data, 
                                   'deathsCrashes', 'numCrashes')

    if return_dictionary:
        return return_dict
    
    return None

def calc_and_print_odds_ratios(treatment: pd.Series, control: pd.Series, 
                               data: pd.DataFrame, numerator: str, 
                               denominator: str) -> None:
    '''
    Given the treatment indices, control indices, data, treatment col name,
    and denominator col name, print the odds ratios for the crash.
    '''
    treat_odds = data.loc[treatment][numerator].sum() / data.loc[treatment][denominator].sum()
    control_odds = data.loc[control][numerator].sum() / data.loc[control][denominator].sum()
        
    to_print = \
    f"""
    Odds ratios (Treatment Odds / Control Odds) for {numerator} and {denominator}:
    \t{(treat_odds) / (control_odds):.2f}        
    """

    print(to_print)