import warnings

import numpy as np
import pandas as pd
from nilearn.glm import expression_to_contrast_vector
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import spm_hrf, spm_time_derivative


def make_stick_function(
    onsets: list[float], durations: list[float], length: float, resolution: float = 0.2
) -> pd.DataFrame:
    """
    Create a stick function with onsets and durations

    Parameters
    ----------
    onsets : list
        List of onset times
    durations : list
        List of duration times
    length : float
        Length of the stick function (in seconds)
    resolution : float
        Resolution of the stick function (in seconds)
        0.2 secs by default

    Returns
    -------
    sf : np.array
        Timepoints of the stick function
    """
    timepoints = np.arange(0, length, resolution)
    sf = np.zeros_like(timepoints)
    for onset, duration in zip(onsets, durations):
        sf[(timepoints >= onset) & (timepoints < onset + duration)] = 1
    sf_df = pd.DataFrame({'sf': sf})
    sf_df.index = timepoints
    return sf_df


def create_design_matrix(
    events_df_long: pd.DataFrame,
    oversampling: int = 50,
    tr: float = 1,
    verbose: bool = False,
    add_deriv: bool = False,
    maxtime: float = None,
) -> pd.DataFrame:
    """
    Create a design matrix for a given events DataFrame, which contains 'onset', 'trial_type'
    and 'duration' columns.  A regressor is constructed for each `trial_type`

    Parameters
    ----------
    events_df_long : pd.DataFrame
        A DataFrame containing event-related information in long format (one row per event).
        Must have the following columns:
        - 'onset' : The time of the event onset.
        - 'trial_type' : The type/category of the event (e.g., stimulus condition).
        - 'duration' : The duration of the event.
    oversampling : int, optional
        The factor by which to oversample the time points, by default 50. This controls
        the temporal resolution of the regressors where convolution is performed and
        the output regressors are downsampled to the tr.
    tr : float, optional
        The repetition time (TR) in seconds, by default 1. It specifies the time between
        successive scans or measurements.
    verbose : bool, optional
        If True, prints additional information about the process, by default False.
    add_deriv : bool, optional
        If True, includes derivative regressors in the design matrix, by default False.
    maxtime : float, optional
        Maximum time for the design matrix, by default None. If None, it is set to the maximum
        onset time plus 15 seconds.

    Returns
    -------
    pd.DataFrame
        A design matrix in the form of a DataFrame where each column corresponds to a regressor
        and each row corresponds to a time point.
    """
    conv_resolution = tr / oversampling
    if maxtime is None:
        maxtime = np.ceil(np.max(events_df_long['onset']) + 15)
    else:
        maxtime = np.round(maxtime)
    timepoints_conv = np.round(np.arange(0, maxtime, conv_resolution), 3)
    timepoints_data = np.round(np.arange(0, maxtime, tr), 3)
    hrf_func = spm_hrf(tr, oversampling=oversampling)
    hrf_deriv_func = spm_time_derivative(tr, oversampling=oversampling)
    if verbose:
        print(f'Maxtime: {maxtime}')
        print(f'Timepoints convolution: {timepoints_conv.shape}')
        print(f'Timepoints data: {timepoints_data.shape}')
    trial_types = events_df_long['trial_type'].unique()
    desmtx_microtime = pd.DataFrame()
    desmtx_conv_microtime = pd.DataFrame()

    for trial_type in trial_types:
        trial_type_onsets = events_df_long[events_df_long['trial_type'] == trial_type][
            'onset'
        ].values
        trial_type_durations = events_df_long[
            events_df_long['trial_type'] == trial_type
        ]['duration'].values
        sf_df = make_stick_function(
            trial_type_onsets, trial_type_durations, maxtime, resolution=conv_resolution
        )
        desmtx_microtime[trial_type] = sf_df.sf.values
        desmtx_conv_microtime[trial_type] = np.convolve(sf_df.sf.values, hrf_func)[
            : sf_df.shape[0]
        ]
        if add_deriv:
            desmtx_conv_microtime[f'{trial_type}_derivative'] = np.convolve(
                sf_df.sf.values, hrf_deriv_func
            )[: sf_df.shape[0]]
    desmtx_conv_microtime.index = timepoints_conv
    desmtx_conv = desmtx_conv_microtime.loc[timepoints_data]
    desmtx_conv = desmtx_conv[sorted(desmtx_conv.columns)]
    desmtx_conv['constant'] = 1
    return desmtx_conv


def est_contrast_vifs(desmat, contrasts):
    """
    IMPORTANT: This is only valid to use on design matrices where each regressor represents a condition vs baseline
     or if a parametrically modulated regressor is used the modulator must have more than 2 levels.  If it is a 2 level modulation,
     split the modulation into two regressors instead.

    Calculates VIF for contrasts based on the ratio of the contrast variance estimate using the
    true design to the variance estimate where between condition correaltions are set to 0
    desmat : pandas DataFrame, design matrix
    contrasts : dictionary of contrasts, key=contrast name,  using the desmat column names to express the contrasts
    returns: pandas DataFrame with VIFs for each contrast
    """
    desmat_copy = desmat.copy()
    # find location of constant regressor and remove those columns (not needed here)
    desmat_copy = desmat_copy.loc[
        :, (desmat_copy.nunique() > 1) | (desmat_copy.isnull().any())
    ]
    # Scaling stabilizes the matrix inversion
    nsamp = desmat_copy.shape[0]
    desmat_copy = (desmat_copy - desmat_copy.mean()) / (
        (nsamp - 1) ** 0.5 * desmat_copy.std()
    )
    vifs_contrasts = {}
    for contrast_name, contrast_string in contrasts.items():
        contrast_cvec = expression_to_contrast_vector(
            contrast_string, desmat_copy.columns
        )
        true_var_contrast = (
            contrast_cvec
            @ np.linalg.inv(desmat_copy.transpose() @ desmat_copy)
            @ contrast_cvec.transpose()
        )
        # The folllowing is the "best case" scenario because the between condition regressor correlations are set to 0
        best_var_contrast = (
            contrast_cvec
            @ np.linalg.inv(
                np.multiply(
                    desmat_copy.transpose() @ desmat_copy,
                    np.identity(desmat_copy.shape[1]),
                )
            )
            @ contrast_cvec.transpose()
        )
        vifs_contrasts[contrast_name] = [true_var_contrast / best_var_contrast]
    return pd.DataFrame(vifs_contrasts)


def load_event_files(subnum):
    try:
        events = pd.read_csv(
            f'../data/event_files/s{subnum}_surveyMedley_events.tsv', sep='\t'
        )
    except FileNotFoundError:
        print(
            f'Warning: surveyMedley_events.tsv is missing for subject {subnum} (subject omitted)'
        )
        return None, None

    try:
        orig_desmat = pd.read_csv(
            f'../data/simplified_events_rt/sub-{subnum}_task-surveyMedley_design-matrix.csv'
        )
    except FileNotFoundError:
        print(
            f'Warning: design-matrix.csv is missing for subject {subnum} (subject omitted)'
        )
        return None, None

    return events, orig_desmat


def make_desmat_sub(subnum: str, tr: float = 0.68) -> pd.DataFrame:
    """
    Create a design matrix for a given subject.

    Parameters
    ----------
    subnum : str
        The subject number.
    tr : float, optional
        The repetition time (TR) in seconds, by default 0.68.

    Returns
    -------
    pd.DataFrame
        The design matrix.
    """
    events, orig_desmat = load_event_files(subnum)
    if events is None or orig_desmat is None:
        return None
    events = events[events['onset'] >= 0]
    events_inc_dur_rt = events.copy()
    events_inc_dur_rt['duration'] = events_inc_dur_rt['response_time'].values
    events_inc_dur_rt['trial_type'] = 'rt'
    events_inc_dur_rt = pd.concat([events, events_inc_dur_rt])

    maxtime = orig_desmat.shape[0] * tr
    desmat = create_design_matrix(
        events_inc_dur_rt, tr=0.68, maxtime=maxtime
    ).reset_index(drop=True)
    # if any columns are all 0s return None
    if (desmat == 0).all(axis=0).any():
        print(
            f'Warning: Subject {subnum} does not have a design matrix because one or more columns was all 0s'
        )
        return None
    # sometimes is 1 TR too long due to rounding
    desmat = desmat.iloc[: orig_desmat.shape[0]]
    # add the motion/first and cosine drift
    nuisance_regs = orig_desmat.filter(
        regex='^(trans|rot|cosine|num|response_time)', axis=1
    ).reset_index(drop=True)
    return pd.concat([desmat, nuisance_regs], axis=1)


def make_desmat_sub_nilearn(subnum: str, tr: float = 0.68) -> pd.DataFrame:
    """
    Create a design matrix for a given subject, but use Nilearn for convolution.

    Parameters
    ----------
    subnum : str
        The subject number.
    tr : float, optional
        The repetition time (TR) in seconds, by default 0.68.

    Returns
    -------
    pd.DataFrame
        The design matrix.
    """
    from nilearn.plotting import plot_design_matrix

    events, orig_desmat = load_event_files(subnum)
    if events is None or orig_desmat is None:
        return None
    events = events[events['onset'] >= 0]
    if events['response_time'].isna().any():
        # print(f'Warning: NaNs found in response_time for subject {subnum}')
        events = events.dropna(subset=['response_time'])
    events_inc_dur_rt = events.copy()
    events_inc_dur_rt['duration'] = events_inc_dur_rt['response_time'].values
    events_inc_dur_rt['trial_type'] = 'rt_duration'
    events_inc_dur_rt = pd.concat([events, events_inc_dur_rt])

    events_inc_dur_rt = events_inc_dur_rt[['onset', 'duration', 'trial_type']]

    maxtime = orig_desmat.shape[0] * tr
    frame_times = np.arange(0, maxtime, tr)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        desmat = make_first_level_design_matrix(
            frame_times,
            events=events_inc_dur_rt,
            hrf_model='spm',
            drift_model=None,
            min_onset=-24,
            oversampling=5,
        ).reset_index(drop=True)
    if w:
        print(
            f'Error for subnum {subnum} \n from make_first_level_design_matrix: {w[-1].message}'
        )
    # if any columns are all 0s return None
    if (desmat == 0).all(axis=0).any():
        print(
            f'Warning: Subject {subnum} does not have a design matrix because one or more columns was all 0s'
        )
        return None
    # sometimes is 1 TR too long due to rounding
    desmat = desmat.iloc[: orig_desmat.shape[0]]
    # add the motion/first and cosine drift
    nuisance_regs = orig_desmat.filter(
        regex='^(trans|rot|cosine|num|response_time)', axis=1
    ).reset_index(drop=True)

    return pd.concat([desmat, nuisance_regs], axis=1)
