import warnings

import pandas as pd
import numpy as np

def compute_expsumlogp(
        df, seq, min_cov, window, min_window_values, 
        subtract_background=False):
    fwd_logp = np.full(len(seq), np.nan)
    sel = (
        (df.dir == 'fwd') & 
        ~pd.isna(df.u_test_pval) & 
        (df.N_wga >= min_cov) & 
        (df.N_nat >= min_cov))
    fwd_logp[df.loc[sel].index] = -np.log10(df.loc[sel, 'u_test_pval'])
    rev_logp = np.full(len(seq), np.nan)
    sel = (
        (df.dir == 'rev') & 
        ~pd.isna(df.u_test_pval) & 
        (df.N_wga >= min_cov) & 
        (df.N_nat >= min_cov))
    rev_logp[df.loc[sel].index] = -np.log10(df.loc[sel, 'u_test_pval'])
    fwd_expsumlogp = 10**(
        -pd.Series(fwd_logp).rolling(
            window, center=True, min_periods=min_window_values).sum())
    rev_expsumlogp = 10**(
        -pd.Series(rev_logp).rolling(
            window, center=True, min_periods=min_window_values).sum())
    if subtract_background:
        dist = round(window / 2 + 0.5)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            fwd_background = np.nanmax(
                np.column_stack(
                    [np.pad(fwd_expsumlogp[dist:], (0,dist), 
                        constant_values=np.nan), 
                    np.pad(fwd_expsumlogp[:-dist], (dist,0), 
                        constant_values=np.nan)]
                ), axis=1)
            rev_background = np.nanmax(
                np.column_stack(
                    [np.pad(rev_expsumlogp[dist:], (0,dist), 
                        constant_values=np.nan), 
                    np.pad(rev_expsumlogp[:-dist], (dist,0), 
                        constant_values=np.nan)]
                ), axis=1)
        return fwd_expsumlogp + fwd_background, rev_expsumlogp + rev_background
    return fwd_expsumlogp, rev_expsumlogp

def compute_diffs(df, seq, min_cov=10):
    fwd_diff = np.full(len(seq), np.nan)
    sel = (
        (df.dir == 'fwd') & 
        ~pd.isna(df.mean_diff) &
        (df.N_wga >= min_cov) & 
        (df.N_nat >= min_cov))
    fwd_diff[df.loc[sel].index] = df.loc[sel, 'mean_diff']
    rev_diff = np.full(len(seq), np.nan)
    sel = (
        (df.dir == 'rev') & 
        ~pd.isna(df.mean_diff) &
        (df.N_wga >= min_cov) & 
        (df.N_nat >= min_cov))
    rev_diff[df.loc[sel].index] = df.loc[sel, 'mean_diff']
    return fwd_diff, rev_diff

def compute_meanabsdiff(
        df, seq, min_cov, window, min_window_values, 
        subtract_background=False):
    fwd_diff, rev_diff = compute_diffs(df, seq, min_cov=min_cov)
    fwd_mean_abs_diff = pd.Series(np.abs(fwd_diff)).rolling(
        window, center=True, min_periods=min_window_values).mean()
    rev_mean_abs_diff = pd.Series(np.abs(rev_diff)).rolling(
        window, center=True, min_periods=min_window_values).mean()
    if subtract_background:
        dist = round(window / 2 + 0.5)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            fwd_background = np.nanmin(
                np.column_stack(
                    [np.pad(fwd_mean_abs_diff[dist:], (0,dist), 
                        constant_values=np.nan), 
                     np.pad(fwd_mean_abs_diff[:-dist], (dist,0), 
                        constant_values=np.nan)]
                ), axis=1)
            rev_background = np.nanmin(
                np.column_stack(
                    [np.pad(rev_mean_abs_diff[dist:], (0,dist), 
                        constant_values=np.nan), 
                     np.pad(rev_mean_abs_diff[:-dist], (dist,0), 
                        constant_values=np.nan)]
                ), axis=1)
        return (fwd_mean_abs_diff - fwd_background, 
                rev_mean_abs_diff - rev_background)
    return (fwd_mean_abs_diff, 
            rev_mean_abs_diff)

def compute_meanfrac(
        df, seq, min_cov, window, min_window_values, 
        subtract_background=False):
    df = df.reset_index().sort_values("fracmod", ascending=False)
    df = df.groupby(
        ['position', 'dir'], as_index=False).first().set_index('position')
    fwd_frac = np.full(len(seq), np.nan)
    sel = (
        (df.dir == 'fwd') & 
        ~pd.isna(df.fracmod) &
        (df['cov'] >= min_cov))
    fwd_frac[df.loc[sel].index] = df.loc[sel, 'fracmod']
    rev_frac = np.full(len(seq), np.nan)
    sel = (
        (df.dir == 'rev') & 
        ~pd.isna(df.fracmod) &
        (df['cov'] >= min_cov))
    rev_frac[df.loc[sel].index] = df.loc[sel, 'fracmod']
    fwd_mean_max_frac = pd.Series(fwd_frac).rolling(
        window, center=True, min_periods=min_window_values).mean()
    rev_mean_max_frac = pd.Series(rev_frac).rolling(
        window, center=True, min_periods=min_window_values).mean()
    if subtract_background:
        dist = round(window / 2 + 0.5)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            fwd_background = np.nanmin(
                np.column_stack(
                    [np.pad(fwd_mean_max_frac[dist:], (0,dist), 
                        constant_values=np.nan), 
                     np.pad(fwd_mean_max_frac[:-dist], (dist,0), 
                        constant_values=np.nan)]
                ), axis=1)
            rev_background = np.nanmin(
                np.column_stack(
                    [np.pad(rev_mean_max_frac[dist:], (0,dist), 
                        constant_values=np.nan), 
                     np.pad(rev_mean_max_frac[:-dist], (dist,0), 
                        constant_values=np.nan)]
                ), axis=1)
        return (fwd_mean_max_frac - fwd_background, 
                rev_mean_max_frac - rev_background)
    return (fwd_mean_max_frac, 
            rev_mean_max_frac)