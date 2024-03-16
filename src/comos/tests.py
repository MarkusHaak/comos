import warnings
import re

import pandas as pd
import numpy as np
import sanamos

from .constants import IUPAC_TO_IUPAC,IUPAC_TO_LIST,IUPAC_NOT
from .motif import reverse_complement,explode_motif

def test_ambiguous_metric(
        motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, 
        rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=30, 
        debug=False, min_tests=2, bases="AC", rec=False):
    biparite = False
    if "NNN" in motif: # bipartite
        biparite = True
        # only check flanking Ns
        callback = lambda pat: (
            pat.group(1) + 
            pat.group(2).lower() + 
            pat.group(3))
        motif = re.sub(r"(N)(N+)(N)", callback, motif)
    else:
        motif = motif.replace('N', 'n')
    motif_exp = ["N"] + list(motif) + ["N"]
    to_test = []
    num_amb = 0
    for i in range(len(motif_exp)):
        if motif_exp[i] =='N':
            num_amb += 1
            for b in IUPAC_TO_LIST[motif_exp[i]]:
                testcase = "".join(
                    motif_exp[:i] + [b] + 
                    motif_exp[i+1:]).strip('N').upper()
                to_test.append(testcase)
    aggregates, Ns = aggr_fct(
        to_test, max_motif_len+1, fwd_metric, 
        rev_metric, sa, bases=bases)
    # shift the data of the first four testcases to the left, 
    # because they have an additional leading base
    for i in range(4):
        aggregates[i, :-1] = aggregates[i, 1:]
        aggregates[i, -1] = np.nan 
        Ns[i, :-1] = Ns[i, 1:]
        Ns[i, -1] = 0
    no_sites = np.all(Ns == 0, axis=1)
    pois = np.all(~np.isnan(aggregates[~no_sites]), axis=0)
    if np.all(~pois):
        # no single C/A site with sufficient coverage
        if biparite and not rec:
            # for bipartite motifs, its OK if only RC passes 
            # (RC always mathylated too, 2x amount of tests anyways)
            return test_ambiguous_metric(
                reverse_complement(motif), len(motif) - poi - 1, 
                sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
                aggr_fct, opt_dir, ambiguity_thr, min_N=min_N, 
                debug=debug, min_tests=min_tests, bases=bases, rec=True)
        return False
    # instead of taking min over all array (best),
    # take max over columns (repr. poi) and min over those
    Ns = np.clip(Ns, 1, len(mu)-1)
    stddevs = (aggregates - mu[Ns]) / sigma[Ns] # TODO: handle RuntimeWarnings
    stddevs[Ns < min_N] = np.nan # mask sites with low coverage
    if debug:
        pois_idx = np.where(pois)[0]
        to_test_ = [' ' + m if i >= 4 else m for i,m in enumerate(to_test)]
        to_test__ = []
        for m in to_test_:
            m = list(m)
            for i in pois_idx+1:
                m[i] = m[i].lower()
            to_test__.append("".join(m))
        print(pd.DataFrame(
            stddevs[~no_sites][:, pois].round(2),
            index=np.array(to_test__)[~no_sites],
            columns=pois_idx))
    if np.any((~np.isnan(stddevs[:, pois]))\
              .reshape(stddevs.shape[0] // 4, 4,-1)\
              .sum(axis=1).min(axis=1) < min_tests):
        # at least one ambiguous position has 
        # not enough test cases with sufficient coverage
        if biparite and not rec:
            # for bipartite motifs, its OK if only RC passes 
            # (RC always mathylated too, 2x amount of tests anyways)
            return test_ambiguous_metric(
                reverse_complement(motif), len(motif) - poi - 1, 
                sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
                aggr_fct, opt_dir, ambiguity_thr, min_N=min_N, 
                debug=debug, min_tests=min_tests, bases=bases, rec=True)
        return False
    if opt_dir == 'min':
        max_stddev_per_poi = np.nanmax(
            stddevs[~no_sites][:, pois], axis=0)
        # careful: best_pois refers to selection of pois, not a motif index!
        best_poi = np.argmin(max_stddev_per_poi) 
    else:
        max_stddev_per_poi = np.nanmin(stddevs[~no_sites][:, pois], axis=0)
        best_poi = np.argmax(max_stddev_per_poi)
    if debug:
        print('best column:', best_poi, '= motif index',
              np.searchsorted(np.cumsum(pois), best_poi+1))
    if opt_dir == 'min':
        return max_stddev_per_poi[best_poi] < ambiguity_thr
    else:
        return max_stddev_per_poi[best_poi] > ambiguity_thr

def get_EDLogo_enrichment_scores(
        motif, poi, ind, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, 
        min_N=10, stddev=3.0, n_pseudo=5, Nnull=500):
    if ind < 0:
        motif_exp = ["N"] * (-ind) + list(motif)
        poi_ = int(poi) - ind
        ind_ = 0
    elif ind >= len(motif):
        motif_exp = list(motif) + ["N"] * (len(motif) - ind + 1)
        poi_ = int(poi)
        ind_ = ind
    else:
        motif_exp = list(motif)
        poi_ = int(poi)
        ind_ = ind
    test_cases = []
    for b in IUPAC_TO_LIST['N']:
        testcase = "".join(motif_exp[:ind_] + [b] + motif_exp[ind_+1:])
        testcase = testcase.strip('N').upper()
        test_values = []
        for sa_, fwd_metric_, rev_metric_ in zip(sa, fwd_metric, rev_metric):
            ind_fwd, ind_rev = sanamos.find_motif(testcase, sa_, poi=poi_)
            test_values.append(fwd_metric_[ind_fwd])
            test_values.append(rev_metric_[ind_rev])
        test_values = np.concatenate(test_values)
        Ntot = test_values.shape[0]
        test_values = test_values[~np.isnan(test_values)]
        Nval = test_values.shape[0]
        mu_, sigma_ = mu[Nnull], sigma[Nnull]
        if opt_dir == 'min':
            can = (test_values >= mu_ - stddev * sigma_).sum()
        else:
            can = (test_values <= mu_ + stddev * sigma_).sum()
        mod = Nval - can
        test_cases.append((ind, b, Ntot, Nval, can, mod))
    df = pd.DataFrame(
        test_cases, columns=['mindex','base','Ntot','Nval','can','mod'])
    # filter bases with insufficient coverage
    if min_N:
        df = df.loc[df.Nval >= min_N]
    # stabilizing estimates
    # by adding pseudocounts
    df['q'] = df['Nval'] / df['Nval'].sum()
    df['modpseudo'] = df['mod']
    df['modpseudo'] += df['q'] * df.shape[0] * n_pseudo

    #df['p'] = df['mod'] / df['mod'].sum()
    df['p'] = df['modpseudo'] / df['modpseudo'].sum()
    df['r~'] = np.log2(df['p'] / df['q'])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        df['r'] = df['r~'] - np.nanmedian(df['r~'])
    return df

def test_ambiguous_logo(
        motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
        aggr_fct, opt_dir, ambiguity_thr, min_N=30, debug=False, min_tests=2, 
        bases="AC", rec=False):
    biparite = False
    if "NNN" in motif: # bipartite
        biparite = True
        # only check flanking Ns
        callback = lambda pat: pat.group(1)+pat.group(2).lower()+pat.group(3)
        motif_ = re.sub(r"(N)(N+)(N)", callback, motif[:])
    else:
        motif_ = motif[:].replace('N', 'n')
    motif_ = list(motif_)
    ambiguous_pos = [-1, len(motif)]
    for i in range(len(motif_)):
        if motif_[i] == 'N':
            ambiguous_pos.append(i)
    for i in ambiguous_pos:
        df = get_EDLogo_enrichment_scores(
            motif, poi, i, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, 
            min_N=min_N, stddev=3.0, n_pseudo=5, Nnull=500)
        if debug:
            print(df)
        #(df['mod'] >= min_N).sum() < min_tests or \
        if np.abs(df['r']).max() > abs(ambiguity_thr) or \
           df.shape[0] < min_tests:
            if biparite and not rec:
                rc = reverse_complement(motif)
                if rc != motif:
                    # determine poi in RC
                    aggregates, Ns = aggr_fct(
                        [rc], len(rc), fwd_metric, rev_metric, sa, bases=bases)
                    if opt_dir == "min":
                        best_aggregate = np.nanmin(aggregates[0])
                    else:
                        best_aggregate = np.nanmax(aggregates[0])
                    assert ~np.isnan(best_aggregate), \
                        f"unhandled exception: found no motif "+\
                        f"aggregate for motif {diff}"
                    if opt_dir == "min":
                        poi_rc = np.nanargmin(aggregates[0])
                    else:
                        poi_rc = np.nanargmax(aggregates[0])
                    if debug:
                        print(f'fail, check RC {rc}:{int(poi)}')
                    if test_ambiguous_logo(
                            rc, poi_rc, sa, max_motif_len, mu, sigma, 
                            fwd_metric, rev_metric, aggr_fct, opt_dir, 
                            ambiguity_thr, min_N=min_N, debug=debug, 
                            min_tests=min_tests, bases=bases, rec=True):
                        return True
            return False
    return True

def test_replace(
        motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
        aggr_fct, opt_dir, thr, min_N=30):
    nonambiguous_pos = [i for i in range(len(motif)) if motif[i] != 'N']
    for i in nonambiguous_pos:
        df = get_EDLogo_enrichment_scores(
            motif, poi, i, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, 
            min_N=min_N, stddev=3.0, n_pseudo=5, Nnull=500)
        sel = df['base'].isin(IUPAC_TO_LIST[motif[i]])
        if sel.sum() != len(IUPAC_TO_LIST[motif[i]]):
            return False
        if not np.all(df.loc[sel, 'r'] >= thr):
            return False
    return True

def test_replace_order(
        motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
        aggr_fct, opt_dir, min_N=30):
    nonambiguous_pos = [i for i in range(len(motif)) if motif[i] != 'N']
    for i in nonambiguous_pos:
        df = get_EDLogo_enrichment_scores(
            motif, poi, i, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, 
            min_N=min_N, stddev=3.0, n_pseudo=5, Nnull=500)
        sel = df['base'].isin(IUPAC_TO_LIST[motif[i]])
        if sel.sum() != len(IUPAC_TO_LIST[motif[i]]):
            return False
        if (set(df.sort_values('r', ascending=False)\
                [:len(IUPAC_TO_LIST[motif[i]])]['base']) 
            != set(IUPAC_TO_LIST[motif[i]])):
            return False
    return True

def test_replace_total(
        motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
        aggr_fct, opt_dir, thr, min_N=30):
    nonambiguous_pos = [i for i in range(len(motif)) if motif[i] != 'N']
    for i in nonambiguous_pos:
        df = get_EDLogo_enrichment_scores(
            motif, poi, i, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, #
            min_N=min_N, stddev=3.0, n_pseudo=5, Nnull=500)
        if not df['r'].abs().sum() >= thr:
            return False
    return True

def test_ambiguous(
        motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
        aggr_fct, opt_dir, ambiguity_thr, min_N=30, debug=False, min_tests=2, 
        bases="AC", rec=False, ambiguity_type='logo'):
    if ambiguity_type == 'logo':
        return test_ambiguous_logo(
            motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
            aggr_fct, opt_dir, ambiguity_thr, min_N=min_N, debug=debug, 
            min_tests=min_tests, bases=bases, rec=rec)
    elif ambiguity_type == 'metric':
        return test_ambiguous_metric(
            motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
            aggr_fct, opt_dir, ambiguity_thr, min_N=min_N, debug=debug, 
            min_tests=min_tests, bases=bases, rec=rec)

def test_exploded(
        motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, 
        aggr_fct, opt_dir, selection_thr, min_N=30, debug=False, bases="AC"):
    to_test = explode_motif(motif)
    if len(to_test) == 1:
        return True
    
    aggregates, Ns = aggr_fct(
        to_test, max_motif_len+1, fwd_metric, rev_metric, sa, bases=bases)
    no_sites = np.all(Ns == 0, axis=1)
    pois = np.all(~np.isnan(aggregates[~no_sites]), axis=0)
    # instead of taking min over all array (best),
    # take max over columns (repr. poi) and min over those
    Ns = np.clip(Ns, 1, len(mu)-1)
    stddevs = (aggregates - mu[Ns]) / sigma[Ns] # TODO: handle RuntimeWarnings
    stddevs[Ns < min_N] = np.nan # mask sites with low coverage
    if opt_dir == "min":
        max_stddev_per_poi = np.nanmax(stddevs[~no_sites][:, pois], axis=0)
        # careful: best_pois refers to selection of pois, not a motif index!
        best_poi = np.argmin(max_stddev_per_poi) 
    else:
        max_stddev_per_poi = np.nanmin(stddevs[~no_sites][:, pois], axis=0)
        # careful: best_pois refers to selection of pois, not a motif index!
        best_poi = np.argmax(max_stddev_per_poi) 
    if debug:
        pois_idx = np.where(pois)[0]
        to_test_ = [m for i,m in enumerate(to_test)]
        to_test__ = []
        for m in to_test_:
            m = list(m)
            for i in pois_idx:
                m[i] = m[i].lower()
            to_test__.append("".join(m))
        print(pd.DataFrame(stddevs[~no_sites][:, pois].round(2),
                           index=to_test__,
                           columns=pois_idx))
        print('best column:', best_poi, '= motif index', 
              np.searchsorted(np.cumsum(pois), best_poi+1))
    if opt_dir == "min":
        return max_stddev_per_poi[best_poi] < selection_thr
    else:
        return max_stddev_per_poi[best_poi] > selection_thr