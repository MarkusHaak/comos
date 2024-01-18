import os
import warnings
import argparse
import logging
import itertools
import pickle
import time
import re

import pandas as pd
import numpy as np
import matplotlib
import pyreadr
from matplotlib import pyplot as plt
from Bio import SeqIO
from scipy.optimize import curve_fit
from scipy.stats import norm
from tqdm import tqdm

import suffix_array

matplotlib.use('TkAgg')

class ArgHelpFormatter(argparse.HelpFormatter):
    '''
    Formatter that properly adds default values to help texts.
    '''
    def __init__(self, prog):
        super().__init__(prog)

    def _get_help_string(self, action):
        text = action.help
        if  action.default is not None and \
            action.default != argparse.SUPPRESS and \
            'default:' not in text.lower():
            text += ' (default: {})'.format(action.default)
        return text

# dict storing rules for combining two IUPAC letters
IUPAC_TO_IUPAC = {
        "A" : {"A":"A", "C":"M", "G":"R", "T":"W", "M":"M", "R":"R", "W":"W", "S":"V", "Y":"H", "K":"D", "V":"V", "H":"H", "D":"D", "B":"N", "N":"N",},
        "C" : {"A":"M", "C":"C", "G":"S", "T":"Y", "M":"M", "R":"V", "W":"H", "S":"S", "Y":"Y", "K":"B", "V":"V", "H":"H", "D":"N", "B":"B", "N":"N",},
        "G" : {"A":"R", "C":"S", "G":"G", "T":"K", "M":"V", "R":"R", "W":"D", "S":"S", "Y":"B", "K":"K", "V":"V", "H":"N", "D":"D", "B":"B", "N":"N",},
        "T" : {"A":"W", "C":"Y", "G":"K", "T":"T", "M":"H", "R":"D", "W":"W", "S":"B", "Y":"Y", "K":"K", "V":"N", "H":"H", "D":"D", "B":"B", "N":"N",},
        "M" : {"A":"M", "C":"M", "G":"V", "T":"H", "M":"M", "R":"V", "W":"H", "S":"V", "Y":"H", "K":"N", "V":"V", "H":"H", "D":"N", "B":"N", "N":"N",},
        "R" : {"A":"R", "C":"V", "G":"R", "T":"D", "M":"V", "R":"R", "W":"D", "S":"V", "Y":"N", "K":"D", "V":"V", "H":"N", "D":"D", "B":"N", "N":"N",},
        "W" : {"A":"W", "C":"H", "G":"D", "T":"W", "M":"H", "R":"D", "W":"W", "S":"N", "Y":"H", "K":"D", "V":"N", "H":"H", "D":"D", "B":"N", "N":"N",},
        "S" : {"A":"V", "C":"S", "G":"S", "T":"B", "M":"V", "R":"V", "W":"N", "S":"S", "Y":"B", "K":"B", "V":"V", "H":"N", "D":"N", "B":"B", "N":"N",},
        "Y" : {"A":"H", "C":"Y", "G":"B", "T":"Y", "M":"H", "R":"N", "W":"H", "S":"B", "Y":"Y", "K":"B", "V":"N", "H":"H", "D":"N", "B":"B", "N":"N",},
        "K" : {"A":"D", "C":"B", "G":"K", "T":"K", "M":"N", "R":"D", "W":"D", "S":"B", "Y":"B", "K":"K", "V":"N", "H":"N", "D":"D", "B":"B", "N":"N",},
        "V" : {"A":"V", "C":"V", "G":"V", "T":"N", "M":"V", "R":"V", "W":"N", "S":"V", "Y":"N", "K":"N", "V":"V", "H":"N", "D":"N", "B":"N", "N":"N",},
        "H" : {"A":"H", "C":"H", "G":"N", "T":"H", "M":"H", "R":"N", "W":"H", "S":"N", "Y":"H", "K":"N", "V":"N", "H":"H", "D":"N", "B":"N", "N":"N",},
        "D" : {"A":"D", "C":"N", "G":"D", "T":"D", "M":"N", "R":"D", "W":"D", "S":"N", "Y":"N", "K":"D", "V":"N", "H":"N", "D":"D", "B":"N", "N":"N",},
        "B" : {"A":"N", "C":"B", "G":"B", "T":"B", "M":"N", "R":"N", "W":"N", "S":"B", "Y":"B", "K":"B", "V":"N", "H":"N", "D":"N", "B":"B", "N":"N",},
        "N" : {"A":"N", "C":"N", "G":"N", "T":"N", "M":"N", "R":"N", "W":"N", "S":"N", "Y":"N", "K":"N", "V":"N", "H":"N", "D":"N", "B":"N", "N":"N",},
    }

IUPAC_TO_LIST = {
    "A" : ["A"],
    "C" : ["C"],
    "G" : ["G"],
    "T" : ["T"],
    "M" : ["A", "C"],
    "R" : ["A", "G"],
    "W" : ["A", "T"],
    "S" : ["C", "G"],
    "Y" : ["C", "T"],
    "K" : ["G", "T"],
    "V" : ["A", "C", "G"],
    "H" : ["A", "C", "T"],
    "D" : ["A", "G", "T"],
    "B" : ["C", "G", "T"],
    "N" : ["A", "C", "G", "T"],
}

# for fast reverse complement translation
comp_trans = str.maketrans("ACGTMRWSYKVHDBN", "TGCAKYWSRMBDHVN")
def reverse_complement(seq):
    return seq.translate(comp_trans)[::-1]

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=ArgHelpFormatter,
        )
    io_grp = parser.add_argument_group('Input/Output Options')
    io_grp.add_argument(
        '--genome', '-g', 
        required=True,
        help='Fasta file containing the genome contig(s).')
    io_grp.add_argument(
        '--rds', '-r', 
        required=True,
        help='RDS Rdata file containing the sample differences.')
    io_grp.add_argument(
        '--out', '-o', 
        default='./cosmos_output',
        help='Folder for the output files.')
    io_grp.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Create and save plots.'
    )
    io_grp.add_argument(
        '--cache',
        default='tmp',
        help='Directory used for storing cache files.'
    )

    motif_grp = parser.add_argument_group('Motif Options')
    motif_grp.add_argument(
        '--min-k',
        type=int,
        default=4,
        help='Minimum number of specific motif positions.'
    )
    motif_grp.add_argument(
        '--max-k',
        type=int,
        default=8,
        help='Maximum number of specific motif positions.'
    )
    motif_grp.add_argument(
        '--min-g',
        type=int,
        default=4,
        help='Minimum gap size in TypeI motifs.'
    )
    motif_grp.add_argument(
        '--max-g',
        type=int,
        default=9,
        help='Maximum gap size in TypeI motifs.'
    )

    hyper_grp = parser.add_argument_group('Hyperparameters')
    hyper_grp.add_argument(
        '--selection-thr',
        type=float,
        default=4.0,
        help='Minimum diversion from Null-model in number of std.dev. required for a motif to get selected.' 
    )
    hyper_grp.add_argument(
        '--ambiguity-thr',
        type=float,
        default=3.,
        help='Minimum diversion from Null-model in number of std.dev. required for an extended motif.' 
    )
    hyper_grp.add_argument(
        '--ambiguity-cov',
        type=float,
        default=30,
        help='Minimum coverage required to consider a motif in ambiguity testing.' 
    )
    hyper_grp.add_argument(
        '--ambiguity-quantile',
        type=float,
        default=0.8,
        help='Quantile of the ambiguity-replaced motifs that needs to be below the ambiguity threshold.' 
    )

    misc_grp = parser.add_argument_group('Misc')
    misc_grp.add_argument(
        '--test-motifs',
        nargs='+',
        help='Analyze the given IUPAC motifs.' 
    )
    misc_grp.add_argument(
        '--analysis-mode',
        type=float,
        default=None,
        help='Analyze influence of the selection threshold on resulting motifs down to this threshold value (default: None).' 
    )


    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.genome):
        logging.getLogger('comos').error(f"Input file {args.genome} does not exist.")
        exit(1)
    if not os.path.exists(args.rds):
        logging.getLogger('comos').error(f"Input file {args.rds} does not exist.")
        exit(1)
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    args.genome = os.path.abspath(args.genome)
    args.rds = os.path.abspath(args.rds)

    return args

def parse_diff_file(fp, contig_id):
    df = pyreadr.read_r(fp)[None]
    # correct positions
    df['position'] = df['position'].astype(int) - 1 + 3 # convert to 0-based indexing
    df.loc[df.dir == 'rev', 'position'] += 1
    df = df.set_index(df.position)
    df = df.loc[df.contig == contig_id]
    return df

def parse_largest_contig(fp, cache_dir='tmp'):
    n_largest = 0
    contig_id = None
    n = 0
    for record in SeqIO.parse(fp, "fasta"):
        n += 1
        if len(record.seq) > n_largest:
            contig_id = record.id
            n_largest = len(record.seq)
            seq = record.seq
            sa = suffix_array.get_suffix_array(record.id, record.seq, cache_dir=cache_dir)
    return seq, sa, contig_id, n

def compute_expsumlogp(df, seq):
    fwd_logp = np.full(len(seq), np.nan)
    fwd_logp[df.loc[(df.dir == 'fwd') & ~pd.isna(df.u_test_pval)].index] = \
        -np.log10(df.loc[(df.dir == 'fwd') & ~pd.isna(df.u_test_pval), 'u_test_pval'])
    rev_logp = np.full(len(seq), np.nan)
    rev_logp[df.loc[(df.dir == 'rev') & ~pd.isna(df.u_test_pval)].index] = \
        -np.log10(df.loc[(df.dir == 'rev') & ~pd.isna(df.u_test_pval), 'u_test_pval'])
    fwd_expsumlogp = 10**(-pd.Series(fwd_logp).rolling(5, center=True, min_periods=3).sum())
    rev_expsumlogp = 10**(-pd.Series(rev_logp).rolling(5, center=True, min_periods=3).sum())
    return fwd_expsumlogp, rev_expsumlogp

def compute_diffs(df, seq, min_cov=10):
    fwd_diff = np.full(len(seq), np.nan)
    sel = (df.dir == 'fwd') & ~pd.isna(df.mean_diff) & \
          (df.N_wga >= min_cov) & (df.N_nat >= min_cov)
    fwd_diff[df.loc[sel].index] = df.loc[sel, 'mean_diff']
    rev_diff = np.full(len(seq), np.nan)
    sel = (df.dir == 'rev') & ~pd.isna(df.mean_diff) & \
          (df.N_wga >= min_cov) & (df.N_nat >= min_cov)
    rev_diff[df.loc[sel].index] = df.loc[sel, 'mean_diff']
    return fwd_diff, rev_diff

def compute_expsumlogp(df, seq):
    fwd_logp = np.full(len(seq), np.nan)
    fwd_logp[df.loc[(df.dir == 'fwd') & ~pd.isna(df.u_test_pval)].index] = \
        -np.log10(df.loc[(df.dir == 'fwd') & ~pd.isna(df.u_test_pval), 'u_test_pval'])
    rev_logp = np.full(len(seq), np.nan)
    rev_logp[df.loc[(df.dir == 'rev') & ~pd.isna(df.u_test_pval)].index] = \
        -np.log10(df.loc[(df.dir == 'rev') & ~pd.isna(df.u_test_pval), 'u_test_pval'])
    fwd_expsumlogp = 10**(-pd.Series(fwd_logp).rolling(5, center=True, min_periods=3).sum())
    rev_expsumlogp = 10**(-pd.Series(rev_logp).rolling(5, center=True, min_periods=3).sum())
    return fwd_expsumlogp, rev_expsumlogp

def get_seq_index_combinations(k_min, k_max, min_gap, max_gap, bases, blur):
    def get_indexes(comb, k):
        indexes = set()
        for i,b in enumerate(comb):
            if b in bases:
                for j in range(i-blur,i+blur+1):
                    if j >= 0 and j < k:
                        if comb[j] != "N":
                            indexes.add(j)
        return indexes
    seq_index_combinations = {}
    for k in range(k_min, k_max+1):
        for comb in itertools.product(['A','C','G','T'], repeat=k):
            comb = list(comb)
            indexes = get_indexes(comb, k)
            if not indexes:
                continue
            seq = "".join(comb)
            seq_index_combinations[seq] = list(indexes)
            for gap in range(min_gap, max_gap+1):
                #symmetrical sequence lengths next to gap
                if not k%2:
                    gap_pos = k//2
                    comb_ = comb[:gap_pos] + ["N"]*gap + comb[gap_pos:]
                    indexes_ = get_indexes(comb_, k+gap)
                    seq_ = "".join(comb_)
                    seq_index_combinations[seq_] = indexes_
                else:
                    for gap_pos in [k//2, (k//2)+1]:
                        comb_ = comb[:gap_pos] + ["N"]*gap + comb[gap_pos:]
                        indexes_ = get_indexes(comb_, k+gap)
                        seq_ = "".join(comb_)
                        seq_index_combinations[seq_] = indexes_
    return seq_index_combinations

def plot_diffs_with_complementary(motif, sa, fwd_diff, rev_diff, ymax=np.inf, offset=3, savepath=None):
    mlen = len(motif)
    positions = list(range(-offset, mlen+offset))
    ind_fwd, ind_rev = suffix_array.find_motif(motif, sa)
    # original strand
    n_fwd, n_rev = len(ind_fwd), len(ind_rev)
    data_fwd = []
    for i in positions:
        fwd_diffs = fwd_diff[ind_fwd + i]
        rev_diffs = rev_diff[ind_rev - i]
        diffs = np.abs(np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]]))
        data_fwd.append(diffs)
    # complementary strand
    n_fwd_c, n_rev_c = len(ind_fwd), len(ind_rev)
    data_rev = []
    for i in positions:
        fwd_diffs = rev_diff[ind_fwd + i]
        rev_diffs = fwd_diff[ind_rev - i]
        diffs = np.abs(np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]]))
        data_rev.append(diffs)
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.3)
    ax1.set_title(motif)
    ax1.boxplot(data_fwd)
    ax2.boxplot(data_rev)
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels([motif[i] if 0 <= i < mlen else "" for i in positions])
    ax2.set_xticks(ax2.get_xticks())
    ax2.set_xticklabels([motif[i].translate(comp_trans) if 0 <= i < mlen else "" for i in positions])
    ax2.xaxis.tick_top()
    ax1.set(ylabel=r"(+) strand abs. diff / $pA$")
    ax2.set(ylabel=r"(-) strand abs. diff / $pA$")
    max_ylim = (0, min(max(ax1.get_ylim()[1], ax2.get_ylim()[1]), ymax))
    ax1.set_ylim(max_ylim)
    ax2.set_ylim(max_ylim)
    ax1.grid(axis='y', zorder=-4)
    ax2.grid(axis='y', zorder=-4)
    ax2.invert_yaxis()
    ax1.spines[['right', 'top']].set_visible(False)
    ax2.spines[['right', 'bottom']].set_visible(False)
    if savepath:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

def func(x, a, b):
    # x, a, b, c
    #return a * np.exp(-b * x) + c # Exponential
    
    # x, a, b, c, d, e
    #return a * np.exp(-b * x) + c + d * np.log(e * x)
    #return a * np.exp(-b * x) + c + (d * x) / (e + x)
    #return a * np.exp(-b * x) + c + d * np.power(x, e)
    
    # x, a, b
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'divide by zero encountered in power')
        return a * np.power(x, b) # Power Curve
    #return (a * x) / (b + x) # Plateau Curve

    #return (1 / (1 + np.exp(-d*(x + f)))) * (a * np.exp(-b * x) + c) + (1 - 1 / (1 + np.exp(-d*(x + f)))) * (g * np.exp(-h * x) + i)


def normal_approx(d):
    mu = []
    sigma = []
    window = lambda x : 20 + x
    #N = np.unique(np.linspace(1,d.N.max(),1000).astype(int))
    N = np.unique(np.concatenate([np.geomspace(1,d.N.max(),500).astype(int), 
                                  np.linspace(1,d.N.max(),1000).astype(int)]))
    for n in N:
        sel = (d.N >= n - window(n)//2) & (d.N <= n + window(n)//2)
        sigma.append(d.loc[sel, 'val'].std())
        mu.append(d.loc[sel, 'val'].mean())
    
    #fig,(ax1,ax2) = plt.subplots(2,1)
    #try:
    #    popt, pcov = curve_fit(func, N, mu)
    #    #print(popt)
    #    p_mu = lambda x : func(x, *popt)
    #    ax1.scatter(N,mu,s=5)
    #    ax1.plot(np.linspace(0,d.N.max(),1000), p_mu(np.linspace(0,d.N.max(),1000)))
    #except:
    #    pass
    #try:
    #    popt2, pcov = curve_fit(func, N, sigma)
    #    #print(popt2)
    #    p_sigma = lambda x : func(x, *popt2)
    #    ax2.scatter(N,sigma,s=5)
    #    ax2.plot(np.linspace(0,d.N.max(),1000), p_sigma(np.linspace(0,d.N.max(),1000)))
    #except:
    #    pass
    #plt.show()

    popt, pcov = curve_fit(func, N, mu)
    p_mu = lambda x : func(x, *popt)
    popt2, pcov = curve_fit(func, N, sigma)
    p_sigma = lambda x : func(x, *popt2)
    return p_mu(np.array(range(0,d.N.max()+1))), p_sigma(np.array(range(0,d.N.max()+1))) # TODO: suppress warnings?

def plot_motif_scores(results, mu, sigma, thr=6., savepath=None):
    fig,ax = plt.subplots(figsize=(6,6))
    ax.scatter(results.N,results.val,s=1,alpha=0.25, color='black')
    X = np.linspace(0,5000, 1000)
    ax.plot(X, mu[X.astype(int)], color='C0')
    #ax.plot(X, mu[X.astype(int)] + thr*sigma[X.astype(int)], color='C0', linestyle=':')
    ax.plot(X, mu[X.astype(int)] - thr*sigma[X.astype(int)], color='C0', linestyle=':')
    ax.grid()
    ax.set(ylim=(-0.0005, ax.get_ylim()[1]), xlim=(-50,5000),
        xlabel="# motif sites", ylabel="10**(sumlog p)")
    if savepath:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

def combine_IUPAC(m1, m2):
    assert len(m1) == len(m2)
    return "".join([IUPAC_TO_IUPAC[b1][b2] for (b1,b2) in zip(m1, m2)])

def combine_motifs(m1, m2):
    if len(m1) < len(m2):
        mshort = m1
        mlong = m2
    else:
        mshort = m2
        mlong = m1
    diff = len(mlong) - len(mshort)
    
    combined_motifs = []
    for shift in range(-1,diff+2):
        if shift < 0:
            mlong_ = "N"*(-shift) + mlong
            mshort_ = mshort + "N"*(diff-shift)
        elif shift > diff:
            mlong_ = mlong + "N"*(shift - diff)
            mshort_ = "N"*(diff + (shift - diff)) + mshort
        else:
            mlong_ = mlong
            mshort_ = "N"*shift + mshort + "N"*(diff - shift)
        combined = combine_IUPAC(mshort_, mlong_).strip('N')
        if combined:
            combined_motifs.append(combined)
    return combined_motifs

def test_ambiguous(motif, poi, sa, max_motif_len, mu, sigma, fwd_expsumlogp, rev_expsumlogp, ambiguity_thr, ambiguity_quantile, min_N=50, debug=False):
    if "NNN" in motif: # Type I
        # only check flanking Ns
        callback = lambda pat: pat.group(1)+pat.group(2).lower()+pat.group(3)
        motif = re.sub(r"(N)(N+)(N)", callback, motif)
    motif_exp = ["N"] + list(motif) + ["N"]
    to_test = []
    pois = []
    for i in range(len(motif_exp)):
        #if motif_exp[i] == 'N':
        #    for b in "ACTG":
        if motif_exp[i] not in "ACGTn":
            for b in IUPAC_TO_LIST[motif_exp[i]]:
                testcase = "".join(motif_exp[:i] + [b] + motif_exp[i+1:]).strip('N').upper()
                to_test.append(testcase)
                if i == 0:
                    pois.append(poi+1)
                else:
                    pois.append(poi)
    pois = np.array(pois)
    means, Ns = suffix_array.motif_means(to_test, max_motif_len+1, fwd_expsumlogp, rev_expsumlogp, sa)
    Ns = Ns[range(Ns.shape[0]), pois]
    means = means[range(Ns.shape[0]), pois]
    sel = Ns > min_N
    Ns = Ns[sel]
    Ns = np.clip(Ns, 1, len(mu)-1)
    means = means[sel]
    stddevs = (means - mu[Ns]) / sigma[Ns]
    if debug:
        print(pd.DataFrame([(m, N, stddev) for m, N, stddev in zip(np.array(to_test)[sel], Ns, stddevs)], columns=['ext_motif', 'N', 'stddev']))
    return np.quantile(stddevs, ambiguity_quantile) < ambiguity_thr

def get_pois(motif):
    pois = []
    for i in range(len(motif)):
        if motif[i] in "CA":
            pois.append(i)
    return pois

def get_num_absorbed(motif, absorbed):
    if motif not in absorbed:
        return 1
    else:
        n = 0
        for m, stddevs in absorbed[motif]:
            n += get_num_absorbed(m, absorbed)
        return n

def reduce_motifs(d, sa, max_motif_len, mu, sigma, fwd_expsumlogp, rev_expsumlogp, thr, ambiguity_thr=-3., ambiguity_quantile=0.8, ambiguity_cov=50, absorbed=None):
    d = d.copy().sort_values(['N','stddevs'], ascending=[False,True])
    # initial filtering
    if ambiguity_thr is not None:
        d['pass'] = d.apply(lambda row: test_ambiguous(row.motif, row.poi, sa, max_motif_len, mu, sigma, fwd_expsumlogp, rev_expsumlogp, ambiguity_thr, ambiguity_quantile=ambiguity_quantile, min_N=ambiguity_cov), axis=1)
        for i,row in d.loc[d['pass'] == False].iterrows():
            logging.getLogger('comos').debug(f"excluded {row.motif}:{row.poi} ({row.N}, {row.stddevs:.2f}) : did not pass ambiguity filter")
        n_excluded = (d['pass'] == False).sum()
        if n_excluded:
            logging.info(f"Excluded {n_excluded} motifs because they did not pass ambiguity testing.")
        d = d.loc[d['pass']].drop(['pass'], axis=1)
    
    changed = True
    tested = set()
    if absorbed is None:
        absorbed = {}
    while changed == True:
        changed = False
        for i in range(0,len(d)):
            for j in range(i+1,len(d)):
                if ((d.iloc[i].motif, d.iloc[i].poi), (d.iloc[j].motif, d.iloc[j].poi)) in tested:
                    continue
                tested.add(((d.iloc[i].motif, d.iloc[i].poi), (d.iloc[j].motif, d.iloc[j].poi)))
                combined_motifs = combine_motifs(d.iloc[i].motif, d.iloc[j].motif)
                if d.iloc[i].motif in combined_motifs[1:-1]:
                    logging.getLogger('comos').debug(f"dropped {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f}), covered by {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f})")
                    # note which motifs get absorbed
                    if d.iloc[i].motif not in absorbed:
                        absorbed[d.iloc[i].motif] = []
                    #if d.iloc[j].motif in absorbed:
                    #    absorbed[d.iloc[i].motif].extend(absorbed[d.iloc[j].motif])
                    absorbed[d.iloc[i].motif].append((d.iloc[j].motif, d.iloc[j].stddevs))
                    # drop entry
                    d = d.drop(d.index[j])
                    changed = True
                    break
                if d.iloc[j].motif in combined_motifs[1:-1]:
                    logging.getLogger('comos').debug(f"dropped {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f}), covered by {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f})")
                    # note which motifs get absorbed
                    if d.iloc[j].motif not in absorbed:
                        absorbed[d.iloc[j].motif] = []
                    #if d.iloc[i].motif in absorbed:
                    #    absorbed[d.iloc[j].motif].extend(absorbed[d.iloc[i].motif])
                    absorbed[d.iloc[j].motif].append((d.iloc[i].motif, d.iloc[i].stddevs))
                    # drop entry
                    d = d.drop(d.index[i])
                    changed = True
                    break

                means, Ns = suffix_array.motif_means(combined_motifs, max_motif_len, fwd_expsumlogp, rev_expsumlogp, sa)
                #calculate number of std. deviations
                best = (None, None, np.inf, 0, np.inf, np.inf)
                for (m,motif) in enumerate(combined_motifs):
                    for poi in get_pois(motif):
                        N = min(Ns[m][poi], len(mu)-1)
                        stddevs = (means[m][poi] - mu[N]) / sigma[N]
                        if stddevs < best[4]:
                            if stddevs < thr:
                                if ambiguity_thr is not None:
                                    test_passed = test_ambiguous(motif, poi, sa, max_motif_len, mu, sigma, fwd_expsumlogp, rev_expsumlogp, ambiguity_thr, ambiguity_quantile=ambiguity_quantile, min_N=ambiguity_cov)
                                else:
                                    test_passed = True
                                if test_passed:
                                    best = (motif, poi, means[m][poi], Ns[m][poi], stddevs, norm().cdf(stddevs))
                #logging.getLogger('comos').debug("comparison:", best[4], max(d.iloc[i].stddevs, d.iloc[j].stddevs))
                if best[4] < thr:
                    new_motif = pd.Series(best, index=d.columns)
                    logging.getLogger('comos').debug(f"combined {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f}) and {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f}) to {best[0]}:{best[1]} ({best[3]}, {best[4]:.2f})")
                    # note which motifs get absorbed
                    #if d.iloc[i].motif in absorbed:
                    #    absorbed[new_motif.motif].extend(absorbed[d.iloc[i].motif])
                    #if d.iloc[j].motif in absorbed:
                    #    absorbed[new_motif.motif].extend(absorbed[d.iloc[j].motif])
                    absorbed[new_motif.motif] = [(d.iloc[i].motif, d.iloc[i].stddevs), 
                                                 (d.iloc[j].motif, d.iloc[j].stddevs)]
                    # drop / replace entries
                    d.iloc[i] = new_motif
                    d = d.drop(d.index[j])
                    changed = True
                    break
            if changed:
                break
    # remove reverse complements of non-palindromic motifs
    # keep the ones with better score
    d['representative'] = d['motif']
    changed = True
    tested = set()
    while changed == True:
        changed = False
        for i in range(0,len(d)):
            for j in range(i+1,len(d)):
                if (d.iloc[i].motif, d.iloc[j].motif) in tested:
                    continue
                tested.add((d.iloc[i].motif, d.iloc[j].motif))
                if d.iloc[i].motif == reverse_complement(d.iloc[j].motif):
                    if d.iloc[i].stddevs <= d.iloc[j].stddevs:
                        logging.getLogger('comos').debug(f"flagged {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f}) as reverse complement of {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f})")
                        #d = d.drop(d.index[j])
                        d.iloc[j, d.columns.get_loc('representative')] = d.iloc[i].motif
                        changed = True
                        break
                    else:
                        logging.getLogger('comos').debug(f"flagged {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f}) as reverse complement of {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f})")
                        #d = d.drop(d.index[i])
                        d.iloc[i, d.columns.get_loc('representative')] = d.iloc[j].motif
                        changed = True
                        break
            if changed:
                break
    return d.sort_values('stddevs').reset_index(drop=True), absorbed

def main(args):
    tstart = time.time()
    seq, sa, contig_id, n_contigs = parse_largest_contig(args.genome, args.cache)
    tstop = time.time()
    logging.getLogger('comos').info(f"{'Loaded' if args.cache else 'Created'} Index in {tstop - tstart:.2f} seconds")
    if n_contigs > 1: # TODO: analyze all contigs
        logging.getLogger('comos').warning(
            f"Dataset contains {n_contigs} contig(s), "\
            f"only {contig_id} of length {len(seq)} is analyzed")
    else:
        logging.getLogger('comos').info(f"Analyzing contig {contig_id} of length {len(seq):,} bp")
    tstart = time.time()
    df = parse_diff_file(args.rds, contig_id)
    tstop = time.time()
    logging.getLogger('comos').info(f"Parsed RDS file in {tstop - tstart:.2f} seconds")
    
    fwd_expsumlogp, rev_expsumlogp = compute_expsumlogp(df, seq)
    
    canon_motifs = get_seq_index_combinations(args.min_k, args.max_k, args.min_g, args.max_g, ['A', 'C'], 0)
    all_canon_motifs = list(canon_motifs.keys())
    logging.getLogger('comos').info(f"Analyzing {len(canon_motifs):,} canonical motifs and "\
        f"{sum([len(canon_motifs[i]) for i in canon_motifs]):,} indices within these motifs")
    # do computation
    cache_fp = os.path.join(args.cache, f"{contig_id}_k{args.min_k}-{args.max_k}_g{args.min_g}-{args.max_g}.pkl")
    if os.path.exists(cache_fp) and args.cache:
        tstart = time.time()
        with open(cache_fp, 'rb') as f:
            means, means_counts = pickle.load(f)
        tstop = time.time()
        logging.getLogger('comos').info(f"Loaded canonical motif scores from cache in {tstop - tstart:.2f} seconds")
    else:
        tstart = time.time()
        means, means_counts = suffix_array.motif_means(all_canon_motifs, args.max_k + args.max_g, fwd_expsumlogp, rev_expsumlogp, sa)
        tstop = time.time()
        logging.getLogger('comos').info(f"Computed canonical motif scores in {tstop - tstart:.2f} seconds")
        if args.cache:
            with open(cache_fp, 'wb') as f:
                pickle.dump((means, means_counts), f)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        best = np.nanmin(means, axis=1)
    mask = ~np.isnan(best)
    results = pd.DataFrame(best, columns=['val'], index=all_canon_motifs)[mask]
    results['poi'] =  np.nanargmin(means[mask], axis=1)
    results['N'] = means_counts[mask][range(results['poi'].shape[0]), results['poi']]
    results = results[['poi', 'val', 'N']] # change order of columns to match that in function reduce_motifs

    tstart = time.time()
    mu, sigma = normal_approx(results)
    tstop = time.time()
    logging.getLogger('comos').info(f"Performed normal approximation in {tstop - tstart:.2f} seconds")

    results['stddevs'] = (results.val - mu[results.N]) / sigma[results.N]
    results['p-value'] = norm().cdf(results['stddevs'])
    results = results.reset_index().rename(columns={'index':'motif'})

    if args.test_motifs:
        means, Ns = suffix_array.motif_means(args.test_motifs, np.max([len(m) for m in args.test_motifs]), fwd_expsumlogp, rev_expsumlogp, sa)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            best = np.nanmin(means, axis=1)
        mask = ~np.isnan(best)
        res = pd.DataFrame(best, columns=['val'], index=args.test_motifs)[mask]
        res['poi'] =  np.nanargmin(means[mask], axis=1)
        res['poi'] = res['poi'].astype('int32')
        res['N'] = Ns[mask][range(res['poi'].shape[0]), res['poi']]
        res = res[['poi', 'val', 'N']]
        res['stddevs'] = (res.val - mu[np.clip(res.N, 1, len(mu)-1)]) / sigma[np.clip(res.N, 1, len(mu)-1)]
        res['p-value'] = norm().cdf(res['stddevs'])

        for m,row in res.iterrows():
            print(f"\nAmbiguous-Test results for motif {m}:{int(row.poi)}")
            passed = test_ambiguous(m, int(row.poi), sa, len(m), mu, sigma, fwd_expsumlogp, rev_expsumlogp, ambiguity_thr=-args.ambiguity_thr, ambiguity_quantile=args.ambiguity_quantile, min_N=args.ambiguity_cov, debug=True)
        res = res.reset_index().rename(columns={'index':'motif'})
        print(res)
        exit()

    if args.plot:
        plot_motif_scores(results, mu, sigma, thr=args.selection_thr, savepath=os.path.join(args.out, f"scores_scatterplot.png"))

    if args.analysis_mode is not None:
        dev_fp = "dev.pkl"
        if os.path.exists(dev_fp):
            with open(dev_fp, "rb") as f:
                thresholds,motifs_dfs,absorbed = pickle.load(f)
        else:
            results = results.sort_values('stddevs')
            thresholds = []
            motifs_dfs = []
            absorbed = {}
            motifs = pd.DataFrame([], columns=['motif', 'poi', 'val', 'N', 'stddevs', 'p-value'])
            #sel = (results.stddevs <= -args.analysis_mode)
            #for _,row in tqdm(results.loc[sel].iterrows(), total=results.loc[sel].shape[0]):
            min_stddev = motifs['stddevs'].min()
            for sel_thr in tqdm(np.arange(np.round(min_stddev, 1), -(args.analysis_mode-0.001), 0.1)):
                motifs = motifs[['motif', 'poi', 'val', 'N', 'stddevs', 'p-value']]
                for _,row in results.loc[(results.stddevs > thresholds[-1]) & (results.stddevs <= sel_thr)].iterrows():
                    motifs.loc[len(motifs.index)] = row[['motif', 'poi', 'val', 'N', 'stddevs', 'p-value']]
                thresholds.append(sel_thr)
                motifs, absorbed = reduce_motifs(
                    motifs, 
                    sa, args.max_k + args.max_g, 
                    mu, sigma, 
                    fwd_expsumlogp, rev_expsumlogp,
                    thr=sel_thr, #thr=row.stddevs, 
                    ambiguity_thr=-args.ambiguity_thr,
                    ambiguity_quantile=args.ambiguity_quantile, 
                    ambiguity_cov=-args.ambiguity_cov,
                    absorbed=absorbed)
                motifs['n_canonical'] = motifs['motif'].apply(get_num_absorbed, absorbed=absorbed)
                thresholds.append(row.stddevs)
                motifs_dfs.append(motifs.copy())
            with open(dev_fp, "wb") as f:
                pickle.dump((thresholds,motifs_dfs,absorbed), f)
        breakpoint()

    
    sel = (results.stddevs <= -args.selection_thr)
    logging.getLogger('comos').info(f"Selected {sel.sum():,} motifs based on selection threshold of {args.selection_thr} std. deviations:")
    print(results.loc[sel])
    tstart = time.time()
    motifs_found, absorbed = reduce_motifs(
        results.loc[sel], 
        sa, args.max_k + args.max_g, 
        mu, sigma, 
        fwd_expsumlogp, rev_expsumlogp,
        thr=-args.selection_thr, 
        ambiguity_thr=-args.ambiguity_thr,
        ambiguity_quantile=args.ambiguity_quantile, 
        ambiguity_cov=-args.ambiguity_cov,)
    tstop = time.time()
    logging.getLogger('comos').info(f"Performed motif reduction in {tstop - tstart:.2f} seconds")
    motifs_found['n_canonical'] = motifs_found['motif'].apply(get_num_absorbed, absorbed=absorbed)
    motifs_found = motifs_found[['motif', 'poi', 'representative', 'val', 'N', 'stddevs', 'p-value', 'n_canonical']]
    logging.getLogger('comos').info(f"Reduced to {len(motifs_found)} motifs in {len(motifs_found.representative.unique())} MTase groups:")
    print(motifs_found.set_index(['representative', 'motif']))
    
    if args.plot:
        fwd_diff, rev_diff = compute_diffs(df, seq)
        for r,row in motifs_found.loc[motifs_found.motif == motifs_found.representative].iterrows():
            plot_diffs_with_complementary(row.motif, sa, fwd_diff, rev_diff, savepath=os.path.join(args.out, f"{row.motif}_{row.poi}.png"))

    motifs_found.to_csv(os.path.join(args.out, f"results.csv"))
    

if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, 
                        format=f'%(levelname)s [%(asctime)s] : %(message)s',
                        datefmt='%H:%M:%S')
    main(args)