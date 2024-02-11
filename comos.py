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
import networkx as nx

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
        "A" : {"A":"A", "C":"M", "G":"R", "T":"W", "M":"M", "R":"R", "W":"W", "S":"V", "Y":"H", "K":"D", "V":"V", "H":"H", "D":"D", "B":"N", "N":"N", " ":"A",},
        "C" : {"A":"M", "C":"C", "G":"S", "T":"Y", "M":"M", "R":"V", "W":"H", "S":"S", "Y":"Y", "K":"B", "V":"V", "H":"H", "D":"N", "B":"B", "N":"N", " ":"C",},
        "G" : {"A":"R", "C":"S", "G":"G", "T":"K", "M":"V", "R":"R", "W":"D", "S":"S", "Y":"B", "K":"K", "V":"V", "H":"N", "D":"D", "B":"B", "N":"N", " ":"G",},
        "T" : {"A":"W", "C":"Y", "G":"K", "T":"T", "M":"H", "R":"D", "W":"W", "S":"B", "Y":"Y", "K":"K", "V":"N", "H":"H", "D":"D", "B":"B", "N":"N", " ":"T",},
        "M" : {"A":"M", "C":"M", "G":"V", "T":"H", "M":"M", "R":"V", "W":"H", "S":"V", "Y":"H", "K":"N", "V":"V", "H":"H", "D":"N", "B":"N", "N":"N", " ":"M",},
        "R" : {"A":"R", "C":"V", "G":"R", "T":"D", "M":"V", "R":"R", "W":"D", "S":"V", "Y":"N", "K":"D", "V":"V", "H":"N", "D":"D", "B":"N", "N":"N", " ":"R",},
        "W" : {"A":"W", "C":"H", "G":"D", "T":"W", "M":"H", "R":"D", "W":"W", "S":"N", "Y":"H", "K":"D", "V":"N", "H":"H", "D":"D", "B":"N", "N":"N", " ":"W",},
        "S" : {"A":"V", "C":"S", "G":"S", "T":"B", "M":"V", "R":"V", "W":"N", "S":"S", "Y":"B", "K":"B", "V":"V", "H":"N", "D":"N", "B":"B", "N":"N", " ":"S",},
        "Y" : {"A":"H", "C":"Y", "G":"B", "T":"Y", "M":"H", "R":"N", "W":"H", "S":"B", "Y":"Y", "K":"B", "V":"N", "H":"H", "D":"N", "B":"B", "N":"N", " ":"Y",},
        "K" : {"A":"D", "C":"B", "G":"K", "T":"K", "M":"N", "R":"D", "W":"D", "S":"B", "Y":"B", "K":"K", "V":"N", "H":"N", "D":"D", "B":"B", "N":"N", " ":"K",},
        "V" : {"A":"V", "C":"V", "G":"V", "T":"N", "M":"V", "R":"V", "W":"N", "S":"V", "Y":"N", "K":"N", "V":"V", "H":"N", "D":"N", "B":"N", "N":"N", " ":"V",},
        "H" : {"A":"H", "C":"H", "G":"N", "T":"H", "M":"H", "R":"N", "W":"H", "S":"N", "Y":"H", "K":"N", "V":"N", "H":"H", "D":"N", "B":"N", "N":"N", " ":"H",},
        "D" : {"A":"D", "C":"N", "G":"D", "T":"D", "M":"N", "R":"D", "W":"D", "S":"N", "Y":"N", "K":"D", "V":"N", "H":"N", "D":"D", "B":"N", "N":"N", " ":"D",},
        "B" : {"A":"N", "C":"B", "G":"B", "T":"B", "M":"N", "R":"N", "W":"N", "S":"B", "Y":"B", "K":"B", "V":"N", "H":"N", "D":"N", "B":"B", "N":"N", " ":"B",},
        "N" : {"A":"N", "C":"N", "G":"N", "T":"N", "M":"N", "R":"N", "W":"N", "S":"N", "Y":"N", "K":"N", "V":"N", "H":"N", "D":"N", "B":"N", "N":"N", " ":"N",},
        " " : {"A":"A", "C":"C", "G":"G", "T":"T", "M":"M", "R":"R", "W":"W", "S":"S", "Y":"Y", "K":"K", "V":"V", "H":"H", "D":"D", "B":"B", "N":"N", " ":" ",}
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

IUPAC_NOT = {
    "A" : "B",
    "C" : "D",
    "G" : "H",
    "T" : "V",
    "M" : "K",
    "R" : "Y",
    "W" : "S",
    "S" : "W",
    "Y" : "R",
    "K" : "M",
    "V" : "T",
    "H" : "G",
    "D" : "C",
    "B" : "A",
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
        '--min-cov',
        type=int,
        default=10,
        help="Minimum coverage for a genomic position to be considered in the analysis."
    )
    hyper_grp.add_argument(
        '--metric',
        choices=["meanabsdiff", "expsumlogp"],
        default="meanabsdiff",
        help="Which metric to compute for each motif site."
    )
    hyper_grp.add_argument(
        '--window',
        type=int,
        default=5,
        help="Sequence window size around analyzed motif position used for calculating the motif metric."
    )
    hyper_grp.add_argument(
        '--min-window-values',
        type=int,
        default=3,
        help="Minimum positions with sufficient coverage inside window."
    )
    hyper_grp.add_argument(
        '--subtract-background',
        action="store_true",
        help="Compute background based on adjacent windows to the left and right of the window and substract the more conservative from metric values."
    )
    hyper_grp.add_argument(
        '--aggregate',
        choices=["median", "mean"],
        default="median",
        help="How to aggregate the motif metrics over all motif sites."
    )
    hyper_grp.add_argument(
        '--selection-thr',
        type=float,
        default=4.0,
        help='Minimum diversion from Null-model in number of std.dev. required for a motif to get selected.' 
    )
    hyper_grp.add_argument(
        '--ambiguity-thr',
        type=float,
        default=2.0,
        help='Minimum diversion from Null-model in number of std.dev. required for an extended motif.' 
    )
    hyper_grp.add_argument(
        '--ambiguity-min-sites',
        type=float,
        default=10,
        help='Minimum coverage required to consider a motif in ambiguity testing.' 
    )
    hyper_grp.add_argument(
        '--ambiguity-min-tests',
        type=int,
        default=2,
        help="Minimum number of test cases per ambiguous position (1-4) with sufficient sites, otherwise test fails."
    )

    misc_grp = parser.add_argument_group('Misc')
    misc_grp.add_argument(
        '--test-motifs',
        nargs='+',
        help='Analyze the given IUPAC motifs.' 
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

    # check hyperparameters
    if args.ambiguity_thr > args.selection_thr:
        logging.getLogger('comos').warning(f"Ambiguity threshold is set to selection threshold since it should not be larger.")
        args.ambiguity_thr = args.selection_thr

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

def compute_expsumlogp(df, seq, min_cov, window, min_window_values, subtract_background=False):
    fwd_logp = np.full(len(seq), np.nan)
    sel = (df.dir == 'fwd') & ~pd.isna(df.u_test_pval) & (df.N_wga >= min_cov) & (df.N_nat >= min_cov)
    fwd_logp[df.loc[sel].index] = \
        -np.log10(df.loc[sel, 'u_test_pval'])
    rev_logp = np.full(len(seq), np.nan)
    sel = (df.dir == 'rev') & ~pd.isna(df.u_test_pval) & (df.N_wga >= min_cov) & (df.N_nat >= min_cov)
    rev_logp[df.loc[sel].index] = \
        -np.log10(df.loc[sel, 'u_test_pval'])
    fwd_expsumlogp = 10**(-pd.Series(fwd_logp).rolling(window, center=True, min_periods=min_window_values).sum())
    rev_expsumlogp = 10**(-pd.Series(rev_logp).rolling(window, center=True, min_periods=min_window_values).sum())
    if subtract_background:
        dist = round(window / 2 + 0.5)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            fwd_background = np.nanmax(np.column_stack([np.pad(fwd_expsumlogp[dist:], (0,dist), constant_values=np.nan), 
                                                        np.pad(fwd_expsumlogp[:-dist], (dist,0), constant_values=np.nan)]), axis=1)
            rev_background = np.nanmax(np.column_stack([np.pad(rev_expsumlogp[dist:], (0,dist), constant_values=np.nan), 
                                                        np.pad(rev_expsumlogp[:-dist], (dist,0), constant_values=np.nan)]), axis=1)
        return fwd_expsumlogp + fwd_background, rev_expsumlogp + rev_background
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

def compute_meanabsdiff(df, seq, min_cov, window, min_window_values, subtract_background=False):
    fwd_diff, rev_diff = compute_diffs(df, seq, min_cov=min_cov)
    fwd_mean_abs_diff = pd.Series(np.abs(fwd_diff)).rolling(window, center=True, min_periods=min_window_values).mean()
    rev_mean_abs_diff = pd.Series(np.abs(rev_diff)).rolling(window, center=True, min_periods=min_window_values).mean()
    if subtract_background:
        dist = round(window / 2 + 0.5)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            fwd_background = np.nanmin(np.column_stack([np.pad(fwd_mean_abs_diff[dist:], (0,dist), constant_values=np.nan), 
                                                        np.pad(fwd_mean_abs_diff[:-dist], (dist,0), constant_values=np.nan)]), axis=1)
            rev_background = np.nanmin(np.column_stack([np.pad(rev_mean_abs_diff[dist:], (0,dist), constant_values=np.nan), 
                                                        np.pad(rev_mean_abs_diff[:-dist], (dist,0), constant_values=np.nan)]), axis=1)
        return fwd_mean_abs_diff - fwd_background, rev_mean_abs_diff - rev_background
    return fwd_mean_abs_diff, rev_mean_abs_diff

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
                    gap_positions = [k//2]
                else:
                    gap_positions = [k//2, (k//2)+1]
                for gap_pos in gap_positions:
                    if not (('C' in comb[:gap_pos] and 'G' in comb[gap_pos:]) or ('G' in comb[:gap_pos] and 'C' in comb[gap_pos:]) or \
                            ('A' in comb[:gap_pos] and 'T' in comb[gap_pos:]) or ('T' in comb[:gap_pos] and 'A' in comb[gap_pos:])):
                        continue
                    comb_ = comb[:gap_pos] + ["N"]*gap + comb[gap_pos:]
                    indexes_ = get_indexes(comb_, k+gap)
                    seq_ = "".join(comb_)
                    seq_index_combinations[seq_] = indexes_
    return seq_index_combinations

def plot_context_dependent_differences(motif, poi, sa, fwd_diff, rev_diff, ymax=np.inf, pad=2, absolute=False, savepath=None):
    poi = int(poi)
    mlen = len(motif)

    #mean_fwd_diff = pd.Series(np.abs(fwd_diff) - np.nanmedian(np.abs(fwd_diff))).rolling(5, center=True, min_periods=3).sum()
    #mean_rev_diff = pd.Series(np.abs(rev_diff) - np.nanmedian(np.abs(rev_diff))).rolling(5, center=True, min_periods=3).sum()
    #median_mean, Ns_median_mean = suffix_array.motif_medians([motif], len(motif), mean_fwd_diff, mean_rev_diff, sa)
    #medians_rev, Ns_med_rev = suffix_array.motif_medians([reverse_complement(motif)], len(motif), mean_fwd_diff, mean_rev_diff, sa)

    motif_padded = "N"*max(0, pad - poi) + motif + "N"*max(0,pad-(mlen-poi-1))
    poi_ = poi + max(0, pad - poi)
    positions = list(range(poi_-pad, poi_+pad+1))
    xtick_labels = motif_padded[positions[0]:positions[-1]+1]
    bases = []
    for i in range(len(motif_padded)):
        if i in positions:
            bases.append(IUPAC_TO_LIST[motif_padded[i]])
        else:
            bases.append(list(motif_padded[i]))
    expanded_motifs = ["".join(p) for p in itertools.product(*bases)]

    medians_fwd, Ns_med_fwd = suffix_array.all_motif_medians(expanded_motifs, len(expanded_motifs[0]), fwd_diff, rev_diff, sa, pad=0)
    medians_fwd = medians_fwd[:, positions]
    Ns_med_fwd = Ns_med_fwd[:, positions]

    positions = list(range(poi-pad, poi+pad+1))
    ind_fwd, ind_rev = suffix_array.find_motif(motif, sa)
    # original strand
    n_fwd, n_rev = len(ind_fwd), len(ind_rev)
    data_fwd = []
    for i in positions:
        fwd_diffs = fwd_diff[ind_fwd + i]
        rev_diffs = rev_diff[ind_rev - i]
        diffs = np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]])
        data_fwd.append(diffs)
    
    fig,ax = plt.subplots()
    ax.grid(axis="y", zorder=-100)
    boxprops = dict(linewidth=2.0, color='black')
    whiskerprops = dict(linewidth=2.0, color='black')
    medianprops = dict(linewidth=2.0, color='black', linestyle=":")
    if absolute is False:
        ax.boxplot(data_fwd, boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)
    #elif absolute == "both":
    #    ax.boxplot(data_fwd)
    #    ax.boxplot([np.abs(d) for d in data_fwd])
    else:
        ax.boxplot([np.abs(d) for d in data_fwd], boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)
    
    for i in range(len(expanded_motifs)):
        if absolute is False:
            ax.plot(range(1, len(positions)+1), medians_fwd[i], zorder=-50, alpha=0.5)
        else:
            ax.plot(range(1, len(positions)+1), np.abs(medians_fwd[i]), zorder=-50, alpha=0.5)
    
    if absolute:
        max_ylim = (0, 
                    min(ax.get_ylim()[1], ymax))
    else:
        max_ylim = (max(min(ax.get_ylim()[0], -ax.get_ylim()[1]), -ymax), 
                    min(max(ax.get_ylim()[1], -ax.get_ylim()[0]), ymax))
    ax.set_ylim(max_ylim)
    ax.set_xticklabels(xtick_labels)
    ax.set_title(f"{motif}:{poi}")

    if savepath:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

def plot_diffs_with_complementary(motif, sa, fwd_diff, rev_diff, ymax=np.inf, offset=3, savepath=None, absolute=True):
    mlen = len(motif)
    positions = list(range(-offset, mlen+offset))

    expl_motifs = explode_motif(motif)
    group_width = 0.75
    width = group_width / len(expl_motifs)
    offsets = np.arange(width/2, group_width+width/2, width) - group_width / 2
    width *= 0.8

    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.3)
    ind_fwd, ind_rev = suffix_array.find_motif(motif, sa)
    n_fwd, n_rev = len(ind_fwd), len(ind_rev)
    ax1.set_title(f"{motif}, N={n_fwd+n_rev:,}")
    boxes, labels = [], []
    for c, (m, offset) in enumerate(zip(expl_motifs, offsets)):
        ind_fwd, ind_rev = suffix_array.find_motif(m, sa)
        # original strand
        n_fwd, n_rev = len(ind_fwd), len(ind_rev)
        data_fwd = []
        for i in positions:
            fwd_diffs = fwd_diff[ind_fwd + i]
            rev_diffs = rev_diff[ind_rev - i]
            if absolute:
                diffs = np.abs(np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]]))
            else:
                diffs = np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]])
            data_fwd.append(diffs)
        # complementary strand
        n_fwd_c, n_rev_c = len(ind_fwd), len(ind_rev)
        data_rev = []
        for i in positions:
            fwd_diffs = rev_diff[ind_fwd + i]
            rev_diffs = fwd_diff[ind_rev - i]
            if absolute:
                diffs = np.abs(np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]]))
            else:
                diffs = np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]])
            data_rev.append(diffs)
        boxes.append(
            ax1.boxplot(data_fwd, positions=np.array(positions)+offset, widths=width, 
                flierprops={'marker': 'o', 'markersize': 3, 'alpha':0.5}, 
                patch_artist=True, 
                boxprops=dict(facecolor=f"C{c}"), 
                medianprops=dict(color="black",linewidth=1.5))
            )
        ax2.boxplot(data_rev, positions=np.array(positions)+offset, widths=width, 
            flierprops={'marker': 'o', 'markersize': 3, 'alpha':0.5}, 
            patch_artist=True, 
            boxprops=dict(facecolor=f"C{c}"), 
            medianprops=dict(color="black",linewidth=1.5))
        labels.append(f"{m}, n={n_fwd+n_rev:>4,}")

    ax1.legend([boxes[i]["boxes"][0] for i in range(len(boxes))], labels, loc='upper right', prop={'family': 'monospace', 'size':10})
    ax1.set_xticks(positions)
    ax1.set_xticklabels([motif[i] if 0 <= i < mlen else "" for i in positions])
    ax2.set_xticks(positions)
    ax2.set_xticklabels([motif[i].translate(comp_trans) if 0 <= i < mlen else "" for i in positions])
    ax2.xaxis.tick_top()
    ax1.set(ylabel=r"(+) strand abs. diff / $pA$")
    ax2.set(ylabel=r"(-) strand abs. diff / $pA$")
    if absolute:
        max_ylim = (0, 
                    min(max(ax1.get_ylim()[1], ax2.get_ylim()[1]), ymax))
    else:
        max_ylim = (max(min(ax1.get_ylim()[0], ax2.get_ylim()[0]), -ymax), 
                    min(max(ax1.get_ylim()[1], ax2.get_ylim()[1]), ymax))
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
    
    # piecewise linear interpolation
    return np.interp(np.array(range(0,d.N.max()+1)), N, mu), np.interp(np.array(range(0,d.N.max()+1)), N, sigma)

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
    return p_mu(np.array(range(0,d.N.max()+1))), p_sigma(np.array(range(0,d.N.max()+1)))

def plot_motif_scores(results, mu, sigma, thr=6., savepath=None):
    fig,ax = plt.subplots(figsize=(6,6))
    ax.scatter(results.N,results.val,s=1,alpha=0.25, color='black')
    X = np.linspace(0,5000, 1000)
    ax.plot(X, mu[X.astype(int)], color='C0')
    ax.plot(X, mu[X.astype(int)] + thr*sigma[X.astype(int)], color='C0', linestyle=':')
    #ax.plot(X, mu[X.astype(int)] - thr*sigma[X.astype(int)], color='C0', linestyle=':')
    ax.grid()
    ax.set(ylim=(-0.0005, ax.get_ylim()[1]),
           xlim=(-50,5000),
        xlabel="# motif sites", ylabel="10**(sumlog p)")
    if savepath:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

def motif_contains(m1, m2):
    """
    Checks if IUPAC motif m2 is identically contained in IUPAC motif m1.
    Returns a tuple (idx, ident, diff), where
    idx : first index of m2 in m1 if m1 is contained in m2 and None if it is not contained
    ident : True if the substring of m1 at idx of length len(m2) is identical to m2
    diff : non-identical positions between m1 and m2 relative to m2 if m2 is contained in m1
    
    Examples:
    ATC is identically contained in CATC at index 1, diff=[-1]
    ATC is non-identically contained in CATS at index 1, diff=[-1, 2]
    CAT is identically contained in CATS at index 0, diff=[3]
    ATN is not contained in CATS
    """
    lm1, lm2 = len(m1), len(m2)
    if lm2 > len(m1):
        return None, False, None
    for i in range(lm1 - lm2 + 1):
        identical = True
        diff = [j for j in range(-i,0)]
        for j in range(lm2):
            if m1[i+j] != m2[j]:
                identical = False
                #diff += 1
                diff.append(j)
            if m2[j] != 'N':
                if m1[i+j] == 'N':
                    # do not match agaist TypeI motif gaps
                    break
                if m1[i+j] != IUPAC_TO_IUPAC[m1[i+j]][m2[j]]:
                    break
        else:
            return i, identical, diff + [j for j in range(lm2,lm1-i)]
    return None, False, None

def motif_diff(m1, m2, m2_idx):
    """
    Assumes m1 contains m2 at index m2_idx.
    """
    lm1, lm2 = len(m1), len(m2)
    diff = []
    for i in range(lm1):
        if i < m2_idx or i >= m2_idx + lm2 or m2[i - m2_idx] == 'N':
            if m1[i] == 'N':
                diff.append('N')
            else:
                diff.append(IUPAC_NOT[m1[i]])
        else:
            # use m1 bases at overlap positions. 
            # These might not be identical to m2, e.g. ASST instead of ACGT
            diff.append(m1[i])
    return "".join(diff).strip('N') 

def motif_diff_multi(mshrt, mlong_and_idx):
    to_combine = []
    padding = [0,0]
    for mlong, mshrt_idx in mlong_and_idx:
        padding[0] = max(padding[0], mshrt_idx)
        padding[1] = max(padding[1], len(mlong) - len(mshrt) - mshrt_idx)
    #print(padding)
    for mlong, mshrt_idx in mlong_and_idx:
        to_combine.append(" "*(padding[0] - mshrt_idx) + mlong.replace("N", " ") + " "*(padding[1] - (len(mlong) - len(mshrt) - mshrt_idx)))
    #print(to_combine)
    combined_motif = to_combine[0]
    for m in to_combine[1:]:
        combined_motif = combine_IUPAC(combined_motif, m)
    combined_motif = combined_motif.replace(" ", "N")
    #print(combined_motif)
    
    return motif_diff(combined_motif, mshrt, padding[0])

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

def test_ambiguous(motif, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=30, debug=False, min_tests=2):
    if "NNN" in motif: # Type I
        # only check flanking Ns
        callback = lambda pat: pat.group(1)+pat.group(2).lower()+pat.group(3)
        motif = re.sub(r"(N)(N+)(N)", callback, motif)
    motif_exp = ["N"] + list(motif) + ["N"]
    to_test = []
    num_amb = 0
    for i in range(len(motif_exp)):
        if motif_exp[i] =='N': # not in "ACGTn":
            num_amb += 1
            for b in IUPAC_TO_LIST[motif_exp[i]]:
                testcase = "".join(motif_exp[:i] + [b] + motif_exp[i+1:]).strip('N').upper()
                to_test.append(testcase)
    aggregates, Ns = aggr_fct(to_test, max_motif_len+1, fwd_metric, rev_metric, sa)
    # shift the data of the first four testcases to the left, because they have an additional leading base
    for i in range(4):
        aggregates[i, :-1] = aggregates[i, 1:]
        aggregates[i, -1] = np.nan 
        Ns[i, :-1] = Ns[i, 1:]
        Ns[i, -1] = 0
    no_sites = np.all(Ns == 0, axis=1)
    pois = np.all(~np.isnan(aggregates[~no_sites]), axis=0)
    if np.all(~pois):
        # no single C/A site with sufficient coverage
        return False
    # instead of taking min over all array (best),
    # take max over columns (repr. poi) and min over those
    Ns = np.clip(Ns, 1, len(mu)-1)
    stddevs = (aggregates - mu[Ns]) / sigma[Ns]
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
        print(pd.DataFrame(stddevs[~no_sites][:, pois].round(2),
                           index=np.array(to_test__)[~no_sites],
                           columns=pois_idx))
    #if np.sum(~np.isnan(stddevs[~no_sites][:, pois]), axis=0).min() < min_tests:
    #    # not enough test cases with sufficient coverage
    #    return False
    if np.any((~np.isnan(stddevs[:, pois])).reshape(4,stddevs.shape[0] // 4,-1).sum(axis=1).min(axis=1) < min_tests):
        # not enough test cases with sufficient coverage for at least one ambiguous position
        return False
    #max_stddev_per_poi = np.nanquantile(stddevs[~no_sites][:, pois], ambiguity_quantile, axis=0) # nanquantile alone not sufficient because we want to exclude columns e.g. with single entries
    if opt_dir == 'min':
        max_stddev_per_poi = np.nanmax(stddevs[~no_sites][:, pois], axis=0)
        best_poi = np.argmin(max_stddev_per_poi) # careful: best_pois refers to selection of pois, not a motif index!
    else:
        max_stddev_per_poi = np.nanmin(stddevs[~no_sites][:, pois], axis=0)
        best_poi = np.argmax(max_stddev_per_poi)
    if debug:
        print('best column:', best_poi, '= motif index', np.searchsorted(np.cumsum(pois), best_poi+1))
    if opt_dir == 'min':
        return max_stddev_per_poi[best_poi] < ambiguity_thr
    else:
        return max_stddev_per_poi[best_poi] > ambiguity_thr

def explode_motif(motif):
    exploded_motifs = IUPAC_TO_LIST[motif[0]][:]
    for i in range(1,len(motif)):
        exploded_motifs_ = []
        for m in exploded_motifs:
            if motif[i] == 'N':
                exploded_motifs_.append(m + 'N')
                continue
            for b in IUPAC_TO_LIST[motif[i]]:
                exploded_motifs_.append(m + b)
        exploded_motifs = exploded_motifs_
    return exploded_motifs

def test_exploded(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, selection_thr, min_N=30, debug=False):
    to_test = explode_motif(motif)
    if len(to_test) == 1:
        return True
    
    aggregates, Ns = aggr_fct(to_test, max_motif_len+1, fwd_metric, rev_metric, sa)
    no_sites = np.all(Ns == 0, axis=1)
    pois = np.all(~np.isnan(aggregates[~no_sites]), axis=0)
    # instead of taking min over all array (best),
    # take max over columns (repr. poi) and min over those
    Ns = np.clip(Ns, 1, len(mu)-1)
    stddevs = (aggregates - mu[Ns]) / sigma[Ns]
    stddevs[Ns < min_N] = np.nan # mask sites with low coverage
    if opt_dir == "min":
        max_stddev_per_poi = np.nanmax(stddevs[~no_sites][:, pois], axis=0)
        best_poi = np.argmin(max_stddev_per_poi) # careful: best_pois refers to selection of pois, not a motif index!
    else:
        max_stddev_per_poi = np.nanmin(stddevs[~no_sites][:, pois], axis=0)
        best_poi = np.argmax(max_stddev_per_poi) # careful: best_pois refers to selection of pois, not a motif index!
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
        print('best column:', best_poi, '= motif index', np.searchsorted(np.cumsum(pois), best_poi+1))
    if opt_dir == "min":
        return max_stddev_per_poi[best_poi] < selection_thr
    else:
        return max_stddev_per_poi[best_poi] > selection_thr

def get_pois(motif):
    pois = []
    for i in range(len(motif)):
        if motif[i] in "CA":
            pois.append(i)
    return pois

def prune_edges(G, edges, d):
    pruned = []
    nodes = []
    while edges:
        for u,v in edges:
            G.remove_edge(u,v)
            if v not in nodes:
                nodes.append(v)
        edges = []
        for v in nodes:
            if G.in_degree(v) == 0:
                for edge in G.out_edges(v):
                    edges.append(edge)
                #G.remove_node(v)
                #pruned.append(v)
        #nodes = []
    for v in nodes:
        if G.in_degree(v) == 0:
            G.remove_node(v)
            pruned.append(v)
        else:
            d.loc[v, 'to_prune'] = True
    return pruned

def resolve_motif_graph(sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr):
    draw = False
    #if 342 in sG:
    #    breakpoint()
    #    draw = 342
    if draw:
        print(d.loc[list(sG)].sort_index())
        fig,ax = plt.subplots(1,1,figsize=(6,4))
        pos = nx.spring_layout(sG)
        nx.draw_networkx(sG, pos=pos, ax=ax)
        nx.draw_networkx(sG, nodelist=d.loc[d.index.isin(list(sG)) & ~d['pass']].index, edgelist=[], node_color="red", pos=pos, ax=ax)
        if draw in sG:
            nx.draw_networkx(sG, nodelist=[draw], edgelist=sG.in_edges(draw), node_color="green", edge_color="green", pos=pos, ax=ax)
        plt.show(block=False)

    # resolve graph, starting with root nodes
    last_round = False # to check root nodes a last time after all edges were removed
    while True:
        root_nodes = [n for n,deg in sG.in_degree() if deg==0]
        for root_node in root_nodes:
            root_motif = d.loc[root_node, 'motif']
            # check if node was previously marked to be pruned but was not pruned yet due to previously unresolved incoming edges
            if d.loc[root_node, 'to_prune'] == True:
                pruned = prune_edges(sG, list(sG.out_edges(root_node)), d)
                d = d.drop(pruned)
                logging.getLogger('comos').debug(f"pruned previously marked root node {d.loc[root_node, 'motif']} and consequently pruned {len(pruned)} other nodes")
                sG.remove_node(root_node)
                d = d.drop(root_node)
                continue

            # check if it passed ambiguity filtering
            if d.loc[root_node, 'pass'] == False:
                # drop root motif
                for u,v in list(sG.out_edges(root_node)):
                    sG.remove_edge(u,v)
                sG.remove_node(root_node)
                d = d.drop(root_node)
                continue

            # sort outgoing edges by the position that is over-specifying the shorter motif
            by_index = {}
            for edge in sG.out_edges(root_node):
                idx, ident, diff = motif_contains(d.loc[edge[1]].motif, root_motif)
                edge_data = sG.get_edge_data(*edge)
                if len(diff) != 1:
                    # at the moment, this should not be possible because each edge has edit distance 1
                    logging.getLogger('comos').error(f"unhandled exception: motifs for {root_motif} and {d.loc[edge[1], 'motif']} differ in more than one position")
                    exit(1)
                if diff[0] not in by_index:
                    by_index[diff[0]] = {"edges":[], "data":[]}
                by_index[diff[0]]['edges'].append(edge)
                by_index[diff[0]]['data'].append((d.loc[edge[1], 'motif'], idx))
            
            # for each position indipendently, determine if the shorter motif or all longer motifs shall be pruned
            drop_root_node = False
            for idx in by_index:
                m_diff = motif_diff_multi(root_motif, by_index[idx]['data'])
                if m_diff == root_motif:
                    # prune all longer motifs
                    pruned = prune_edges(sG, by_index[idx]['edges'], d)
                    d = d.drop(pruned)
                    logging.getLogger('comos').debug(f"kept root node {d.loc[root_node, 'motif']}, pruned {len(pruned)} nodes")
                else:
                    aggregates, Ns = aggr_fct([m_diff], len(m_diff), fwd_metric, rev_metric, sa)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
                        if opt_dir == "min":
                            best = np.nanmin(aggregates[0])
                        else:
                            best = np.nanmax(aggregates[0])
                    if np.isnan(best):
                        logging.getLogger('comos').error("unhandled exception: no A or C in difference motif")
                        exit(1)
                    poi = np.nanargmin(aggregates[0])
                    N = min(Ns[0][poi], len(mu)-1)
                    stddev = (best - mu[N]) / sigma[N]
                    if (opt_dir == "min" and stddev >= ambiguity_thr) or (opt_dir == "max" and stddev <= ambiguity_thr):
                        # drop root motif, keep longer ones
                        for u,v in by_index[idx]['edges']:
                            sG.remove_edge(u,v)
                        logging.getLogger('comos').info(f"{m_diff} : {stddev} --> drop root node {d.loc[root_node, 'motif']} for {[d.loc[v, 'motif'] for u,v in by_index[idx]['edges']]}")
                        drop_root_node = True
                    else:
                        # prune all longer motifs
                        pruned = prune_edges(sG, by_index[idx]['edges'], d)
                        d = d.drop(pruned)
                        logging.getLogger('comos').info(f"{m_diff} : {stddev} --> kept root node {d.loc[root_node, 'motif']}, pruned {len(pruned)} nodes")
            if drop_root_node:
                sG.remove_node(root_node)
                d = d.drop(root_node)
            if draw:
                fig,ax = plt.subplots(1,1,figsize=(6,4))
                pos = nx.spring_layout(sG)
                nx.draw_networkx(sG, pos=pos, ax=ax)
                nx.draw_networkx(sG, nodelist=d.loc[d.index.isin(list(sG)) & ~d['pass']].index, edgelist=[], node_color="red", pos=pos, ax=ax)
                if draw in sG:
                    nx.draw_networkx(sG, nodelist=[draw], edgelist=sG.in_edges(draw), node_color="green", edge_color="green", pos=pos, ax=ax)
                plt.show(block=False)
        if len(sG.edges) == 0:
            if last_round:
                break
            last_round = True
    if draw:
        breakpoint()
    return d

def reduce_motifs(d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests):
    d = d.copy().sort_values(['N','stddevs'], ascending=[False,True])
    d['typeI'] = d.motif.str.contains('NN')
    # initial filtering
    if ambiguity_thr is not None:
        d['pass'] = d.apply(lambda row: test_ambiguous(row.motif, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=ambiguity_min_sites, min_tests=ambiguity_min_tests), axis=1)
        for i,row in d.loc[d['pass'] == False].iterrows():
            logging.getLogger('comos').debug(f"excluded {row.motif}:{row.poi} ({row.N}, {row.stddevs:.2f}) : did not pass ambiguity filter")
        n_excluded = (d['pass'] == False).sum()
        if n_excluded:
            logging.info(f"Excluded {n_excluded} motifs because they did not pass ambiguity testing.")
        #d = d.loc[d['pass']].drop(['pass'], axis=1)
    logging.getLogger('comos').info(f"{len(d.loc[d['pass']])} motifs after ambiguity testing:")
    print(d.loc[d['pass']].sort_values("stddevs", ascending=(opt_dir=="min")))
    
    # removing nested canonical motifs
    d['mlen'] = d.motif.str.len()
    d['slen'] = d.motif.str.replace('N','').str.len()
    # d.motif.str.extract(r'([ACGT]+)(N*)([ACGT]*)').rename(columns={0:'up', 1:'gap',2:'down'})
    d = d.sort_values(['mlen', 'slen']).reset_index(drop=True)
    G = nx.DiGraph()
    for i in range(0,len(d)):
        G.add_node(i)
        #for j in range(i+1,len(d)):
        #    # due to sorting, motif i is ensured to be smaller or equally long than motif j
        #    if d.iloc[i].typeI != d.iloc[j].typeI:
        #        continue
        #    elif d.iloc[i].slen > d.iloc[j].slen:
        #        # ensure that number of specific bases is smaller for motif i than for motif j (only relevant for TypeI)
        #        continue
        #    # check if motif i is contained in motif j
        #    idx, ident, diff = motif_contains(d.iloc[j].motif, d.iloc[i].motif)
        #    if idx is not None:
        #        G.add_edge(i, j, weight=len(diff), diff=diff, idx=idx)
        sel = (d.index > i) & (d.typeI == d.loc[i, 'typeI']) & (0 <= d.slen - d.loc[i, 'slen']) # TODO: check if longer edges needed, if not: & (d.slen - d.loc[i, 'slen'] <= 1)
        contain_i = d.loc[sel].loc[d.loc[sel].motif.str.contains(d.loc[i].motif.replace('N','.'))]
        if not contain_i.empty:
            #G.add_edge_from(itertools.product([i], contain_i.index), weight=1)
            for j, row in contain_i.iterrows():
                #G.add_edge(i, j, weight=1, diff=diff, idx=idx)
                G.add_edge(i, j, weight=row.slen - d.loc[i, 'slen'])
    n_sGs = len([G.subgraph(c) for c in nx.connected_components(G.to_undirected()) if np.any(d.loc[list(c), 'pass'])])
    G.remove_edges_from([(f, t) for f,t,d in G.edges.data() if d['weight'] > 1])
    sGs = [G.subgraph(c).copy() for c in nx.connected_components(G.to_undirected()) if np.any(d.loc[list(c), 'pass'])]
    if len(sGs) != n_sGs:
        # TODO : handle this, e.g. by keeping "longest" paths between any two nodes?
        logging.getLogger('comos').warning(f"Unhandled case: Number of subgraphs changed from {n_sGs} to {len(sGs)} by removing edges with >1 edit distance between motifs.")
    logging.getLogger('comos').info(f"{len(sGs)} subgraphs in nested motif network")
    sGs = [sG for sG in sGs if np.any(d.loc[list(sG), 'pass'])]
    logging.getLogger('comos').info(f"{len(sGs)} subgraphs with any valid motif")

    for sG in sGs:
        # filter nodes that did not pass ambiguity filtering
        #changed = True
        #while changed:
        #    changed = False
        #    for v in sG:
        #        if not d.loc[v, 'pass']:
        #            pruned = prune_edges(sG, list(sG.out_edges(v)))
        #            sG.remove_node(v)
        #            d = d.drop(v)
        #            d = d.drop(pruned)
        #            changed = True
        #            break
        
        #edges_to_remove = set()
        #nodes_to_remove = set()
        #for v in sG:
        #    if d.loc[v, 'pass'] == False:
        #        edges_to_remove |= set(sG.out_edges(v))
        #        nodes_to_remove.add(v)
        #pruned = prune_edges(sG, list(edges_to_remove))
        #nodes_to_remove |= set(pruned)
        #sG.remove_nodes_from(nodes_to_remove)
        #d = d.drop(nodes_to_remove)
        ##print(d.loc[list(sG)])

        # filter nodes that did not pass ambiguity filtering
        #changed = True
        #while changed:
        #    changed = False
        #    for v in sG:
        #        if not d.loc[v, 'pass'] and sG.in_degree(v) == 0:
        #            #pruned = prune_edges(sG, list(sG.out_edges(v)))
        #            sG.remove_node(v)
        #            d = d.drop(v)
        #            #d = d.drop(pruned)
        #            changed = True
        #            break
        pass
    #d = d.loc[d['pass']]
    #sGs = [sG for sG in sGs if len(list(sG))]

    # fist reduce contiguous trees, then prune bipartite trees with contiguous motifs, then reduce bipartite trees
    # --> remove bipartite motifs that contain short contiguous motifs
    typeI_sGs = [sG for sG in sGs if d.loc[list(sG)[0], "typeI"]]
    non_typeI_sGs = [sG for sG in sGs if not d.loc[list(sG)[0], "typeI"]]
    d['to_prune'] = False
    for sG in non_typeI_sGs:
        d = resolve_motif_graph(sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr)
    #d = d.loc[d['pass'] | d['typeI']]
    for sG in typeI_sGs:
        # search non-TypeI motifs in TypeI motifs and remove all hits
        for i,_ in d.loc[~d.typeI].iterrows():
            changed = True
            seen = set()
            while changed:
                changed = False
                # TODO: check if it is better to only check the root nodes
                for j,_ in d.loc[list(sG)].iterrows():
                    if j in seen:
                        continue
                    seen.add(j)
                    # check if motif i is contained in motif j
                    idx, ident, diff = motif_contains(d.loc[j].motif, d.loc[i].motif)
                    if idx is not None:
                        pruned = prune_edges(sG, list(sG.out_edges(j)), d)
                        sG.remove_node(j)
                        d = d.drop(pruned)
                        logging.getLogger('comos').debug(f"Motif {d.loc[i, 'motif']} found in TypeI motif {d.loc[j, 'motif']}, pruned {len(pruned)+1} nodes")
                        changed = True
                        break
        # reduce the remaining graph
        d = resolve_motif_graph(sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr)
    d = d.loc[d['pass']]

    d = d.drop(columns=['mlen', 'slen', 'pass', 'to_prune'])

    logging.info(f"{len(d)} canonical motifs remaining after removing nested motifs:")
    print(d)

    changed = True
    tested = set()
    while changed == True:
        changed = False
        for i in range(0,len(d)):
            for j in range(i+1,len(d)):
                if d.iloc[i].typeI != d.iloc[j].typeI:
                    continue
                if ((d.iloc[i].motif, d.iloc[i].poi), (d.iloc[j].motif, d.iloc[j].poi)) in tested:
                    continue
                tested.add(((d.iloc[i].motif, d.iloc[i].poi), (d.iloc[j].motif, d.iloc[j].poi)))

                combined_motifs = combine_motifs(d.iloc[i].motif, d.iloc[j].motif)
                aggregates, Ns = aggr_fct(combined_motifs, max_motif_len, fwd_metric, rev_metric, sa)
                #calculate number of std. deviations
                if opt_dir == "min":
                    best = (None, None, np.inf, 0, np.inf, np.inf, d.iloc[i].typeI)
                else:
                    best = (None, None, -np.inf, 0, -np.inf, np.inf, d.iloc[i].typeI)
                for (m,motif) in enumerate(combined_motifs):
                    for poi in get_pois(motif):
                        N = min(Ns[m][poi], len(mu)-1)
                        stddevs = (aggregates[m][poi] - mu[N]) / sigma[N]
                        if (opt_dir == "min" and stddevs < best[4]) or (opt_dir == "max" and stddevs > best[4]):
                            if (opt_dir == "min" and stddevs < thr) or (opt_dir == "max" and stddevs > thr):
                                if ambiguity_thr is not None:
                                    if not test_ambiguous(motif, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=ambiguity_min_sites ,min_tests=ambiguity_min_tests):
                                        if d.iloc[i].typeI:
                                            if test_ambiguous(reverse_complement(motif), sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=ambiguity_min_sites, min_tests=ambiguity_min_tests):
                                                logging.getLogger('comos').info(f'ambiguous test failed for combined motif {motif} but not for its RC --> kept.')
                                            else:
                                                continue
                                        else:
                                            continue
                                # test if all exploded, canonical motifs of the combined motif are above the selection threshold
                                if not test_exploded(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr):
                                    logging.getLogger('comos').debug(f'exploded motif test failed for combined motif {motif}')
                                    continue
                                if opt_dir == "min":
                                    best = (motif, poi, aggregates[m][poi], Ns[m][poi], stddevs, norm().cdf(stddevs), d.iloc[i].typeI)
                                else:
                                    best = (motif, poi, aggregates[m][poi], Ns[m][poi], stddevs, 1. - norm().cdf(stddevs), d.iloc[i].typeI)
                #logging.getLogger('comos').debug("comparison:", best[4], max(d.iloc[i].stddevs, d.iloc[j].stddevs))
                if (opt_dir == "min" and best[4] < thr) or (opt_dir == "max" and best[4] > thr):
                    keep_combined, drop_mi, drop_mj = True, True, True
                    if len(best[0]) <= len(d.iloc[i].motif):
                        idx, ident, _ = motif_contains(d.iloc[i].motif, best[0])
                        if idx is not None:
                            m_diff = motif_diff(d.iloc[i].motif, best[0], idx)
                            aggregates, Ns = aggr_fct([m_diff], len(m_diff), fwd_metric, rev_metric, sa)
                            if opt_dir == "min":
                                best_mean = np.nanmin(aggregates[0])
                            else:
                                best_mean = np.nanmax(aggregates[0])
                            if np.isnan(best_mean):
                                # TODO: can this ever occur?
                                logging.getLogger('comos').error(f"unhandled exception: found no motif aggregate for motif {m_diff}")
                                exit(1)
                            if opt_dir == "min":
                                poi = np.nanargmin(aggregates[0])
                            else:
                                poi = np.nanargmax(aggregates[0])
                            N = min(Ns[0][poi], len(mu)-1)
                            stddev = (best_mean - mu[N]) / sigma[N]
                            if (opt_dir == "min" and stddev >= thr) or (opt_dir == "max" and stddev <= thr):
                                # do NOT keep combined motif, but mi
                                keep_combined = False
                                drop_mi = False
                    if len(best[0]) <= len(d.iloc[j].motif):
                        idx, ident, _ = motif_contains(d.iloc[j].motif, best[0])
                        if idx is not None:
                            m_diff = motif_diff(d.iloc[j].motif, best[0], idx)
                            aggregates, Ns = aggr_fct([m_diff], len(m_diff), fwd_metric, rev_metric, sa)
                            if opt_dir == "min":
                                best_mean = np.nanmin(aggregates[0])
                            else:
                                best_mean = np.nanmax(aggregates[0])
                            if np.isnan(best_mean):
                                # TODO: can this ever occur?
                                logging.getLogger('comos').error(f"unhandled exception: found no motif mean for motif {m_diff}")
                                exit(1)
                            if opt_dir == "min":
                                poi = np.nanargmin(aggregates[0])
                            else:
                                poi = np.nanargmax(aggregates[0])
                            N = min(Ns[0][poi], len(mu)-1)
                            stddev = (best_mean - mu[N]) / sigma[N]
                            if stddev >= thr:
                                # do NOT keep combined motif, but mj
                                keep_combined = False
                                drop_mj = False
                    if keep_combined:
                        new_motif = pd.Series(best, index=d.columns)
                        logging.getLogger('comos').debug(f"combined {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f}) and {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f}) to {best[0]}:{best[1]} ({best[3]}, {best[4]:.2f})")
                        # drop / replace entries
                        d.iloc[i] = new_motif
                        d = d.drop(d.index[j])
                        changed = True
                        break
                    elif drop_mi and not drop_mj:
                        logging.getLogger('comos').debug(f"dropped {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f}) for {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f}) considering {best[0]}:{best[1]} ({best[3]}, {best[4]:.2f})")
                        d = d.drop(d.index[i])
                        changed = True
                        break
                    elif drop_mj and not drop_mi:
                        logging.getLogger('comos').debug(f"dropped {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f}) for {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f}) considering {best[0]}:{best[1]} ({best[3]}, {best[4]:.2f})")
                        d = d.drop(d.index[j])
                        changed = True
                        break
                    else:
                        # actually not impossible
                        # example:
                        # CATC
                        #  ATCG
                        #  ATC   (combined)
                        #
                        # CATCG  (true motif) !
                        # however: then CATCG should be in the list of candidates itself!
                        # TODO: think about this more
                        logging.getLogger('comos').warning(f"unhandled case: combined motif and both original motifs shall be dropped")
                    
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
                    if (opt_dir == "min" and d.iloc[i].stddevs <= d.iloc[j].stddevs) or (opt_dir == "max" and d.iloc[i].stddevs >= d.iloc[j].stddevs):
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
    return d.sort_values('stddevs').reset_index(drop=True)

def is_palindromic(motif):
    return reverse_complement(motif) == motif

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
    
    cache_fp = os.path.join(args.cache, f"{os.path.abspath(args.rds).replace('/','_')}.pkl")
    if os.path.exists(cache_fp) and args.cache:
        tstart = time.time()
        df = pd.read_pickle(cache_fp)
        logging.getLogger('comos').info(f"Parsed cached RDS data in {tstop - tstart:.2f} seconds")
        tstop = time.time()
    else:
        tstart = time.time()
        df = parse_diff_file(args.rds, contig_id)
        tstop = time.time()
        logging.getLogger('comos').info(f"Parsed RDS file in {tstop - tstart:.2f} seconds")
        if args.cache:
            df.to_pickle(cache_fp)
    
    if args.metric == "expsumlogp":
        fwd_metric, rev_metric = compute_expsumlogp(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
        opt_dir = "min"
        selection_thr = -args.selection_thr
        ambiguity_thr = -args.ambiguity_thr
    elif args.metric == "meanabsdiff":
        fwd_metric, rev_metric = compute_meanabsdiff(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
        opt_dir = "max"
        selection_thr = args.selection_thr
        ambiguity_thr = args.ambiguity_thr
    if args.aggregate == "mean":
        aggr_fct = suffix_array.motif_means
    elif args.aggregate == "median":
        aggr_fct = suffix_array.motif_medians

    canon_motifs = get_seq_index_combinations(args.min_k, args.max_k, args.min_g, args.max_g, ['A', 'C'], 0)
    all_canon_motifs = list(canon_motifs.keys())
    logging.getLogger('comos').info(f"Analyzing {len(canon_motifs):,} canonical motifs and "\
        f"{sum([len(canon_motifs[i]) for i in canon_motifs]):,} indices within these motifs")
    logging.getLogger('comos').info(f"Using {args.metric} metric with {args.aggregate} aggregation of site metrics over {args.window} nt windows (min {args.min_window_values} per window){', with background substraction' if args.subtract_background else ''}")
    # do computation
    cache_fp = os.path.join(args.cache, f"{contig_id}_{args.metric}_{args.aggregate}_c{args.min_cov}_w{args.window}_wv{args.min_window_values}_b{args.subtract_background}_k{args.min_k}-{args.max_k}_g{args.min_g}-{args.max_g}.pkl")
    if os.path.exists(cache_fp) and args.cache:
        tstart = time.time()
        with open(cache_fp, 'rb') as f:
            aggr_metric, aggr_metric_counts = pickle.load(f)
        tstop = time.time()
        logging.getLogger('comos').info(f"Loaded canonical motif scores from cache in {tstop - tstart:.2f} seconds")
    else:
        tstart = time.time()
        aggr_metric, aggr_metric_counts = aggr_fct(all_canon_motifs, args.max_k + args.max_g, fwd_metric, rev_metric, sa)
        tstop = time.time()
        logging.getLogger('comos').info(f"Computed canonical motif scores in {tstop - tstart:.2f} seconds")
        if args.cache:
            with open(cache_fp, 'wb') as f:
                pickle.dump((aggr_metric, aggr_metric_counts), f)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        if opt_dir == "min":
            best = np.nanmin(aggr_metric, axis=1)
        else:
            best = np.nanmax(aggr_metric, axis=1)
    mask = ~np.isnan(best)
    results = pd.DataFrame(best, columns=['val'], index=all_canon_motifs)[mask]
    if opt_dir == "min":
        results['poi'] =  np.nanargmin(aggr_metric[mask], axis=1)
    else:
        results['poi'] =  np.nanargmax(aggr_metric[mask], axis=1)
    results['N'] = aggr_metric_counts[mask][range(results['poi'].shape[0]), results['poi']]
    results = results[['poi', 'val', 'N']] # change order of columns to match that in function reduce_motifs

    tstart = time.time()
    mu, sigma = normal_approx(results)
    tstop = time.time()
    logging.getLogger('comos').info(f"Performed normal approximation in {tstop - tstart:.2f} seconds")
    
    results['stddevs'] = (results.val - mu[results.N]) / sigma[results.N]
    if opt_dir == "min":
        results['p-value'] = norm().cdf(results['stddevs'])
    else:
        results['p-value'] = 1.0 - norm().cdf(results['stddevs'])
    results = results.reset_index().rename(columns={'index':'motif'})

    if args.test_motifs:
        aggr_metric, aggr_metric_counts = aggr_fct(args.test_motifs, np.max([len(m) for m in args.test_motifs]), fwd_metric, rev_metric, sa)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            if opt_dir == "min":
                best = np.nanmin(aggr_metric, axis=1)
            else:
                best = np.nanmax(aggr_metric, axis=1)
        mask = ~np.isnan(best)
        res = pd.DataFrame(best, columns=['val'], index=args.test_motifs)[mask]
        if opt_dir == "min":
            res['poi'] =  np.nanargmin(aggr_metric[mask], axis=1)
        else:
            res['poi'] =  np.nanargmax(aggr_metric[mask], axis=1)
        res['poi'] = res['poi'].astype('int32')
        res['N'] = aggr_metric_counts[mask][range(res['poi'].shape[0]), res['poi']]
        res = res[['poi', 'val', 'N']]
        res['stddevs'] = (res.val - mu[np.clip(res.N, 1, len(mu)-1)]) / sigma[np.clip(res.N, 1, len(mu)-1)]
        if opt_dir == "min":
            res['p-value'] = norm().cdf(res['stddevs'])
        else:
            res['p-value'] = 1.0 - norm().cdf(res['stddevs'])
        for m,row in res.iterrows():
            print(f"\nAmbiguous-Test results for motif {m}:{int(row.poi)}")
            passed = test_ambiguous(m, sa, len(m), mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=args.ambiguity_min_sites, debug=True, min_tests=args.ambiguity_min_tests)
            print(passed)
            print(f"\nExploded-Test results for motif {m}:{int(row.poi)}")
            passed = test_exploded(m, int(row.poi), sa, len(m), mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, selection_thr, debug=True)
            print(passed)
            print()
        
        if args.plot:
            fwd_diff, rev_diff = compute_diffs(df, seq)
            for m,row in res.iterrows():
                plot_context_dependent_differences(m, row.poi, sa, fwd_diff, rev_diff, savepath=os.path.join(args.out, f"{m}_{int(row.poi)}_median.png"))
                plot_diffs_with_complementary(m, sa, fwd_diff, rev_diff, savepath=os.path.join(args.out, f"{m}_{int(row.poi)}.png"), ymax=10., absolute=True)
        res = res.reset_index().rename(columns={'index':'motif'})
        print(res)
        exit()

    if args.plot:
        plot_motif_scores(results, mu, sigma, thr=selection_thr, savepath=os.path.join(args.out, f"scores_scatterplot.png"))
    
    if opt_dir == "min":
        sel = (results.stddevs <= selection_thr)
    else:
        sel = (results.stddevs >= selection_thr)
    logging.getLogger('comos').info(f"Selected {sel.sum():,} motifs based on selection threshold of {selection_thr} std. deviations.")
    #print(results.loc[sel])
    tstart = time.time()
    motifs_found = reduce_motifs(
        results.loc[sel], 
        sa, args.max_k + args.max_g, 
        mu, sigma, 
        fwd_metric, rev_metric,
        aggr_fct, opt_dir,
        thr=selection_thr, 
        ambiguity_thr=ambiguity_thr,
        ambiguity_min_sites=args.ambiguity_min_sites,
        ambiguity_min_tests=args.ambiguity_min_tests)
    tstop = time.time()
    logging.getLogger('comos').info(f"Performed motif reduction in {tstop - tstart:.2f} seconds")
    motifs_found['Type'] = ""
    gapped = motifs_found['motif'].str.contains('NNN')
    motifs_found.loc[gapped, 'Type'] = "I"
    motifs_found.loc[(~gapped) & (motifs_found['motif'].apply(is_palindromic)), 'Type'] = "II"
    motifs_found.loc[motifs_found['Type'] == "", 'Type'] = "III"
    motifs_found = motifs_found[['motif', 'poi', 'representative', 'Type', 'val', 'N', 'stddevs', 'p-value']]
    logging.getLogger('comos').info(f"Reduced to {len(motifs_found)} motifs in {len(motifs_found.representative.unique())} MTase groups:")
    print(motifs_found.set_index(['representative', 'motif']).sort_values(by=['Type', 'representative', 'stddevs'], ascending=[True, True, opt_dir=='min']))
    
    if args.plot:
        fwd_diff, rev_diff = compute_diffs(df, seq)
        for r,row in motifs_found.loc[motifs_found.motif == motifs_found.representative].iterrows():
            plot_diffs_with_complementary(row.motif, sa, fwd_diff, rev_diff, absolute=True, savepath=os.path.join(args.out, f"{row.motif}_{row.poi}.png"))

    motifs_found.to_csv(os.path.join(args.out, f"results.csv"))
    

if __name__ == '__main__':
    #logging.basicConfig(format=f'%(levelname)s [%(asctime)s] : %(message)s',
    #                    datefmt='%H:%M:%S')
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, 
                        format=f'%(levelname)s [%(asctime)s] : %(message)s',
                        datefmt='%H:%M:%S')
    main(args)