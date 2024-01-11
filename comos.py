import pyreadr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Bio import SeqIO
import sys
import itertools
from tqdm import tqdm
import random
import re
import scipy
import pickle
import os
import time
import argparse
import logging
import suffix_array

import matplotlib
matplotlib.use('TkAgg')

class ArgHelpFormatter(argparse.HelpFormatter):
    '''
    Formatter properly adding default values to help texts.
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
        default='cosmos_output.csv',
        help='CSV filename to store the output in.')

    motif_grp = parser.add_argument_group('Motif Options')
    motif_grp.add_argument(
        '--min_k',
        type=int,
        default=4,
        help='Minimum number of specific motif positions.'
    )
    motif_grp.add_argument(
        '--max_k',
        type=int,
        default=8,
        help='Maximum number of specific motif positions.'
    )
    motif_grp.add_argument(
        '--min_g',
        type=int,
        default=4,
        help='Minimum gap size in TypeI motifs.'
    )
    motif_grp.add_argument(
        '--max_g',
        type=int,
        default=9,
        help='Maximum gap size in TypeI motifs.'
    )

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.genome):
        logging.getLogger('comos').error(f"Input file {args.genome} does not exist.")
        exit(1)
    if not os.path.exists(args.rds):
        logging.getLogger('comos').error(f"Input file {args.rds} does not exist.")
        exit(1)
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

def parse_largest_contig(fp):
    n_largest = 0
    contig_id = None
    n = 0
    for record in SeqIO.parse(fp, "fasta"):
        n += 1
        if len(record.seq) > n_largest:
            contig_id = record.id
            n_largest = len(record.seq)
            seq = record.seq
            sa = suffix_array.get_suffix_array(record.id, record.seq)
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

def plot_diffs_with_complementary(motif, sa, fwd_diff, rev_diff, ymax=np.inf, offset=3):
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
    plt.show()

    
def main(args):
    seq, sa, contig_id, n_contigs = parse_largest_contig(args.genome)
    if n_contigs > 1: # TODO: analyze all contigs
        logging.getLogger('comos').warning(
            f"Dataset contains {n_contigs} contig(s), "\
            f"only {contig_id} of length {len(seq)} is analyzed")
    df = parse_diff_file(args.rds, contig_id)
    fwd_expsumlogp, rev_expsumlogp = compute_expsumlogp(df, seq)
    #fwd_diff, rev_diff = compute_diffs(df, seq)
    #plot_diffs_with_complementary("GAGNNNNNTGA", sa, fwd_diff, rev_diff, ymax=6., offset=3)
    
    motifs = get_seq_index_combinations(args.min_k, args.max_k, args.min_g, args.max_g, ['A', 'C'], 0)
    all_motifs = list(motifs.keys())
    logging.getLogger('comos').info(f"Analyzing {len(motifs)} motifs and "\
        f"{sum([len(motifs[i]) for i in motifs])} indices within these motifs")
    # do computation
    tstart = time.time()
    means, means_counts = suffix_array.motif_means(all_motifs, args.max_k+args.max_g, fwd_expsumlogp, rev_expsumlogp, sa)
    tstop = time.time()
    logging.getLogger('comos').info(f"Computed motif scores in {tstop - tstart:.2f} seconds")
    # retrieve results
    results = {}
    for i,motif in enumerate(all_motifs):
        results[motif] = {}
        for poi in motifs[motif]:
            results[motif][poi] = means[i][poi], means_counts[i][poi]

if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=f'%(levelname)s : %(message)s')
    main(args)