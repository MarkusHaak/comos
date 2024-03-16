import os
import warnings
import argparse
import logging
import itertools
import pickle
import time

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats.mstats import gmean, hmean
import networkx as nx
import matplotlib

import sanamos

from .constants import IUPAC_TO_IUPAC,IUPAC_TO_LIST,IUPAC_NOT
from .helper import ArgHelpFormatter,parse_rds_file,parse_bed_file,parse_tombo_files,parse_largest_contig,parse_contigs
from .metrics import compute_expsumlogp,compute_diffs,compute_meanabsdiff,compute_meanfrac
from .motif import reverse_complement,is_palindromic,combine_IUPAC,combine_motifs,explode_motif,get_pois,motif_contains,motif_diff
from .plotting import plot_context_dependent_differences,plot_diffs_with_complementary,plot_motif_scores,letterAt,plot_EDLogo
from .tests import test_ambiguous,get_EDLogo_enrichment_scores,test_replace,test_replace_order,test_replace_total,test_exploded

matplotlib.use('TkAgg')

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=ArgHelpFormatter,
        )
    io_grp = parser.add_argument_group('Input/Output Options')
    io_grp.add_argument(
        '--fasta', 
        required=True,
        help='Fasta file containing contig(s).')
    io_grp.add_argument(
        '--mod', 
        required=True,
        help='''File containing modification information, 
        either an RDS Rdata file (.RDS) created with Nanodisco, 
        a bedMethyl file (.bed) from modkit or 
        the common prefix of tombo text-output files.''')
    io_grp.add_argument(
        '--out', '-o', 
        default='./comos_output',
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
        help='Minimum gap size in biparite motifs.'
    )
    motif_grp.add_argument(
        '--max-g',
        type=int,
        default=9,
        help='Maximum gap size in biparite motifs.'
    )
    motif_grp.add_argument(
        '--bases',
        default="AC",
        help="Canonical bases that are considered as potentially modified."
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
        choices=["meanabsdiff", "expsumlogp", "meanfrac"],
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
        '--ambiguity-type',
        choices=['metric', 'logo'],
        default="logo",
        help='Which type of ambiguity test should be performed.'
    )
    hyper_grp.add_argument(
        '--ambiguity-thr',
        type=float,
        default=3.0,
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
    hyper_grp.add_argument(
        '--replace-min-enrichment',
        type=int,
        default=-0.1,
        help="Minimum number of test cases per ambiguous position (1-4) with sufficient sites, otherwise test fails."
    )
    hyper_grp.add_argument(
        '--outlier-iqr',
        type=float,
        default=0.0,
        help="Datapoints outside median +- <outlier-iqr> * IQR are considered outliers when calculating background model. Disable outlier detection with value <= 0.0."
    )

    misc_grp = parser.add_argument_group('Misc')
    misc_grp.add_argument(
        '--test-motifs',
        nargs='+',
        help='Analyze the given IUPAC motifs.' 
    )
    misc_grp.add_argument(
        '--metagenome',
        action="store_true",
        help="Analyze contigs in the input fasta individually for MTase motifs."
    )
    misc_grp.add_argument(
        '--min-seq-len',
        type=int,
        default=200_000,
        help="Minimum length of contigs in the Multiple-Fasta to be perform an initial motif search on them."
    )


    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.fasta):
        logging.getLogger('comos').error(f"Input file {args.fasta} does not exist.")
        exit(1)
    if not os.path.exists(args.mod):
        if not os.path.exists(os.path.dirname(args.mod)):
            logging.getLogger('comos').error(f"Input file {args.mod} does not exist.")
            exit(1)
        else:
            missing = []
            for suffix in ['.valid_coverage.plus.wig', '.valid_coverage.minus.wig', '.dampened_fraction_modified_reads.plus.wig', '.dampened_fraction_modified_reads.minus.wig']:
                candidate_fns = [fn for fn in os.listdir(os.path.dirname(args.mod)) if fn.startswith(os.path.basename(args.mod)) and fn.endswith(suffix)]
                if len(candidate_fns) != 1:
                    missing.append(suffix)
            if missing:
                logging.getLogger('comos').error(f"Detected tombo input. Missing text-output files with suffixes {missing} for prefix {args.mod}.")
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if args.cache and not os.path.exists(args.cache):
        os.makedirs(args.cache)
    args.fasta = os.path.abspath(args.fasta)
    args.mod = os.path.abspath(args.mod)

    # check hyperparameters
    if args.ambiguity_thr > args.selection_thr:
        logging.getLogger('comos').warning(f"Ambiguity threshold is set to selection threshold since it should not be larger.")
        args.ambiguity_thr = args.selection_thr

    return args

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
                    valid = False
                    if 'C' in bases:
                        valid = valid or ('C' in comb[:gap_pos] and 'G' in comb[gap_pos:]) or ('G' in comb[:gap_pos] and 'C' in comb[gap_pos:])
                    if 'A' in bases:
                        valid = valid or ('A' in comb[:gap_pos] and 'T' in comb[gap_pos:]) or ('T' in comb[:gap_pos] and 'A' in comb[gap_pos:])
                    if not valid:
                        continue
                    comb_ = comb[:gap_pos] + ["N"]*gap + comb[gap_pos:]
                    indexes_ = get_indexes(comb_, k+gap)
                    seq_ = "".join(comb_)
                    seq_index_combinations[seq_] = indexes_
    return seq_index_combinations

def normal_approx(d, outlier_iqr=0.):
    mu = []
    sigma = []
    med = []
    iqrs = []
    window = lambda x : 20 + x
    N = np.unique(np.concatenate([np.geomspace(1,d.N.max(),min(d.N.max()-1,500)).astype(int), 
                                  np.linspace(1,d.N.max(),min(d.N.max()-1,1000)).astype(int)]))
    for n in N:
        sel = (d.N >= n - window(n)//2) & (d.N <= n + window(n)//2)
        X = d.loc[sel, 'val']
        q75, q25 = np.percentile(X, [75 ,25])
        iqr = q75 - q25
        if outlier_iqr > 0.:
            median = np.median(X)
            outlier = (X < median - outlier_iqr*iqr) | (X > median + outlier_iqr*iqr)
            X = X[~outlier]
        sigma.append(X.std())
        mu.append(X.mean())
        med.append(X.median())
        iqrs.append(iqr)
    
    # piecewise linear interpolation
    Ns = np.array(range(0,d.N.max()+1))
    return np.interp(Ns, N, mu), np.interp(Ns, N, sigma), np.interp(Ns, N, med), np.interp(Ns, N, iqrs)

def mutate_replace(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, min_N=30):
    poi = int(poi)
    likelihoods = []
    for i in range(len(motif)):
        likelihoods.append({})
        if motif[i] == 'N' or i == poi:
            likelihoods[-1][motif[i]] = 1.0
            continue
        df = get_EDLogo_enrichment_scores(motif, poi, i, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, min_N=min_N, stddev=3.0, n_pseudo=5, Nnull=500)
        bases = df.loc[df.r >= 0, 'base']
        for l in range(1,len(bases)+1):
            for comb in itertools.combinations(bases, l):
                iupac = [b for b in IUPAC_TO_LIST if set(IUPAC_TO_LIST[b]) == set(comb)][0] # TODO: optimize
                likelihood = (len(comb) * hmean(df.loc[df.base.isin(comb), 'r']) + (-df.loc[df.base.isin(IUPAC_TO_LIST[IUPAC_NOT[iupac]]), 'r']).sum()) / df.r.abs().sum()
                likelihoods[-1][iupac] = likelihood
    most_likely, probs = [], []
    for d in likelihoods:
        maxval, maxbase = -np.inf, None
        for base,likelihood in d.items():
            if likelihood > maxval:
                maxval, maxbase = likelihood, base
        most_likely.append(maxbase)
        probs.append(maxval)
    return "".join(most_likely), probs

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
    for v in nodes:
        if G.in_degree(v) == 0:
            G.remove_node(v)
            pruned.append(v)
        else:
            d.loc[v, 'to_prune'] = True
    return pruned

def resolve_nested_motif_graph(sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, bases="AC"):
    all_sG_nodes = list(sG)
    draw = False
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
    while True:
        root_nodes = [n for n,deg in sG.in_degree() if deg==0]
        for root_node in root_nodes:
            root_motif = d.loc[root_node, 'motif']
            # check if node was previously flagged to be pruned but was not pruned yet due to previously unresolved incoming edges
            if d.loc[root_node, 'to_prune'] == True:
                pruned = prune_edges(sG, list(sG.out_edges(root_node)), d)
                d = d.drop(pruned)
                logging.getLogger('comos').debug(f"pruned previously flagged root node {d.loc[root_node, 'motif']} and consequently pruned {len(pruned)} other nodes")
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

            # always keep shorter motif, prune the longer ones
            pruned = prune_edges(sG, [edge for edge in sG.out_edges(root_node)], d)
            d = d.drop(pruned)
            logging.getLogger('comos').debug(f"kept root node {d.loc[root_node, 'motif']}, pruned {len(pruned)} nodes")

            if draw:
                fig,ax = plt.subplots(1,1,figsize=(6,4))
                pos = nx.spring_layout(sG)
                nx.draw_networkx(sG, pos=pos, ax=ax)
                nx.draw_networkx(sG, nodelist=d.loc[d.index.isin(list(sG)) & ~d['pass']].index, edgelist=[], node_color="red", pos=pos, ax=ax)
                if draw in sG:
                    nx.draw_networkx(sG, nodelist=[draw], edgelist=sG.in_edges(draw), node_color="green", edge_color="green", pos=pos, ax=ax)
                plt.show(block=False)
        if len(sG.edges) == 0 and ~np.any(d.to_prune):
            break
    if draw:
        breakpoint()
    if len(d.loc[d.index.isin(all_sG_nodes)]) != len(list(sG)):
        d = d.drop([n for n in all_sG_nodes if n in d.index and n not in list(sG)])
    return d

def prune_combined_node(n, v, sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, bases="AC", ambiguity_type='logo'):
    """
    Prune node n that is the result of a motif combination of v
    #but test if the difference n - v passes tests and shall remain.
    Also, recursively apply the same principle to nodes that were combined from n
    """
    if n not in list(sG):
        return d
    comb_nodes = [u for u,_ in sG.in_edges(n)]
    for u in comb_nodes:
        sG.remove_edge(u,n)
    for u in comb_nodes:
        if u in d.index:
            # climb up the tree towards root nodes
            d = prune_combined_node(u, n, sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, bases=bases, ambiguity_type=ambiguity_type)
    
    ## test difference motif and potentially add it to graph
    ## find the index where the combined motif u matches n
    ## TODO: pass on this information from the actual combination call
    ## TODO: in theory a nice idea, but I would need to properly integrate it into the Graph that is just being resolved
    #indices = []
    #for m in explode_motif(d.loc[n, 'motif']):
    #    idx,_,_ = motif_contains(d.loc[v, 'motif'], m, d.loc[v, 'bipartite'])
    #    if idx is not None:
    #        indices.append(idx)
    #if len(indices) != 0:
    #    assert len(set(indices)) == 1, f"Unhandled case: exploded motifs of the combined motif {d.loc[u, 'motif']} are found at different indices in  {d.loc[v, 'motif']} that it was combined from"
    #    idx = indices[0]
    #    diff = motif_diff(d.loc[n, 'motif'], d.loc[v, 'motif'][idx:idx+len(d.loc[n, 'motif'])], 0, subtract_matching=True)
    #    aggregates, Ns = aggr_fct([diff], len(diff), fwd_metric, rev_metric, sa, bases=bases)
    #    if opt_dir == "min":
    #        best_aggregate = np.nanmin(aggregates[0])
    #    else:
    #        best_aggregate = np.nanmax(aggregates[0])
    #    assert ~np.isnan(best_aggregate), f"unhandled exception: found no motif aggregate for motif {diff}"
    #    if opt_dir == "min":
    #        poi = np.nanargmin(aggregates[0])
    #    else:
    #        poi = np.nanargmax(aggregates[0])
    #    N = min(Ns[0][poi], len(mu)-1)
    #    stddev = (best_aggregate - mu[N]) / sigma[N]
    #    passed = test_ambiguous(diff, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=ambiguity_min_sites, min_tests=ambiguity_min_tests, bases=bases, ambiguity_type=ambiguity_type) and \
    #            test_exploded(diff, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, bases=bases)
    #    if args.ambiguity_type == 'logo':
    #        passed &= test_replace(diff, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, replace_min_enrichment, min_N=ambiguity_min_sites)
    #
    #    # add a new independent node if tests were passed
    #    if passed:
    #        new = pd.Series([diff, poi, aggregates[0][poi], Ns[0][poi], stddev, norm().cdf(stddev), d.loc[n, 'bipartite'], None, False], index=d.columns)
    #        if ~d.motif.eq(new.motif).any():
    #            logging.info(f"Added motif {new.motif}:{poi} ({new.N}, {new.stddevs:.2f}) as difference {d.loc[n, 'motif']} - {d.loc[v, 'motif']}")
    #            d.loc[d.index.max()+1] = new
    #    else:
    #        logging.info(f"Motif {diff} as difference {d.loc[n, 'motif']} - {d.loc[v, 'motif']} did not pass tests")
    #else:
    #    logging.debug(f"no exploded motif of the combined motif {d.loc[n, 'motif']} is found in {d.loc[v, 'motif']} that it was combined from")
    
    # remove node n, if it is still in the graph, pruning its outgoing edges
    pruned = prune_edges(sG, list(sG.out_edges(n)), d)
    d = d.drop(pruned)
    logging.getLogger('comos').info(f"pruned combined node {d.loc[n, 'motif']} and consequently pruned {len(pruned)} other nodes")
    sG.remove_node(n)
    d = d.drop(n)
    return d

def final_motif_filter(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, min_N=30):
    passes  = test_replace_order(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, min_N=min_N)
    passes &= test_replace_total(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, 1., min_N=min_N)
    return passes

def resolve_combined_motif_graph(sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, bases="AC", ambiguity_type='logo'):
    all_sG_nodes = list(sG)
    # resolve graph, starting with root nodes
    while True:
        root_nodes = [n for n,deg in sG.in_degree() if deg==0]
        for root_node in root_nodes:
            # check if node was previously flagged to be pruned but was not pruned yet due to previously unresolved incoming edges
            if d.loc[root_node, 'to_prune'] == True:
                pruned = prune_edges(sG, list(sG.out_edges(root_node)), d)
                d = d.drop(pruned)
                logging.getLogger('comos').info(f"pruned previously flagged root node {d.loc[root_node, 'motif']} and consequently pruned {len(pruned)} other nodes")
                sG.remove_node(root_node)
                d = d.drop(root_node)
                break
            # check if node passes final filtering
            if not final_motif_filter(d.loc[root_node, 'motif'], d.loc[root_node, 'poi'], sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, min_N=ambiguity_min_sites):
                logging.getLogger('comos').info(f"removed root node {d.loc[root_node, 'motif']}, did not pass final filtering")
                sG.remove_node(root_node)
                d = d.drop(root_node)
                break
        else:
            # update reverse complements
            for i in d.index:
                if d.loc[i, 'rc'] is not None:
                    if d.loc[i, 'rc'] not in d.index:
                        d.loc[i, 'rc'] = None
            for n in d.loc[d.index.isin(root_nodes) & (d.rc == d.index)].index:
                # prioritize palindromic motifs
                # prune graph for nodes that were combined to this one
                if list(sG.out_edges(n)):
                    pruned = prune_edges(sG, list(sG.out_edges(n)), d)
                    d = d.drop(pruned)
                    logging.getLogger('comos').info(f"kept palindromic root node {d.loc[n, 'motif']} and consequently pruned {len(pruned)} other nodes")
                    break
            else:
                indices = d.index.copy() # to check if any nodes where pruned
                for n in d.loc[d.index.isin(root_nodes) & ~pd.isna(d.rc) & (d.rc != d.index)].index:
                    # prioritize motifs with RC over those without
                    # keep RC (which is not necessarily a root node) and handle all motifs that it was combined to
                    # TODO: currently, for the RC, no final_motif_filter is applied! Think about how to handle this
                    rc = d.loc[n, 'rc']
                    comb_nodes = [u for u,_ in sG.in_edges(rc)]
                    for u in comb_nodes:
                        sG.remove_edge(u,rc)
                    for u in comb_nodes:
                        if u in d.index and u != n:
                            d = prune_combined_node(u, rc, sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, bases=bases, ambiguity_type=ambiguity_type)
                    
                    # prune graph for nodes that were combined to this one and its RC
                    pruned, pruned_rc = [], []
                    if list(sG.out_edges(n)):
                        pruned = prune_edges(sG, list(sG.out_edges(n)), d)
                        d = d.drop(pruned)
                    if list(sG.out_edges(rc)):
                        pruned_rc = prune_edges(sG, list(sG.out_edges(rc)), d)
                        d = d.drop(pruned_rc)
                    if len(indices) != len(d.index):
                        logging.getLogger('comos').info(f"kept root node {d.loc[n, 'motif']} (and its RC {d.loc[rc, 'motif']}) and consequently pruned {len(indices) - len(d.index)} other nodes")
                        break
                else:
                    for n in d.loc[d.index.isin(root_nodes) & pd.isna(d.rc)].index:
                        # prune graph for nodes that were combined to this one and its RC
                        if list(sG.out_edges(n)):
                            pruned = prune_edges(sG, list(sG.out_edges(n)), d)
                            d = d.drop(pruned)
                            logging.getLogger('comos').info(f"kept root node {d.loc[n, 'motif']} and consequently pruned {len(pruned)} other nodes")
                            break
        if len(sG.edges) == 0 and ~np.any(d.to_prune):
            break
    # check last time if all motifs pass final filtering
    root_nodes = [n for n,deg in sG.in_degree() if deg==0]
    for root_node in root_nodes:
        # check if node passes final filtering
        if not final_motif_filter(d.loc[root_node, 'motif'], d.loc[root_node, 'poi'], sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, min_N=ambiguity_min_sites):
            logging.getLogger('comos').info(f"removed root node {d.loc[root_node, 'motif']}, did not pass final filtering")
            sG.remove_node(root_node)
            d = d.drop(root_node)
    if len(d.loc[d.index.isin(all_sG_nodes)]) != len(list(sG)):
        d = d.drop([n for n in all_sG_nodes if n in d.index and n not in list(sG)])
    return d


def combine_and_test_motifs(d, i, j, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, min_k, bases="AC", ambiguity_type='logo'):
    combined_motifs = [m for m in combine_motifs(d.loc[i].motif, d.loc[j].motif) if len(m) > min_k]
    if opt_dir == "min":
        best = [None, None, np.inf, 0, np.inf, np.inf, d.loc[i].bipartite]
    else:
        best = [None, None, -np.inf, 0, -np.inf, np.inf, d.loc[i].bipartite]
    keep_combined, drop_mi, drop_mj = False, False, False
    if len(combined_motifs) == 0:
        return best, keep_combined, drop_mi, drop_mj
    aggregates, Ns = aggr_fct(combined_motifs, max_motif_len, fwd_metric, rev_metric, sa, bases=bases)
    #calculate number of std. deviations
    for (m,motif) in enumerate(combined_motifs):
        #for poi in get_pois(motif):
        if np.all(np.isnan(aggregates[m])):
            logging.getLogger('comos').debug(f"no motif aggregate for motif {motif}")
            continue
        if opt_dir == "min":
            best_aggregate = np.nanmin(aggregates[m])
        else:
            best_aggregate = np.nanmax(aggregates[m])
        if opt_dir == "min":
            poi = np.nanargmin(aggregates[m])
        else:
            poi = np.nanargmax(aggregates[m])
        N = min(Ns[m][poi], len(mu)-1)
        stddev = (best_aggregate - mu[N]) / sigma[N]
        if (opt_dir == "min" and stddev < best[4]) or (opt_dir == "max" and stddev > best[4]):
            if (opt_dir == "min" and stddev <= thr) or (opt_dir == "max" and stddev >= thr):
                # check if combined motif passes tests
                # check ambiguity (if the motif is too short)
                if not test_ambiguous(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=ambiguity_min_sites, min_tests=ambiguity_min_tests, bases=bases, ambiguity_type=ambiguity_type):
                    logging.getLogger('comos').debug(f'ambiguous test failed for combined motif {motif}')
                    continue
                # test if all exploded, canonical motifs of the combined motif are above the selection threshold
                if not test_exploded(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, bases=bases):
                    logging.getLogger('comos').debug(f'exploded motif test failed for combined motif {motif}')
                    continue
                if ambiguity_type == 'logo':
                    if not test_replace(motif, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, replace_min_enrichment, min_N=ambiguity_min_sites):
                        logging.getLogger('comos').debug(f'replace test failed for combined motif {motif}')
                        continue
                if opt_dir == "min":
                    best = [motif, poi, best_aggregate, N, stddev, norm().cdf(stddev), d.loc[i].bipartite]
                else:
                    best = [motif, poi, best_aggregate, N, stddev, 1. - norm().cdf(stddev), d.loc[i].bipartite]
    keep_combined = best[0] is not None
    if keep_combined:
        drop_mi, drop_mj = True, True
    else:
        # check if any of the combined motifs is contained in one of the motifs that are being combined
        for motif in combined_motifs:
            if len(motif) <= len(d.loc[i].motif):
                idx, ident, diff_locs = motif_contains(d.loc[i].motif, motif, d.loc[i].bipartite)
                if idx is not None:
                    if len(diff_locs) <= 1:
                        drop_mj = True
            if len(motif) <= len(d.loc[j].motif):
                idx, ident, diff_locs = motif_contains(d.loc[j].motif, motif, d.loc[j].bipartite)
                if idx is not None:
                    if len(diff_locs) <= 1:
                        drop_mi = True
            if drop_mi and drop_mj:
                # is combined in both, take the shorter one or the better scoring one
                len_i = (len(d.loc[i].motif) - d.loc[i].motif.count('N'))
                len_j = (len(d.loc[j].motif) - d.loc[j].motif.count('N'))
                if len_i < len_j:
                    drop_mi = False
                elif len_j < len_i:
                    drop_mj = False
                elif (opt_dir == "min" and d.loc[i].stddevs <= d.loc[j].stddevs) or (opt_dir == "max" and d.loc[i].stddevs >= d.loc[j].stddevs):
                    drop_mi = False
                else:
                    drop_mj = False
    return best, keep_combined, drop_mi, drop_mj

def test_mutated_motif(mutated, d, i, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, min_k, bases="AC", ambiguity_type='logo'):
    idx,_,_ = motif_contains(mutated, d.loc[i].motif, d.loc[i].bipartite)
    if mutated != d.loc[i].motif and idx is not None:
        mutated = mutated.strip('N') # do this here in case mutated gets smaller, so that it still contains the original motif
        aggregates, Ns = aggr_fct([mutated], len(mutated), fwd_metric, rev_metric, sa, bases=bases)
        if opt_dir == "min":
            best_aggregate = np.nanmin(aggregates[0])
        else:
            best_aggregate = np.nanmax(aggregates[0])
        if not np.isnan(best_aggregate):
            if opt_dir == "min":
                poi = np.nanargmin(aggregates[0])
            else:
                poi = np.nanargmax(aggregates[0])
            N = min(Ns[0][poi], len(mu)-1)
            stddev = (best_aggregate - mu[N]) / sigma[N]
            if stddev > d.loc[i].stddevs:
                # check if combined motif passes tests
                # check ambiguity (if the motif is too short)
                if not test_ambiguous(mutated, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=ambiguity_min_sites, min_tests=ambiguity_min_tests, bases=bases, ambiguity_type=ambiguity_type):
                    logging.getLogger('comos').info(f'ambiguous test failed for mutated motif {mutated}')
                    return False, poi, aggregates[0], Ns[0][poi], stddev
                # test if all exploded, canonical motifs of the combined motif are above the selection threshold
                if not test_exploded(mutated, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, bases=bases):
                    logging.getLogger('comos').info(f'exploded motif test failed for mutated motif {mutated}')
                    return False, poi, aggregates[0], Ns[0][poi], stddev
                if ambiguity_type == 'logo':
                    if not test_replace(mutated, poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, replace_min_enrichment, min_N=ambiguity_min_sites):
                        logging.getLogger('comos').info(f'replace test failed for mutated motif {mutated}')
                        return False, poi, aggregates[0], Ns[0][poi], stddev
                return True, poi, aggregates[0], Ns[0][poi], stddev
    return False, None, None, None, None

def reduce_motifs(d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, min_k, bases="AC", ambiguity_type='logo'):
    d = d.copy().sort_values(['N','stddevs'], ascending=[False,True])
    d['bipartite'] = d.motif.str.contains('NN')
    
    # initial filtering
    d['pass'] = d.apply(lambda row: test_ambiguous(row.motif, row.poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=ambiguity_min_sites, min_tests=ambiguity_min_tests, bases=bases, ambiguity_type=ambiguity_type), axis=1)
    if ambiguity_type == 'logo':
        d.loc[d['pass'] == True, 'pass'] = d.loc[d['pass'] == True].apply(lambda row: test_replace(row.motif, row.poi, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, replace_min_enrichment, min_N=ambiguity_min_sites), axis=1)
    for i,row in d.loc[d['pass'] == False].iterrows():
        logging.getLogger('comos').debug(f"excluded {row.motif}:{row.poi} ({row.N}, {row.stddevs:.2f}) : did not pass ambiguity filter")
    n_excluded = (d['pass'] == False).sum()
    if n_excluded:
        logging.info(f"Excluded {n_excluded:,} motifs because they did not pass ambiguity testing.")
    #d = d.loc[d['pass']].drop(['pass'], axis=1)
    logging.getLogger('comos').info(f"{len(d.loc[d['pass']])} motifs after ambiguity testing")
    print(d.loc[d['pass']].sort_values("stddevs", ascending=(opt_dir=="min")))
    
    # removing nested canonical motifs
    d['mlen'] = d.motif.str.len()
    d['slen'] = d.motif.str.replace('N','').str.len()
    d = d.sort_values(['mlen', 'slen']).reset_index(drop=True)
    G = nx.DiGraph()
    for i in range(0,len(d)):
        G.add_node(i)
        sel = (d.index > i) & (d.bipartite == d.loc[i, 'bipartite']) & (0 <= d.slen - d.loc[i, 'slen']) # TODO: check if longer edges needed, if not: & (d.slen - d.loc[i, 'slen'] <= 1)
        contain_i = d.loc[sel].loc[d.loc[sel].motif.str.contains(d.loc[i].motif.replace('N','.'))]
        if not contain_i.empty:
            for j, row in contain_i.iterrows():
                G.add_edge(i, j, weight=row.slen - d.loc[i, 'slen'])
    n_sGs = len([G.subgraph(c) for c in nx.connected_components(G.to_undirected())])# if np.any(d.loc[list(c), 'pass'])])
    sGs = []
    for c in nx.connected_components(G.to_undirected()):
        sG = G.subgraph(c).copy()
        # drop entries that are not connected to any other by nesting and did not pass filtering
        if ~np.any(d.loc[list(sG), 'pass']):
            d = d.drop(list(sG))
        else:
            sGs.append(sG)
    logging.getLogger('comos').info(f"{n_sGs} subgraphs in nested motif network")
    logging.getLogger('comos').info(f"{len(sGs)} subgraphs with any valid motif")

    # fist reduce contiguous trees, then prune bipartite trees with contiguous motifs, then reduce bipartite trees
    # --> remove bipartite motifs that contain short contiguous motifs
    bipartite_sGs = [sG for sG in sGs if d.loc[list(sG)[0], "bipartite"]]
    non_bipartite_sGs = [sG for sG in sGs if not d.loc[list(sG)[0], "bipartite"]]
    d['to_prune'] = False
    for sG in non_bipartite_sGs:
        d = resolve_nested_motif_graph(sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, bases=bases)
    for sG in bipartite_sGs:
        # search contiguous motifs in biparite motifs and remove all hits
        # TODO: instead do tests on half of motif that is not modified to check if bases can be ambiguous
        for i,_ in d.loc[~d.bipartite].iterrows():
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
                    idx, ident, diff = motif_contains(d.loc[j].motif, d.loc[i].motif, bipartite=True)
                    if idx is not None:
                        pruned = prune_edges(sG, list(sG.out_edges(j)), d)
                        sG.remove_node(j)
                        logging.getLogger('comos').debug(f"Motif {d.loc[i, 'motif']} found in biparite motif {d.loc[j, 'motif']}, pruned {len(pruned)+1} nodes")
                        d = d.drop(j)
                        d = d.drop(pruned)
                        changed = True
                        break
        # reduce the remaining graph
        d = resolve_nested_motif_graph(sG, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, bases=bases)
    d = d.loc[d['pass']]
    d = d.drop(columns=['mlen', 'slen', 'pass', 'to_prune']).sort_index()
    logging.info(f"{len(d)} canonical motifs remaining after removing nested motifs:")
    print(d)

    # Filter canonical motifs
    # try to find missing reverse complements of biparite motifs
    changed = True
    while changed == True:
        changed = False
        # identify reverse complements of each other
        d['rc'] = None
        for i in d.index:
            for j in d.index:
                if reverse_complement(d.loc[j, 'motif']) == d.loc[i, 'motif']:
                    d.loc[i, 'rc'] = j
        for i in d.loc[d.bipartite & pd.isna(d.rc)].index:
            for j in d.loc[d.bipartite].index:
                idx, ident, diff = motif_contains(reverse_complement(d.loc[i].motif), d.loc[j].motif, bipartite=True)
                if idx is not None and len(diff) == 1:
                    # score the motif
                    motif = reverse_complement(d.loc[j].motif)
                    aggregates, Ns = aggr_fct([motif], len(motif), fwd_metric, rev_metric, sa, bases=bases)
                    if opt_dir == "min":
                        best_aggregate = np.nanmin(aggregates[0])
                    else:
                        best_aggregate = np.nanmax(aggregates[0])
                    if np.isnan(best_aggregate):
                        # TODO: can this ever occur?
                        logging.getLogger('comos').error(f"unhandled exception: found no motif aggregate for motif {motif}")
                        exit(1)
                    if opt_dir == "min":
                        poi = np.nanargmin(aggregates[0])
                    else:
                        poi = np.nanargmax(aggregates[0])
                    N = min(Ns[0][poi], len(mu)-1)
                    stddev = (best_aggregate - mu[N]) / sigma[N]
                    # add to df
                    new = pd.Series([motif, poi, aggregates[0][poi], Ns[0][poi], stddev, norm().cdf(stddev), True, i], index=d.columns)
                    logging.info(f"Motif {d.loc[j].motif} is contained in RC of motif {d.loc[i].motif} which misses a RC --> keep {new.motif}:{poi} ({new.N}, {new.stddevs:.2f}) over {d.loc[i].motif}:{d.loc[i].poi}({d.loc[i].N}, {d.loc[i].stddevs:.2f})")
                    d = d.drop(i)
                    if ~d.motif.eq(new.motif).any():
                        d.loc[d.index.max()+1] = new
                    changed = True
                    break
                idx, ident, diff = motif_contains(reverse_complement(d.loc[j].motif), d.loc[i].motif, bipartite=True)
                if idx is not None and len(diff) == 1 and not d.loc[j].rc:
                    # score the motif
                    motif = reverse_complement(d.loc[i].motif)
                    aggregates, Ns = aggr_fct([motif], len(motif), fwd_metric, rev_metric, sa, bases=bases)
                    if opt_dir == "min":
                        best_aggregate = np.nanmin(aggregates[0])
                    else:
                        best_aggregate = np.nanmax(aggregates[0])
                    if np.isnan(best_aggregate):
                        # TODO: can this ever occur?
                        logging.getLogger('comos').error(f"unhandled exception: found no motif aggregate for motif {motif}")
                        exit(1)
                    if opt_dir == "min":
                        poi = np.nanargmin(aggregates[0])
                    else:
                        poi = np.nanargmax(aggregates[0])
                    N = min(Ns[0][poi], len(mu)-1)
                    stddev = (best_aggregate - mu[N]) / sigma[N]
                    # add to df
                    new = pd.Series([motif, poi, aggregates[0][poi], Ns[0][poi], stddev, norm().cdf(stddev), True, i], index=d.columns)
                    logging.info(f"Motif {d.loc[i].motif} is contained in RC of motif {d.loc[j].motif}, both missing a RC --> keep {new.motif}:{poi} ({new.N}, {new.stddevs:.2f}) over {d.loc[j].motif}:{d.loc[j].poi}({d.loc[j].N}, {d.loc[j].stddevs:.2f})")
                    d = d.drop(j)
                    if ~d.motif.eq(new.motif).any():
                        d.loc[d.index.max()+1] = new
                    changed = True
                    break
            else:
                # check again if RC is sufficiently methylated (but was later filtered in resolving the nested graph)
                # no need to check for ambiguity, since for bipartite motifs it suffices if one of two RCs passes
                motif = reverse_complement(d.loc[i].motif)
                aggregates, Ns = aggr_fct([motif], len(motif), fwd_metric, rev_metric, sa, bases=bases)
                if opt_dir == "min":
                    best_aggregate = np.nanmin(aggregates[0])
                else:
                    best_aggregate = np.nanmax(aggregates[0])
                if np.isnan(best_aggregate):
                    # TODO: can this ever occur?
                    logging.getLogger('comos').error(f"unhandled exception: found no motif aggregate for motif {motif}")
                    exit(1)
                if opt_dir == "min":
                    poi = np.nanargmin(aggregates[0])
                else:
                    poi = np.nanargmax(aggregates[0])
                N = min(Ns[0][poi], len(mu)-1)
                stddev = (best_aggregate - mu[N]) / sigma[N]
                if (opt_dir == "min" and stddev <= thr) or (opt_dir == "max" and stddev >= thr):
                    # add to df
                    new = pd.Series([motif, poi, aggregates[0][poi], Ns[0][poi], stddev, norm().cdf(stddev), True, i], index=d.columns)
                    if ~d.motif.eq(new.motif).any():
                        d.loc[d.index.max()+1] = new
                        changed = True
                        break
            if changed:
                break
    # remove bipartite motifs with missing reverse complement
    sel = d.bipartite & pd.isna(d.rc)
    logging.info(f"removing {sel.sum()} bipartite canonical motifs due to missing RC")
    d = d.drop(d.loc[sel].index)
    
    logging.info(f"{len(d)} canonical motifs remaining after rule-based filtering:")
    print(d)
    
    G = nx.DiGraph()
    G.add_nodes_from(list(d.index))
    # combine motifs
    changed = True
    tested = set()
    while changed == True:
        changed = False
        # find reverse complements
        d['rc'] = None
        for i in d.index:
            for j in d.index:
                if reverse_complement(d.loc[j, 'motif']) == d.loc[i, 'motif']:
                    d.loc[i, 'rc'] = j

        for i in d.index:
            # check if sequence can be mutated to something more significant
            if d.loc[i].motif not in tested:
                tested.add(d.loc[i].motif)
                mutated, probs = mutate_replace(d.loc[i].motif, d.loc[i].poi, sa, len(d.loc[i].motif), mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, min_N=ambiguity_min_sites)
                mutated = "".join([b if p >= 0.9 else o for o,b,p in zip(d.loc[i].motif, mutated, probs)])
                passes, poi, val, N, stddev = test_mutated_motif(mutated, d, i, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, min_k, bases=bases, ambiguity_type=ambiguity_type)
                if passes:
                    # add to df
                    new = pd.Series([mutated, poi, val, N, stddev, norm().cdf(stddev), d.loc[i].bipartite, None], index=d.columns)
                    if ~d.motif.eq(new.motif).any():
                        new_index = d.index.max()+1
                        d.loc[new_index] = new
                        logging.getLogger('comos').info(f"add new mutated motif {new.motif} based on  {d.loc[i].motif}")
                    else:
                        new_index = d.loc[d.motif.eq(new.motif)].index[0]
                    if new_index != i:
                        G.add_edge(new_index, i)
                    
                    # check and add RC
                    # it is not unlikely that the RC would not mutate to the same sequence,
                    # therefore check it here specifically instead of trusting it does.
                    # otherwise the unmutated sequence will be prioritized during resolving the combined graph
                    # TODO: exploded test or not? (see THAF100: RC GYAGNNNNNNGTA fails, but TACNNNNNNCTRC is clearly modified in both locations)
                    if d.loc[i].rc and d.loc[i].rc != i:
                        rc = d.loc[i].rc
                        mutated_rc = reverse_complement(mutated)
                        passes, poi, val, N, stddev = test_mutated_motif(mutated_rc, d, rc, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, min_k, bases=bases, ambiguity_type=ambiguity_type)
                        if passes:
                            # add to df
                            new = pd.Series([mutated_rc, poi, val, N, stddev, norm().cdf(stddev), d.loc[rc].bipartite, None], index=d.columns)
                            if ~d.motif.eq(new.motif).any():
                                new_index = d.index.max()+1
                                d.loc[new_index] = new
                                logging.getLogger('comos').info(f"add new mutated motif {new.motif} based on  {d.loc[rc].motif}, the RC of {d.loc[i].motif}")
                            else:
                                new_index = d.loc[d.motif.eq(new.motif)].index[0]
                            if new_index != rc:
                                G.add_edge(new_index, rc)
                        else:
                            logging.getLogger('comos').info(f"mutated motif {mutated_rc}, the RC of mutated motif {mutated}, failed to pass")



            for j in d.loc[i+1:].index:
                if d.loc[i].bipartite != d.loc[j].bipartite:
                    continue
                if (d.loc[i].motif, d.loc[j].motif) in tested:
                    continue
                tested.add((d.loc[i].motif, d.loc[j].motif))

                best, keep_combined, drop_mi, drop_mj = combine_and_test_motifs(d, i, j, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, min_k, bases=bases, ambiguity_type=ambiguity_type)
                if keep_combined:
                    new = pd.Series(best+[None], index=d.columns)
                    if ~d.motif.eq(new.motif).any():
                        new_index = d.index.max()+1
                        d.loc[new_index] = new
                    else:
                        new_index = d.loc[d.motif.eq(new.motif)].index[0]
                    if new_index != i:
                        G.add_edge(new_index, i)
                    if new_index != j:
                        G.add_edge(new_index, j)
                    changed = True
                elif drop_mi and not drop_mj:
                    G.add_edge(j, i)
                    changed = True
                elif drop_mj and not drop_mi:
                    G.add_edge(i, j)
                    changed = True
                elif drop_mi and drop_mj:
                    logging.getLogger('comos').error(f"unhandled case: both original motifs ({d.loc[i].motif} and {d.loc[j].motif}) shall be dropped")
                    exit(1)

    try:
        nx.find_cycle(G, orientation="original")
        logging.getLogger('comos').error(f"unhandled case: cycle in combined graph")
        exit(1)
    except:
        pass

    draw = False
    if draw:
        fig,ax = plt.subplots(1,1,figsize=(6,4))
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos=pos, ax=ax)
        nx.draw_networkx(G, nodelist=d.loc[d.index.isin(list(G)) & pd.isna(d['rc'])].index, edgelist=[], node_color="red", pos=pos, ax=ax)
        nx.draw_networkx(G, nodelist=d.loc[d.index.isin(list(G)) & (d.index == d.rc)].index, edgelist=[], node_color="green", pos=pos, ax=ax)
        nx.draw_networkx(G, nodelist=[n for n,deg in G.in_degree() if deg==0], edgelist=[], node_color="yellow", node_size=100, pos=pos, ax=ax)
        #if draw in G:
        #    nx.draw_networkx(G, nodelist=[draw], edgelist=G.in_edges(draw), node_color="green", edge_color="green", pos=pos, ax=ax)
        plt.show(block=False)
        print(d)
        breakpoint()

    d['to_prune'] = False
    d = resolve_combined_motif_graph(G, d, sa, max_motif_len, mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, thr, ambiguity_thr, ambiguity_min_sites, ambiguity_min_tests, replace_min_enrichment, bases=bases, ambiguity_type=ambiguity_type)
    logging.info(f"{len(d)} combined motifs remaining after resolving the combined motif graph:")
    print(d)

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
                        d.iloc[j, d.columns.get_loc('representative')] = d.iloc[i].motif
                        changed = True
                        break
                    else:
                        logging.getLogger('comos').debug(f"flagged {d.iloc[i].motif}:{d.iloc[i].poi} ({d.iloc[i].N}, {d.iloc[i].stddevs:.2f}) as reverse complement of {d.iloc[j].motif}:{d.iloc[j].poi} ({d.iloc[j].N}, {d.iloc[j].stddevs:.2f})")
                        d.iloc[i, d.columns.get_loc('representative')] = d.iloc[j].motif
                        changed = True
                        break
            if changed:
                break
    return d.sort_values('stddevs').reset_index(drop=True)

def find_methylated_motifs(all_canon_motifs, fwd_metric, rev_metric, sa, res_dir, contig_id, args):
    if args.metric == "expsumlogp":
        opt_dir = "min"
        selection_thr = -args.selection_thr
        ambiguity_thr = -args.ambiguity_thr
    elif args.metric in ["meanabsdiff", "meanfrac"]:
        opt_dir = "max"
        selection_thr = args.selection_thr
        ambiguity_thr = args.ambiguity_thr
    if args.aggregate == "mean":
        aggr_fct = sanamos.motif_means
    elif args.aggregate == "median":
        aggr_fct = sanamos.motif_medians

    # do computation
    cache_fp = os.path.join(args.cache, f"{os.path.abspath(args.fasta).replace('/','_')}_{contig_id}_{args.bases}_{args.metric}_{args.aggregate}_c{args.min_cov}_w{args.window}_wv{args.min_window_values}_b{args.subtract_background}_k{args.min_k}-{args.max_k}_g{args.min_g}-{args.max_g}.pkl")
    if os.path.exists(cache_fp) and args.cache:
        tstart = time.time()
        with open(cache_fp, 'rb') as f:
            aggr_metric, aggr_metric_counts = pickle.load(f)
        tstop = time.time()
        logging.getLogger('comos').info(f"Loaded canonical motif scores from cache in {tstop - tstart:.2f} seconds")
    else:
        tstart = time.time()
        aggr_metric, aggr_metric_counts = aggr_fct(all_canon_motifs, args.max_k + args.max_g, fwd_metric, rev_metric, sa, bases=args.bases)
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
    mu, sigma, med, iqr = normal_approx(results, outlier_iqr=args.outlier_iqr)
    tstop = time.time()
    logging.getLogger('comos').info(f"Performed normal approximation in {tstop - tstart:.2f} seconds")
    results['stddevs'] = (results.val - mu[results.N]) / sigma[results.N]
    if opt_dir == "min":
        results['p-value'] = norm().cdf(results['stddevs'])
    else:
        results['p-value'] = 1.0 - norm().cdf(results['stddevs'])
    results = results.reset_index().rename(columns={'index':'motif'})

    if args.test_motifs:
        aggr_metric, aggr_metric_counts = aggr_fct(args.test_motifs, np.max([len(m) for m in args.test_motifs]), fwd_metric, rev_metric, sa, bases=args.bases)
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
            passed = test_ambiguous(m, row.poi, sa, len(m), mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, ambiguity_thr, min_N=args.ambiguity_min_sites, debug=True, min_tests=args.ambiguity_min_tests, bases=args.bases, ambiguity_type=args.ambiguity_type)
            print("Passed:", passed, '\n')

            if args.ambiguity_type == 'logo':
                print(f"\nReplace-Test results for motif {m}:{int(row.poi)}")
                passed = test_replace(m, row.poi, sa, len(m), mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, args.replace_min_enrichment, min_N=args.ambiguity_min_sites)
                print("Passed:", passed, '\n')

            print(f"\nExploded-Test results for motif {m}:{int(row.poi)}")
            passed = test_exploded(m, row.poi, sa, len(m), mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, selection_thr, debug=True, bases=args.bases)
            print("Passed:", passed, '\n')

            plot_EDLogo(m, row.poi, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, min_N=args.ambiguity_min_sites, stddev=3.0, n_pseudo=5, Nnull=500)

            print(mutate_replace(m, row.poi, sa, len(m), mu, sigma, fwd_metric, rev_metric, aggr_fct, opt_dir, min_N=args.ambiguity_min_sites))
        
        if args.plot and args.mod.lower().endswith(".rds") and 0:
            fwd_diff, rev_diff = compute_diffs(df, seq)
            for m,row in res.iterrows():
                plot_context_dependent_differences(m, row.poi, sa, fwd_diff, rev_diff, savepath=os.path.join(res_dir, f"{m}_{int(row.poi)}_median.png"))
                plot_diffs_with_complementary(m, sa, fwd_diff, rev_diff, savepath=os.path.join(res_dir, f"{m}_{int(row.poi)}.png"), ymax=10., absolute=True)
        res = res.reset_index().rename(columns={'index':'motif'})
        print(res)
        breakpoint()
        return#exit()
    if args.plot:
        plot_motif_scores(results, mu, sigma, med, iqr, thr=selection_thr, outlier_iqr=args.outlier_iqr, savepath=os.path.join(res_dir, f"scores_scatterplot.png"))
    
    if opt_dir == "min":
        sel = (results.stddevs <= selection_thr)
    else:
        sel = (results.stddevs >= selection_thr)
    logging.getLogger('comos').info(f"Selected {sel.sum():,} motifs based on selection threshold of {selection_thr} std. deviations.")
    print(results.loc[sel].sort_values('stddevs', ascending=(opt_dir == "min")))
    if sel.sum() == 0:
        logging.getLogger('comos').warning("No motifs found given the current set of parameters.")
        return
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
        ambiguity_min_tests=args.ambiguity_min_tests,
        replace_min_enrichment=args.replace_min_enrichment,
        min_k=args.min_k,
        bases=args.bases,
        ambiguity_type=args.ambiguity_type)
    tstop = time.time()
    logging.getLogger('comos').info(f"Performed motif reduction in {tstop - tstart:.2f} seconds")
    motifs_found['palindromic'] = motifs_found['motif'].apply(is_palindromic)
    motifs_found = motifs_found[['motif', 'poi', 'representative', 'bipartite', 'palindromic', 'val', 'N', 'stddevs', 'p-value']]
    logging.getLogger('comos').info(f"Reduced to {len(motifs_found)} motifs in {len(motifs_found.representative.unique())} MTase groups:")
    print(motifs_found.set_index(['representative', 'motif']).sort_values(by=['bipartite', 'palindromic', 'representative', 'stddevs'], ascending=[True, True, True, opt_dir=='min']))
    
    if args.plot and args.mod.lower().endswith('.rds') and 0:
        fwd_diff, rev_diff = compute_diffs(df, seq)
        for r,row in motifs_found.loc[motifs_found.motif == motifs_found.representative].iterrows():
            plot_diffs_with_complementary(row.motif, sa, fwd_diff, rev_diff, absolute=True, savepath=os.path.join(res_dir, f"{row.motif}_{row.poi}.png"))

    motifs_found.to_csv(os.path.join(res_dir, f"results.csv"))

def main():
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, 
                        format=f'%(levelname)-7s [%(asctime)s] : %(message)s',
                        datefmt='%H:%M:%S')

    contigs = parse_contigs(args.fasta)
    sufficient_length = [contig_id for contig_id, seq in contigs.items() if len(seq) >= args.min_seq_len]
    logging.getLogger('comos').info(f"Dataset contains {len(contigs)} contig(s)")
    if not sufficient_length:
        logging.getLogger('comos').error(f"No sequence in {args.fasta} is of sufficient length of >= {args.min_seq_len:,} bp, please adjust argument --min-seq-len")
        exit(1)
    if len(sufficient_length) < len(contigs):
        logging.getLogger('comos').warning(f"Only {len(sufficient_length)} contigs >= {args.min_seq_len:,} bp out of {len(contigs)} are analyzed")
    
    cache_fp = os.path.join(args.cache, f"{os.path.abspath(args.mod).replace('/','_')}.pkl")
    if os.path.exists(cache_fp) and args.cache:
        tstart = time.time()
        df_all = pd.read_pickle(cache_fp)
        tstop = time.time()
        logging.getLogger('comos').info(f"Parsed cached mod data in {tstop - tstart:.2f} seconds")
    else:
        tstart = time.time()
        if args.mod.lower().endswith('.rds'):
            df_all = parse_rds_file(args.mod)
        elif args.mod.lower().endswith('.bed'):
            df_all = parse_bed_file(args.mod)
        else:
            df_all = parse_tombo_files(args.mod)
        tstop = time.time()
        logging.getLogger('comos').info(f"Parsed mod file in {tstop - tstart:.2f} seconds")
        if args.cache:
            df_all.to_pickle(cache_fp)
    
    canon_motifs = get_seq_index_combinations(args.min_k, args.max_k, args.min_g, args.max_g, list(args.bases), 0)
    all_canon_motifs = list(canon_motifs.keys())
    logging.getLogger('comos').info(f"Analyzing {len(canon_motifs):,} canonical motifs and "\
        f"{sum([len(canon_motifs[i]) for i in canon_motifs]):,} indices within these motifs")
    logging.getLogger('comos').info(f"Using {args.metric} metric with {args.aggregate} aggregation of site metrics over {args.window} nt windows (min {args.min_window_values} per window){', with background substraction' if args.subtract_background else ''}")

    if args.metagenome:
        for contig_id in sufficient_length:
            seq = contigs[contig_id]
            cache_fp = ""
            if args.cache:
                cache_fp = os.path.join(args.cache, f"{os.path.abspath(args.fasta).replace('/', '_')}-{contig_id}.idx")
            sa = sanamos.get_suffix_array(contig_id, seq, cache_fp=cache_fp)
            df = df_all.loc[df_all.contig == contig_id]
            if args.metric == "expsumlogp":
                fwd_metric, rev_metric = compute_expsumlogp(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
            elif args.metric == "meanabsdiff":
                fwd_metric, rev_metric = compute_meanabsdiff(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
            elif args.metric == "meanfrac":
                fwd_metric, rev_metric = compute_meanfrac(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
            
            res_dir = os.path.join(args.out, contig_id)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            print()

            logging.getLogger('comos').info(f"Analyzing contig {contig_id} of length {len(contigs[contig_id]):,}")
            find_methylated_motifs(all_canon_motifs, fwd_metric, rev_metric, sa, res_dir, contig_id, args)
    else:
        fwd_metrics, rev_metrics = [], []
        SAs = []
        for contig_id in sufficient_length:
            seq = contigs[contig_id]
            cache_fp = ""
            if args.cache:
                cache_fp = os.path.join(args.cache, f"{os.path.abspath(args.fasta).replace('/', '_')}-{contig_id}.idx")
            sa = sanamos.get_suffix_array(contig_id, seq, cache_fp=cache_fp)
            SAs.append(sa)
            df = df_all.loc[df_all.contig == contig_id]
            if args.metric == "expsumlogp":
                fwd_metric, rev_metric = compute_expsumlogp(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
            elif args.metric == "meanabsdiff":
                fwd_metric, rev_metric = compute_meanabsdiff(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
            elif args.metric == "meanfrac":
                fwd_metric, rev_metric = compute_meanfrac(df, seq, args.min_cov, args.window, args.min_window_values, args.subtract_background)
            fwd_metrics.append(fwd_metric)
            rev_metrics.append(rev_metric)
        
        res_dir = args.out
        find_methylated_motifs(all_canon_motifs, fwd_metrics, rev_metrics, SAs, res_dir, "all", args)

if __name__ == '__main__':
    main()