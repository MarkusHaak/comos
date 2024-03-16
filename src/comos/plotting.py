import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt

def plot_context_dependent_differences(motif, poi, sa, fwd_diff, rev_diff, ymax=np.inf, pad=2, absolute=False, savepath=None):
    poi = int(poi)
    mlen = len(motif)

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

    medians_fwd, Ns_med_fwd = sanamos.all_motif_medians(expanded_motifs, len(expanded_motifs[0]), fwd_diff, rev_diff, sa, pad=0)
    medians_fwd = medians_fwd[:, positions]
    Ns_med_fwd = Ns_med_fwd[:, positions]

    positions = list(range(poi-pad, poi+pad+1))
    ind_fwd, ind_rev = sanamos.find_motif(motif, sa)
    # original strand
    n_fwd, n_rev = len(ind_fwd), len(ind_rev)
    data_fwd = []
    for i in positions:
        fwd_diffs = fwd_diff[(ind_fwd + i)[(ind_fwd + i >= 0) & (ind_fwd + i < len(fwd_diff))]]
        rev_diffs = rev_diff[(ind_rev - i)[(ind_rev - i >= 0) & (ind_rev - i < len(rev_diff))]]
        diffs = np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]])
        data_fwd.append(diffs)
    
    fig,ax = plt.subplots()
    ax.grid(axis="y", zorder=-100)
    boxprops = dict(linewidth=2.0, color='black')
    whiskerprops = dict(linewidth=2.0, color='black')
    medianprops = dict(linewidth=2.0, color='black', linestyle=":")
    if absolute is False:
        ax.boxplot(data_fwd, boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops)
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
    ind_fwd, ind_rev = sanamos.find_motif(motif, sa)
    n_fwd, n_rev = len(ind_fwd), len(ind_rev)
    ax1.set_title(f"{motif}, N={n_fwd+n_rev:,}")
    boxes, labels = [], []
    for c, (m, offset) in enumerate(zip(expl_motifs, offsets)):
        ind_fwd, ind_rev = sanamos.find_motif(m, sa)
        # original strand
        n_fwd, n_rev = len(ind_fwd), len(ind_rev)
        data_fwd = []
        for i in positions:
            fwd_diffs = fwd_diff[(ind_fwd + i)[(ind_fwd + i >= 0) & (ind_fwd + i < len(fwd_diff))]]
            rev_diffs = rev_diff[(ind_rev - i)[(ind_rev - i >= 0) & (ind_rev - i < len(rev_diff))]]
            if absolute:
                diffs = np.abs(np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]]))
            else:
                diffs = np.concatenate([fwd_diffs[~np.isnan(fwd_diffs)], rev_diffs[~np.isnan(rev_diffs)]])
            data_fwd.append(diffs)
        # complementary strand
        n_fwd_c, n_rev_c = len(ind_fwd), len(ind_rev)
        data_rev = []
        for i in positions:
            fwd_diffs = rev_diff[(ind_fwd + i)[(ind_fwd + i >= 0) & (ind_fwd + i < len(fwd_diff))]]
            rev_diffs = fwd_diff[(ind_rev - i)[(ind_rev - i >= 0) & (ind_rev - i < len(rev_diff))]]
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
        labels.append(f"{m}, n={n_fwd+n_rev:>4}")

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
    
def plot_motif_scores(results, mu, sigma, med, iqr, thr=6., outlier_iqr=0., savepath=None):
    fig,ax = plt.subplots(figsize=(6,6))
    ax.scatter(results.N,results.val,s=1,alpha=0.25, color='black')
    X = np.unique(np.linspace(0,mu.shape[0]-1, min(mu.shape[0]-1, 1000)).astype(int))
    ax.plot(X, mu[X], color='C0')
    ax.plot(X, mu[X] + thr*sigma[X], color='C0', linestyle=':')
    if outlier_iqr > 0.:
        ax.plot(X, med[X] + outlier_iqr*iqr[X], color='C1', linestyle=':')
    ax.grid()
    ax.set(ylim=(-0.0005, ax.get_ylim()[1]),
           xlim=(-50,5000),
        xlabel="# motif sites", ylabel="aggregated motif metric")
    if savepath:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

def letterAt(letter, x, y, yscale=1, ax=None):
    letters = { "T" : TextPath((-0.305, 0), "T", size=1, prop=FontProperties(family="Arial", weight="bold") ),
                "G" : TextPath((-0.384, 0), "G", size=1, prop=FontProperties(family="Arial", weight="bold") ),
                "A" : TextPath((-0.350, 0), "A", size=1, prop=FontProperties(family="Arial", weight="bold") ),
                "C" : TextPath((-0.366, 0), "C", size=1, prop=FontProperties(family="Arial", weight="bold") ) }
    color_scheme = {'G': 'orange', 
                    'A': 'red', 
                    'C': 'blue', 
                    'T': 'darkgreen'}
    globscale = 1.35
    text = letters[letter]
    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=color_scheme[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def plot_EDLogo(motif, poi, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, min_N=10, stddev=3.0, n_pseudo=5, Nnull=500):
    poi = int(poi)
    fig,(ax_frac,ax) = plt.subplots(2,1,sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace':0.05})
    
    Nval = []
    fracmod = []
    all_scores_positive = []
    all_scores_negative = []
    for ind in range(-1,len(motif)+1):
        df = get_EDLogo_enrichment_scores(motif, poi, ind, sa, mu, sigma, fwd_metric, rev_metric, opt_dir, min_N=min_N, stddev=stddev, n_pseudo=n_pseudo, Nnull=Nnull)
        all_scores_positive.append(list(df.loc[df.r >= 0][['base', 'r']].itertuples(index=False, name=None)))
        all_scores_negative.append(list(df.loc[df.r < 0][['base', 'r']].itertuples(index=False, name=None)))
        Nval.append(df.Nval.sum())
        fracmod.append(df['mod'].sum() / df['Nval'].sum())
    
    ax_frac.plot(range(-1,len(motif)+1), fracmod, color="C1")
    ax_frac_twin = ax_frac.twinx()
    ax_frac_twin.plot(range(-1,len(motif)+1), Nval, color="C2")
    ax_frac.spines['left'].set_color('C1')
    ax_frac.spines['right'].set_color('C2')
    ax_frac.yaxis.label.set_color('C1')
    ax_frac_twin.yaxis.label.set_color('C2')
    ax_frac.set(ylabel="fraction\nmodified")
    ax_frac_twin.set(ylabel="valid\ncoverage")
    
    x = -1
    maxi = 0
    for scores in all_scores_positive:
        y = 0
        for base, score in scores:
            letterAt(base, x,y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)

    x = -1
    mini = 0
    for scores in all_scores_negative:
        y = 0
        for base, score in scores:
            y += score
            letterAt(base, x,y, np.abs(score), ax)
        x += 1
        mini = min(mini, y)
    ax.set_xticks(range(-1,x))
    ax.set_xlim((-2, x))
    ax.set_ylim(mini-0.1, maxi+0.1) 

    for i in range(0, len(motif)):
        if motif[i] != 'N':
            ax.axvspan(i-0.5, i+0.5, facecolor='black', alpha=0.2, zorder=-100)

    motif_exp = ["N"] + list(motif[:poi]) + [motif[poi]+'*'] + list(motif[poi+1:]) + ["N"]
    ax.set_xticklabels(motif_exp)
    ax.axhline(0, linewidth=1.)
    plt.show(block=False)