import os
import argparse
from io import StringIO

import pandas as pd
from Bio import SeqIO
import pyreadr

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

def parse_rds_file(fp):
    df = pyreadr.read_r(fp)[None]
    # correct positions
    # convert to 0-based indexing
    df['position'] = df['position'].astype(int) - 1 + 3 
    df.loc[df.dir == 'rev', 'position'] += 1
    df = df.set_index(df.position)
    return df

def parse_bed_file(fp):
    df = pd.read_csv(
        fp, 
        sep='\s+', 
        header=None, 
        usecols=[0,1,3,5,9,10,11,12,13,14,15,16,17], 
        names=[
            'contig', 'position', 'modbase', 
            'strand', 'cov', 'fracmod', 'Nmod', 
            'Ncan', 'Nother', 'Ndel', 'Nfail', 
            'Ndiff', 'Nnocall'])
    df['position'] = df['position'].astype(int)
    df = df.set_index('position', drop=True)
    df['dir'] = None
    df.loc[df.strand == '+', 'dir'] = 'fwd'
    df.loc[df.strand == '-', 'dir'] = 'rev'
    return df

def parse_tombo_files(prefix):
    def parse_wig_file(fn, col_name, strand):
        dfs = []
        with open(fn, 'r') as fh:
            fcontent = fh.read().split("variableStep chrom=")[1:]
        for i in range(len(fcontent)):
            d = pd.read_csv(StringIO(fcontent[i]), sep=" ")
            contig = d.columns[0]
            d['contig'] = contig
            d['dir'] = strand
            d = d.rename(
                    columns={"span=1":col_name, contig:"position"}
                ).sort_values('position').set_index(
                    ['contig', 'dir', 'position'])
            dfs.append(d)
        return pd.concat(dfs)

    dirname = os.path.dirname(prefix)
    cov_fwd_fn = [
        os.path.join(dirname, fn) for fn in os.listdir(dirname) 
        if (fn.startswith(os.path.basename(prefix)) and 
            fn.endswith('.valid_coverage.plus.wig'))][0]
    cov_rev_fn = [
        os.path.join(dirname, fn) for fn in os.listdir(dirname) 
        if (fn.startswith(os.path.basename(prefix)) and 
            fn.endswith('.valid_coverage.minus.wig'))][0]
    frac_fwd_fn = [
        os.path.join(dirname, fn) for fn in os.listdir(dirname) 
        if (fn.startswith(os.path.basename(prefix)) and 
            fn.endswith('.dampened_fraction_modified_reads.plus.wig'))][0]
    frac_rev_fn = [
        os.path.join(dirname, fn) for fn in os.listdir(dirname) 
        if (fn.startswith(os.path.basename(prefix)) and 
            fn.endswith('.dampened_fraction_modified_reads.minus.wig'))][0]
    
    
    dfs = []
    dfs.append(parse_wig_file(cov_fwd_fn, "cov", "fwd"))
    dfs.append(parse_wig_file(cov_rev_fn, "cov", "rev"))
    dfs.append(parse_wig_file(frac_fwd_fn, "fracmod", "fwd"))
    dfs.append(parse_wig_file(frac_rev_fn, "fracmod", "rev"))

    df = pd.concat(dfs, axis='columns')
    df['cov'] = df['cov'].min(axis=1)
    df['fracmod'] = df['fracmod'].min(axis=1)
    df = df.loc(axis=1)[df.columns.duplicated()]
    df = df.reset_index()[['contig', 'position', 'dir', 'cov', 'fracmod']]
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
            sa = sanamos.get_suffix_array(
                record.id, record.seq, cache_dir=cache_dir)
    return seq, sa, contig_id, n

def parse_contigs(fp):
    contigs = {}
    for record in SeqIO.parse(fp, "fasta"):
        contigs[record.id] = record.seq
    return contigs