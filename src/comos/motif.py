from .constants import IUPAC_TO_IUPAC,IUPAC_TO_LIST,IUPAC_NOT

# for fast reverse complement translation
comp_trans = str.maketrans("ACGTMRWSYKVHDBN", "TGCAKYWSRMBDHVN")
def reverse_complement(seq):
    return seq.translate(comp_trans)[::-1]

def is_palindromic(motif):
    return reverse_complement(motif) == motif

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

def get_pois(motif, bases="AC"):
    pois = []
    for i in range(len(motif)):
        if motif[i] in bases:
            pois.append(i)
    return pois

def motif_contains(m1, m2, bipartite=True):
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
            if m2[j] == 'N':
                if bipartite:
                    continue
                else:
                    break
            else:
                if m1[i+j] == 'N':
                    if bipartite:
                        # do not match agaist biparite motif gaps
                        break
                    else:
                        continue
                if m1[i+j] != IUPAC_TO_IUPAC[m1[i+j]][m2[j]]:
                    break
        else:
            return i, identical, diff + [j for j in range(lm2,lm1-i)]
    return None, False, None

def motif_diff(m1, m2, m2_idx, subtract_matching=False):
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
            if subtract_matching:
                if m1[i] == m2[i-m2_idx]:
                    diff.append(m1[i])
                else:
                    bases = IUPAC_TO_LIST[m1[i]][:]
                    for b in IUPAC_TO_LIST[m2[i-m2_idx]]:
                        bases.remove(b)
                    for k in IUPAC_TO_LIST:
                        if bases == IUPAC_TO_LIST[k]:
                            diff.append(k)
                            break
            else:
                # use m1 bases at overlap positions. 
                # These might not be identical to m2, e.g. ASST instead of ACGT
                diff.append(m1[i])
    return "".join(diff).strip('N')

