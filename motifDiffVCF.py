#!/usr/bin/env python
import argparse
import numpy as np
import os
import torch
import pandas as pd
from torch.nn import functional as F
import einops
from util import vcfData, MEME
from util import MEME_with_Transformation, TFFM, SegmentData, kmers, return_coef_for_normalization, return_coef_for_normalization_diff, normalize_mat
from IPython import embed

torch.backends.cudnn.deterministic = True

def write_output(filename, mat, names, index=None):
    if index is None:
        pd.DataFrame(mat, columns = names).to_csv(filename, header=True, index=False, sep="\t")
    else:
        pd.DataFrame(mat, columns = names, index=index).to_csv(filename, header=True, index=True, sep="\t")

def createmasks(len1, len2, mat):
    for i1, l1 in enumerate(len1):
        for i2, l2 in enumerate(len2):
            mat[i1, i2, int(np.abs(l1-l2+2)):, 0] = 0

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--genome', required = True, help="fasta file for the genome")
    parser.add_argument('-m','--meme', required = True, help="meme file for the motifs")
    parser.add_argument('-v', '--vcf', required = True, help="vcf file")
    parser.add_argument('--batch', default=1024, type=int, help="batch size")
    parser.add_argument('-o','--out', required = True, help="output directory")
    #parser.add_argument('-t','--transform', default="none", help="transform probabilities to scoring matrix")
    args = parser.parse_args()

    motif = MEME_with_Transformation()
    print(f"Reading the motifs from {args.meme}")
    kernels, kernel_mask, kernel_norms = motif.parse(args.meme)#, args.transform)

    print(f"Reading variants from {args.vcf}")
    print(f"Genome: {args.genome}")

    segments = vcfData(args.vcf, args.batch, args.genome, kernels.shape[2])
    outRef = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    outAlt = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    alpha = 0.1
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        matref, maskref, matalt, maskalt = segments[i]
        ref = F.conv1d(matref, kernels)
        motif_mask = F.conv1d(maskref, kernel_mask)
        ref[motif_mask == 0] = -torch.inf
        ref = F.max_pool1d(ref, ref.shape[2]).numpy()
        ref = np.max(ref.reshape(ref.shape[0],-1,2), axis=2)
        outRef[i1:i2, :motif.nmotifs] = ref 

        alt = F.conv1d(matalt, kernels)
        motif_mask = F.conv1d(maskalt, kernel_mask)
        alt[motif_mask == 0] = -torch.inf
        alt = F.max_pool1d(alt, alt.shape[2]).numpy()
        alt = np.max(alt.reshape(alt.shape[0],-1,2), axis=2)
        outAlt[i1:i2, :motif.nmotifs] = alt

    outRef = 1 - outRef/kernel_norms[np.newaxis,:]
    outAlt = 1 - outAlt/kernel_norms[np.newaxis,:]

    mask = outRef > outAlt
    f = np.empty_like(outRef)
    f[np.where(mask)] = -(1-outAlt[mask]+alpha)/(1-outRef[mask]+alpha)+1
    mask = outAlt >= outRef
    f[np.where(mask)] = (1-outRef[mask]+alpha)/(1-outAlt[mask]+alpha)-1
    outDiff = 2/(1+np.power(2, -2*f)) - 1

    print(f"Writing the results to {args.out}")
    #motif_names = []
    motif_names = motif.names

    write_output(args.out+".alt", outAlt, motif_names, segments.names())
    write_output(args.out+".ref", outRef, motif_names, segments.names())
    write_output(args.out+".diff", outDiff, motif_names, segments.names())

if __name__ == "__main__":
    main()
