#!/usr/bin/env python
import argparse
import numpy as np
import os
import torch
import pandas as pd
from torch.nn import functional as F
import einops
from util import vcfData, MEME

torch.backends.cudnn.deterministic = True

def write_output(filename, mat, names):
    pd.DataFrame(mat, columns = names).to_csv(filename, header=True, index=False, sep="\t")

def createmasks(len1, len2, mat):
    for i1, l1 in enumerate(len1):
        for i2, l2 in enumerate(len2):
            mat[i1, i2, int(np.abs(l1-l2+2)):, 0] = 0

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--genome', required = True, help="fasta file for the genome")
    parser.add_argument('-m','--meme', required = True, help="meme file for the motifs")
    parser.add_argument('-v', '--vcf', required = True, help="vcf file")
    parser.add_argument('--batch', default=128, type=int, help="batch size")
    parser.add_argument('-o','--out', required = True, help="output directory")
    parser.add_argument('-t','--transform', default="none", help="transform probabilities to scoring matrix")
    args = parser.parse_args()

    motif = MEME()
    print(f"Reading the motifs from {args.meme}")
    kernels, kernel_mask = motif.parse(args.meme, args.transform)

    print(f"Reading variants from {args.vcf}")
    print(f"Genome: {args.genome}")


    segments = vcfData(args.vcf, args.batch, args.genome, kernels.shape[2])
    segment_mask = segments.return_mask()
    motif_mask = F.conv1d(segment_mask, kernel_mask)
    outRef = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    outAlt = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        matref, matalt = segments[i]
        tmp = F.conv1d(matref, kernels)
        tmp = einops.einsum(tmp, motif_mask, "batch motif length, motif length -> batch motif length")
        tmp = F.max_pool1d(tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2] * 2), tmp.shape[2] * 2).numpy()
        outRef[i1:i2, :motif.nmotifs] = tmp[:, :, 0] 
        tmp = F.conv1d(matalt, kernels)
        tmp = F.max_pool1d(tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2] * 2), tmp.shape[2] * 2).numpy()
        outAlt[i1:i2, :motif.nmotifs] = tmp[:, :, 0] 

    print(f"Writing the results to {args.out}")
    write_output(args.out+".alt", outAlt, motif.names)
    write_output(args.out+".ref", outRef, motif.names)
    write_output(args.out+".diff", outAlt-outRef, motif.names)

if __name__ == "__main__":
    main()
