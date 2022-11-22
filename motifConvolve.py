#!/usr/bin/env python
import argparse
from torch.nn import functional as F
import pandas as pd
from util import MEME, SegmentData
import torch
import numpy as np

torch.backends.cudnn.deterministic = True

def write_output(filename, mat, names):
    pd.DataFrame(mat, columns = names + ["GC_ratio", "GC_pattern", "CG_pattern", "Masked_ratio"]).to_csv(filename, sep = "\t", float_format = '%.3f', index = False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--genome', required = True, help="fasta file for the genome")
    parser.add_argument('-m','--meme', required = True, help="meme file for the motifs")
    parser.add_argument('-b', '--bed', required = True, help="bed file for the peaks")
    parser.add_argument('--batch', default=128, type=int, help="batch size")
    parser.add_argument('-o','--out', required = True, help="output directory")
    parser.add_argument('-w','--window', default=240, type=int, help="window size")
    parser.add_argument('-u','--up', default=0, type=int, help="add upstream")
    parser.add_argument('-t','--transform', choices =["none", "constant", "local"], default="none", help="transform probabilities to scoring matrix")
    parser.add_argument('-d','--mode', choices = ["max", "average"], default="max", help="Operation mode for the pooling layer (max/average)")
    args = parser.parse_args()

    motif = MEME()
    print(f"Reading motifs from {args.meme}")
    kernels, _ = motif.parse(args.meme, args.transform)
    print(f"Reading peaks from {args.bed}")
    print(f"Genome: {args.genome}")
    print(f"Total window: {args.window+args.up}")

    segments = SegmentData(args.bed, args.batch, args.genome, int(args.window+args.up/2), args.up)
    out = np.empty((segments.n, motif.nmotifs+4))
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        mat, out[i1:i2, motif.nmotifs:] = segments[i]
        tmp = F.conv1d(mat, kernels)
        if args.mode == "average":
            tmp = F.avg_pool1d(tmp, tmp.shape[2]).numpy()
        if args.mode == "max":
            tmp = F.max_pool1d(tmp, tmp.shape[2]).numpy()
        out[i1:i2, :motif.nmotifs] = np.max(tmp.reshape(tmp.shape[0],-1,2), axis=2)
    print(f"Writing the results to {args.out}")
    write_output(args.out, out, motif.names)

if __name__ == "__main__":
    main()
