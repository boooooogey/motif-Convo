#!/usr/bin/env python
import argparse
from torch.nn import functional as F
import pandas as pd
from util import MEME, TFFM, SegmentData, kmers
import torch
import numpy as np

torch.backends.cudnn.deterministic = True

def write_output(filename, mat, names):
    pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()]).to_csv(filename, sep = "\t", float_format = '%.3f', index = False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome', required = True, help="fasta file for the genome")
    parser.add_argument('--motifs', required = True, help="meme file or directory where TFFM XML files located for the motifs")
    parser.add_argument( '--bed', required = True, help="bed file for the peaks")
    parser.add_argument('--batch', default=128, type=int, help="batch size")
    parser.add_argument('--out', required = True, help="output directory")
    parser.add_argument('--window', default=240, type=int, help="window size")
    parser.add_argument('--up', default=0, type=int, help="add upstream")
    parser.add_argument('--transform', choices =["none", "constant", "local"], default="none", help="transform probabilities to scoring matrix")
    parser.add_argument('--mode', choices = ["max", "average"], default="max", help="Operation mode for the pooling layer (max/average)")
    parser.add_argument('--kernel', choices = ["TFFM", "PWM"], default="PWM", help="Choose between PWM (4 dimensional) or TFFM (16 dimensional) (default = PWM).")
    args = parser.parse_args()

    print(f"Reading motifs from {args.motifs}")
    if args.kernel == "PWM":
        motif = MEME()
        kernels, _ = motif.parse(args.motifs, args.transform)
    elif args.kernel == "TFFM":
        motif = TFFM()
        kernels, _ = motif.parse(args.motifs)
    print(f"Reading peaks from {args.bed}")
    print(f"Genome: {args.genome}")
    print(f"Total window: {args.window+args.up}")

    segments = SegmentData(args.bed, args.batch, args.genome, int(args.window+args.up/2), args.up, dinucleotide=(args.kernel == "TFFM"))
    out = np.empty((segments.n, motif.nmotifs+segments.additional))
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
        if args.kernel == "TFFM":
            out[i1:i2, :motif.nmotifs] = tmp.reshape(tmp.shape[0], tmp.shape[1])
        elif args.kernel == "PWM":
            out[i1:i2, :motif.nmotifs] = np.max(tmp.reshape(tmp.shape[0],-1,2), axis=2)
    print(f"Writing the results to {args.out}")
    write_output(args.out, out, motif.names)

if __name__ == "__main__":
    main()
