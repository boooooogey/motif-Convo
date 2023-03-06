#!/usr/bin/env python
import argparse
from torch.nn import functional as F
import pandas as pd
from util import MEME, TFFM, SegmentData, kmers, return_coef_for_normalization, normalize_mat
import torch
import numpy as np
from IPython import embed

torch.backends.cudnn.deterministic = True

def write_output(filename, mat, names, index = None):
    if index is None:
        pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()]).to_csv(filename, sep = "\t", float_format = '%.3f', index = False)
    else:
        pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()], index=index).to_csv(filename, sep = "\t", float_format = '%.3f', index = True)

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
    parser.add_argument('--normalize', action="store_true", help="Apply nonlinear normalization.")
    parser.add_argument('--kernel', choices = ["TFFM", "PWM"], default="PWM", help="Choose between PWM (4 dimensional) or TFFM (16 dimensional) (default = PWM).")
    parser.add_argument('--try-gpu', action="store_true", help="Look for GPU.")
    args = parser.parse_args()
    normalize = args.normalize

    if args.try_gpu:
        device_name = "cuda" if torch.cuda.is_available() else "cpu" 
        device = torch.device(device_name)
    else:
        device_name = "cpu" 

    print(f"Device name: {device_name}")

    print(f"Reading motifs from {args.motifs}")
    if args.kernel == "PWM":
        motif = MEME()
        kernels, _ = motif.parse(args.motifs)#, args.transform)
        if normalize:
            normalization_params = return_coef_for_normalization(kernels)
    elif args.kernel == "TFFM":
        motif = TFFM()
        kernels, _ = motif.parse(args.motifs)
        normalize = False
    print(f"Reading peaks from {args.bed}")
    print(f"Genome: {args.genome}")
    print(f"Total window: {args.window+args.up}")

    segments = SegmentData(args.bed, args.batch, args.genome, int((args.window+args.up)/2), args.up, dinucleotide=(args.kernel == "TFFM"))
    out = np.empty((segments.n, motif.nmotifs+segments.additional))
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        mat, out[i1:i2, motif.nmotifs:] = segments[i]
        if device_name != "cpu":
            tmp = F.conv1d(mat.to(device), kernels.to(device)).cpu()
        else:
            tmp = F.conv1d(mat, kernels)
        if args.mode == "average":
            if args.kernel == "PWM":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
                if normalize:
                    #tmp = torch.from_numpy(normalize_mat(np.exp(tmp), normalization_params))
                    tmp = normalize_mat(np.exp(tmp), normalization_params)
            tmp = F.avg_pool1d(tmp, tmp.shape[2]).numpy()
        if args.mode == "max":
            if args.kernel == "PWM":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
            tmp = F.max_pool1d(tmp, tmp.shape[2])
            if args.kernel == "PWM":
                if normalize:
                    #tmp = torch.from_numpy(normalize_mat(np.exp(tmp), normalization_params))
                    tmp = normalize_mat(np.exp(tmp), normalization_params)
            tmp = tmp.numpy()
        out[i1:i2, :motif.nmotifs] = tmp.reshape(tmp.shape[0], tmp.shape[1])
    print(f"Writing the results to {args.out}")
    write_output(args.out, out, motif.names, segments.names())

if __name__ == "__main__":
    main()
