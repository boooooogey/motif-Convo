#!/usr/bin/env python
import argparse
import numpy as np
import os
import torch
import pandas as pd
from torch.nn import functional as F
import einops
from util import vcfData, MEME
from util import MEME, TFFM, SegmentData, kmers, return_coef_for_normalization, return_coef_for_normalization_diff, normalize_mat
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
    parser.add_argument('-t','--transform', default="none", help="transform probabilities to scoring matrix")
    parser.add_argument('--normalize', action="store_true", help="Apply nonlinear normalization.")
    #parser.add_argument('--adapt-cdf', action="store_true", help="Correct CDF of the distribution of maximum of the random variables.")
    parser.add_argument('--cdf-power', default=1, type=float, help="Correct CDF of the distribution of maximum of the random variables.")
    args = parser.parse_args()
    normalize = args.normalize
    dist_correction = args.cdf_power

    motif = MEME()
    print(f"Reading the motifs from {args.meme}")
    kernels, kernel_mask = motif.parse(args.meme, args.transform)
    if normalize:
        print("Calculating normalization parameters")
        #normalization_params = return_coef_for_normalization(kernels)
        normalization_params = return_coef_for_normalization_diff(kernels, length_correction=dist_correction)

    print(f"Reading variants from {args.vcf}")
    print(f"Genome: {args.genome}")

    segments = vcfData(args.vcf, args.batch, args.genome, kernels.shape[2])
    outRef = np.empty((segments.n, motif.nmotifs*2), dtype=np.float32)
    outAlt = np.empty((segments.n, motif.nmotifs*2), dtype=np.float32)
    #outRef = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    #outAlt = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        matref, maskref, matalt, maskalt = segments[i]
        tmp = F.conv1d(matref, kernels)
        motif_mask = F.conv1d(maskref, kernel_mask)
        #tmp = einops.einsum(tmp, motif_mask, "batch motif length, batch motif length -> batch motif length")
        tmp[motif_mask == 0] = -torch.inf
        tmp = F.max_pool1d(tmp, tmp.shape[2])
        #if normalize:
        #    tmp = normalize_mat(tmp, normalization_params)
        outRef[i1:i2, :motif.nmotifs*2] = tmp.numpy().reshape(tmp.shape[0], -1) #np.max(tmp.view(tmp.shape[0],-1,2).numpy(), axis=2)
        #outRef[i1:i2, :motif.nmotifs] = np.max(tmp.view(tmp.shape[0],-1,2).numpy(), axis=2)
        tmp = F.conv1d(matalt, kernels)
        motif_mask = F.conv1d(maskalt, kernel_mask)
        #tmp = einops.einsum(tmp, motif_mask, "batch motif length, batch motif length -> batch motif length")
        tmp[motif_mask == 0] = -torch.inf
        tmp = F.max_pool1d(tmp, tmp.shape[2])
        #if normalize:
        #    tmp = normalize_mat(tmp, normalization_params)
        outAlt[i1:i2, :motif.nmotifs*2] = tmp.numpy().reshape(tmp.shape[0], -1)
        #outAlt[i1:i2, :motif.nmotifs] = np.max(tmp.view(tmp.shape[0],-1,2).numpy(), axis=2)

    print(f"Writing the results to {args.out}")
    #motif_names = []
    motif_names = motif.names
    #for i in range(len(motif.names)):
    #    motif_names.append(motif.names[i])
    #    motif_names.append(motif.names[i]+"_reversed")
    outAlt = outAlt.reshape(outAlt.shape[0], -1, 2)
    outRef = outRef.reshape(outAlt.shape[0], -1, 2)
    diff = np.abs(outAlt - outRef)
    ii = diff.argmax(axis=2)
    outAlt = np.take_along_axis(outAlt, np.expand_dims(ii, axis=2), axis=2).reshape(outAlt.shape[0], -1)
    outRef = np.take_along_axis(outRef, np.expand_dims(ii, axis=2), axis=2).reshape(outRef.shape[0], -1)

    write_output(args.out+".alt", outAlt, motif_names, segments.names())
    write_output(args.out+".ref", outRef, motif_names, segments.names())
    #write_output(args.out+".diff", outAlt-outRef, motif_names, segments.names())
    #outAlt = pd.read_csv("local/evalenformer/convo_features_diff_norm/human_vcf_motifconvo_features_normalized_correction1_split_1.alt", sep="\t", header=0, index_col=0).to_numpy()
    #outRef = pd.read_csv("local/evalenformer/convo_features_diff_norm/human_vcf_motifconvo_features_normalized_correction1_split_1.ref", sep="\t", header=0, index_col=0).to_numpy()

    if normalize:
        write_output(args.out+".diff", normalize_mat(outAlt-outRef, normalization_params).numpy(), motif_names, segments.names())
    else:
        write_output(args.out+".diff", outAlt-outRef, motif_names, segments.names())

if __name__ == "__main__":
    main()
