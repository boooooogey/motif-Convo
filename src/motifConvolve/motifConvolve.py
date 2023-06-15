#!/usr/bin/env python
from torch.nn import functional as F
import pandas as pd
from motifConvolve.util import vcfData, MEME, MEME_with_Transformation, TFFM, SegmentData, kmers, return_coef_for_normalization, return_coef_for_normalization_diff, normalize_mat, kmers
import torch
import numpy as np
from enum import Enum
import os
import einops
import typer

app = typer.Typer()

torch.backends.cudnn.deterministic = True

def write_output_diff(filename, mat, names, index=None):
    if index is None:
        pd.DataFrame(mat, columns = names).to_csv(filename, header=True, index=False, sep="\t")
    else:
        pd.DataFrame(mat, columns = names, index=index).to_csv(filename, header=True, index=True, sep="\t")

def createmasks(len1, len2, mat):
    for i1, l1 in enumerate(len1):
        for i2, l2 in enumerate(len2):
            mat[i1, i2, int(np.abs(l1-l2+2)):, 0] = 0

def write_output_motif_features(filename, mat, names, index = None):
    if index is None:
        pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()]).to_csv(filename, sep = "\t", float_format = '%.3f', index = False)
    else:
        pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()], index=index).to_csv(filename, sep = "\t", float_format = '%.3f', index = True)

class transform_type(str, Enum):
    none = "none"
    constant = "constant"
    local = "local"

class mode_type(str, Enum):
    max = "max"
    average = "average"

class kernel_type(str, Enum):
    TFFM = "TFFM"
    PWM = "PWM"

@app.command()
def tfextract(genome: str = typer.Option(..., help="fasta file for the genome"),
                  motif_file: str = typer.Option(..., "--motif", help="meme file or directory where TFFM XML files located for the motifs"), 
                  bed: str = typer.Option(..., help="bed file for the peaks"), 
                  batch: int = typer.Option(128, help="batch size"),
                  out_file: str = typer.Option(..., "--out", help="output directory"),
                  window: int = typer.Option(240, help="window size"),
                  up: int = typer.Option(0, help="add upstream"),
                  transform: transform_type = typer.Option(transform_type.none, help="transform probabilities to scoring matrix"),
                  mode: mode_type = typer.Option(mode_type.max, help="Operation mode for the pooling layer (max/average)"),
                  normalize: bool = typer.Option(False, help="Apply nonlinear normalization."),
                  kernel: kernel_type = typer.Option(kernel_type.PWM, help="Choose between PWM (4 dimensional) or TFFM (16 dimensional) (default = PWM)."),
                  try_gpu: bool = typer.Option(False, "--try-gpu", help="Look for GPU.")
                  ):
    if try_gpu:
        device_name = "cuda" if torch.cuda.is_available() else "cpu" 
        device = torch.device(device_name)
    else:
        device_name = "cpu" 

    print(f"Device name: {device_name}")
    kernel = kernel.value
    mode = mode.value
    transform = transform.value

    print(f"Reading motifs from {motif_file}")
    if kernel == "PWM":
        motif = MEME()
        kernels, _ = motif.parse(motif_file)#, args.transform)
        if normalize:
            normalization_params = return_coef_for_normalization(kernels)
    elif kernel == "TFFM":
        motif = TFFM()
        kernels, _ = motif.parse(motif_file)
        normalize = False
    print(f"Reading peaks from {bed}")
    print(f"Genome: {genome}")
    print(f"Total window: {window+up}")

    segments = SegmentData(bed, batch, genome, int((window+up)/2), up, dinucleotide=(kernel == "TFFM"))
    out = np.empty((segments.n, motif.nmotifs+segments.additional))
    print(f"Batch size: {batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*batch, (i+1)*batch
        if i2 >= segments.n: i2 = segments.n
        mat, out[i1:i2, motif.nmotifs:] = segments[i]
        if device_name != "cpu":
            tmp = F.conv1d(mat.to(device), kernels.to(device)).cpu()
        else:
            tmp = F.conv1d(mat, kernels)
        if mode == "average":
            if kernel == "PWM":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
                if normalize:
                    #tmp = torch.from_numpy(normalize_mat(np.exp(tmp), normalization_params))
                    tmp = normalize_mat(np.exp(tmp), normalization_params)
            tmp = F.avg_pool1d(tmp, tmp.shape[2]).numpy()
        if mode == "max":
            if kernel == "PWM":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
            tmp = F.max_pool1d(tmp, tmp.shape[2])
            if kernel == "PWM":
                if normalize:
                    #tmp = torch.from_numpy(normalize_mat(np.exp(tmp), normalization_params))
                    tmp = normalize_mat(np.exp(tmp), normalization_params)
            tmp = tmp.numpy()
        out[i1:i2, :motif.nmotifs] = tmp.reshape(tmp.shape[0], tmp.shape[1])
    print(f"Writing the results to {out_file}")
    write_output_motif_features(out_file, out, motif.names, segments.names())

@app.command()
def variantdiff(genome: str = typer.Option(..., help="fasta file for the genome"),
                 motif_file: str = typer.Option(..., "--motif", help="meme file for the motifs"),
                 vcf: str = typer.Option(..., help="vcf file"), 
                 batch: int = typer.Option(128, help="batch size"),
                 out_file: str = typer.Option(..., "--out", help="output directory"),
                 kernel: kernel_type = typer.Option(kernel_type.PWM, help="Choose between PWM (4 dimensional) or TFFM (16 dimensional) (default = PWM)."),
                ):



    kernel = kernel.value
    if kernel == "PWM":
        motif = MEME_with_Transformation()
        print(f"Reading the motifs from {motif_file}")
        kernels, kernel_mask, kernel_norms = motif.parse(motif_file)#, args.transform)
        windowsize = kernels.shape[2]
        #motif = MEME()
        #kernels, _ = motif.parse(motif_file)#, args.transform)
        #if normalize:
        #    normalization_params = return_coef_for_normalization(kernels)
    elif kernel == "TFFM":
        motif = TFFM()
        print(f"Reading the motifs from {motif_file}")
        kernels, kernel_mask, kernel_norms = motif.parse(motif_file)#, args.transform)
        windowsize = kernels.shape[2] + 1
        #motif = MEME()
        #kernels, _ = motif.parse(motif_file)
        #normalize = False

    print(f"Reading variants from {vcf}")
    print(f"Genome: {genome}")

    segments = vcfData(vcf, batch, genome, windowsize, dinucleotide=(kernel == "TFFM"))
    outRef = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    outAlt = np.empty((segments.n, motif.nmotifs), dtype=np.float32)
    alpha = 0.1
    print(f"Batch size: {batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*batch, (i+1)*batch
        if i2 >= segments.n: i2 = segments.n
        matref, maskref, matalt, maskalt = segments[i]
        ref = F.conv1d(matref, kernels)
        motif_mask = F.conv1d(maskref, kernel_mask)
        ref[motif_mask == 0] = -torch.inf
        ref = F.max_pool1d(ref, ref.shape[2]).numpy()
        if kernel == "PWM":
            ref = np.max(ref.reshape(ref.shape[0],-1,2), axis=2)
        outRef[i1:i2, :motif.nmotifs] = ref 

        alt = F.conv1d(matalt, kernels)
        motif_mask = F.conv1d(maskalt, kernel_mask)
        alt[motif_mask == 0] = -torch.inf
        alt = F.max_pool1d(alt, alt.shape[2]).numpy()
        if kernel == "PWM":
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

    print(f"Writing the results to {out_file}")
    #motif_names = []
    motif_names = motif.names

    write_output_diff(out_file+".alt", outAlt, motif_names, segments.names())
    write_output_diff(out_file+".ref", outRef, motif_names, segments.names())
    write_output_diff(out_file+".diff", outDiff, motif_names, segments.names())

if __name__ == "__main__":
    app()
