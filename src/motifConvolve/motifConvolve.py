#!/usr/bin/env python
from torch.nn import functional as F
import pandas as pd
from utilNEW import vcfData, MEME_probNorm, MEME_FABIAN, TFFM, TFFM_with_Transformation, SegmentData, kmers, return_coef_for_normalization, MCspline_fitting, normalize_mat, mc_spline, kmers #
import torch
import numpy as np
from enum import Enum
import os
import einops
import typer

import pyarrow.feather as feather


app = typer.Typer()

torch.backends.cudnn.deterministic = True

def write_output_diff(filename, mat, names, index=None):
    if index is None:
        pd.DataFrame(mat, columns = names).to_csv(filename, header=True, index=False, sep="\t")
        #feather.write_feather(pd.DataFrame(mat, columns = names), f"{filename}.feather")
    else:
        pd.DataFrame(mat, columns = names, index=index).to_csv(filename, header=True, index=True, sep="\t")
        #feather.write_feather(pd.DataFrame(mat, columns = names, index=index), f"{filename}.feather")

def createmasks(len1, len2, mat):
    for i1, l1 in enumerate(len1):
        for i2, l2 in enumerate(len2):
            mat[i1, i2, int(np.abs(l1-l2+2)):, 0] = 0

def write_output_motif_features(filename, mat, names, index = None):
    if index is None:
        pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()]).to_csv(filename, sep = "\t", float_format = '%.3f', index = False)
        #feather.write_feather(pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()]), f"{filename}.feather")
    else:
        pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()], index=index).to_csv(filename, sep = "\t", float_format = '%.3f', index = True)
        #feather.write_feather(pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()], index=index), f"{filename}.feather")

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
    
class normalize_type(str, Enum):
    spline = "spline"
    logit = "logit"

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
                  #normalize: bool = typer.Option(False, help="Apply nonlinear normalization."),
                  normalize: normalize_type = typer.Option(normalize_type.logit, help="what normalization function to use (spline/logit)"),
                  kernel: kernel_type = typer.Option(kernel_type.PWM, help="Choose between PWM (4 dimensional) or TFFM (16 dimensional) (default = PWM)."),
                  bin: int = typer.Option(1, help="number of bins"),
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
    normalize = normalize.value

    print(f"Reading motifs from {motif_file}")
    if kernel == "PWM":
        motif = MEME_probNorm()
        kernels, _ = motif.parse(motif_file) #, transform)
        if normalize == "logit":
            normalization_params = return_coef_for_normalization(kernels)
        if normalize == "spline":
            spline_list = MCspline_fitting(kernels)
    elif kernel == "TFFM":
        motif = TFFM()
        kernels, _ = motif.parse(motif_file)
        normalize = False
    print(f"Reading peaks from {bed}")
    print(f"Genome: {genome}")
    print(f"Total window: {window+up}")

    segments = SegmentData(bed, batch, genome, int((window+up)/2), up, dinucleotide=(kernel == "TFFM"))
    out = np.empty((segments.n, bin*motif.nmotifs+segments.additional))
    
    print(f"Batch size: {batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*batch, (i+1)*batch
        if i2 >= segments.n: i2 = segments.n
        mat, out[i1:i2, bin*motif.nmotifs:] = segments[i]
        if device_name != "cpu":
            tmp = F.conv1d(mat.to(device), kernels.to(device)).cpu()
        else:
            tmp = F.conv1d(mat, kernels)
        if mode == "average":
            if kernel == "PWM":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
                if normalize == "logit":
                    tmp = normalize_mat(tmp, normalization_params)
                if normalize == "spline":    
                    tmp = mc_spline(tmp, spline_list)
            tmp = np.nan_to_num(tmp, nan=0)
            tmp = F.avg_pool1d(torch.tensor(tmp), int(tmp.shape[2]/bin)).numpy()
        if mode == "max":
            if kernel == "PWM":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
            tmp = np.nan_to_num(tmp, nan=0)
            tmp = F.max_pool1d(torch.tensor(tmp), int(tmp.shape[2]/bin)).numpy()
            if kernel == "PWM":
                if normalize == "logit":
                    tmp = normalize_mat(tmp, normalization_params)
                if normalize == "spline":
                    tmp = mc_spline(tmp, spline_list)
            tmp = tmp.numpy()
        out[i1:i2, :bin*motif.nmotifs] = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
    print(f"Writing the results to {out_file}")
    
    names = motif.names
    new_names =[]
    for i in range(bin*len(names)):
        if i%bin>int(bin/2): new_names.append(f'{names[int(i/bin)]} - right - {i%bin}'); #print(f'new_names{i} = names {int(i/args.bin)} which is "{names[int(i/args.bin)]}" - right')
        if i%bin==int(bin/2): new_names.append(f'{names[int(i/bin)]} - middle'); #print(f'new_names{i} = names {int(i/args.bin)} which is "{names[int(i/args.bin)]}" - middle')
        if i%bin<int(bin/2): new_names.append(f'{names[int(i/bin)]} - left - {i%bin}'); #print(f'new_names{i} = names {int(i/args.bin)} which is "{names[int(i/args.bin)]}" - left')

    write_output_motif_features(out_file, out, new_names, segments.names())


class score_type(str, Enum):
    FABIAN = "FABIAN"
    probNorm = "probNorm"
    NONE = "NONE"
    
class nucleotide_type(str, Enum):
    mono = "mono"
    di = "di"

@app.command()
def variantdiff(genome: str = typer.Option(..., help="fasta file for the genome"),
                 motif_file: str = typer.Option(..., "--motif", help="meme file for the motifs"),
                 vcf: str = typer.Option(..., help="vcf file"), 
                 diff_score: score_type = typer.Option(score_type.FABIAN, "--score", help="how to calculate the diff score (FABIAN/probNorm/NONE)"),
                 nucleotide: nucleotide_type = typer.Option(nucleotide_type.mono, "--nuc", help="length of the nucleotides in the motifs (mono/di)"),
                 normalize: normalize_type = typer.Option(normalize_type.logit, help="what normalization function to use (spline/logit)"),
                 mode: mode_type = typer.Option(mode_type.max, help="Operation mode for the pooling layer (max/average)"),                 
                 batch: int = typer.Option(128, help="batch size"),
                 out_file: str = typer.Option(..., "--out", help="output directory"),
                 window:int = typer.Option(0, help="window size"), # change this in a way that if it gets a value use that value and if not the default will be kernel size (instead of setting 0 as the default, make it none or not receiving an input as the default)
                 kernel: kernel_type = typer.Option(kernel_type.PWM, help="Choose between PWM (4 dimensional) or TFFM (16 dimensional) (default = PWM).")
                ):

    kernel = kernel.value
    if kernel == "PWM":
        if diff_score == "FABIAN":
            motif = MEME_FABIAN()
            print(f"Reading the motifs from {motif_file}")
            kernels, kernel_mask, kernel_norms = motif.parse(motif_file, nuc=nucleotide)#, args.transform)
        else:
            motif = MEME_probNorm()
            kernels, kernel_mask = motif.parse(motif_file, nuc=nucleotide) #, transform)
            if normalize == "logit":
                normalization_params = return_coef_for_normalization(kernels, nuc=nucleotide)
            if normalize == "spline":
                spline_list = MCspline_fitting(kernels, nuc=nucleotide)
        
        if window==0:
            windowsize = kernels.shape[2]
        else:
            windowsize=window


    elif kernel == "TFFM":
       motif = TFFM_with_Transformation()
       print(f"Reading the motifs from {motif_file}")
       kernels, kernel_mask, kernel_norms = motif.parse(motif_file)#, args.transform)
       if window==0:
            windowsize = kernels.shape[2]+1
       else:
            windowsize=window
    
    print("windowsize is:", windowsize)
    
    segments = vcfData(vcf, batch, genome, windowsize, dinucleotide=(nucleotide=="di"))
    
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
        alt = F.conv1d(matalt, kernels)
        
        if window==0:
            motif_mask = F.conv1d(maskref, kernel_mask)
            ref[motif_mask == 0] = -torch.inf
            motif_mask = F.conv1d(maskalt, kernel_mask)        
            alt[motif_mask == 0] = -torch.inf
        
        if diff_score == "NONE":
            if kernel == "PWM":
                ref = ref.view(ref.shape[0], ref.shape[1]//2, ref.shape[2]*2)
                alt = alt.view(alt.shape[0], alt.shape[1]//2, alt.shape[2]*2)
            ref = F.max_pool1d(ref, ref.shape[2])
            alt = F.max_pool1d(alt, alt.shape[2])
            ref = np.squeeze(ref).numpy()
            alt = np.squeeze(alt).numpy()
            
        if diff_score == "probNorm":
            if mode == "average":
                if kernel == "PWM":
                    ref = ref.view(ref.shape[0], ref.shape[1]//2, ref.shape[2]*2)
                    alt = alt.view(alt.shape[0], alt.shape[1]//2, alt.shape[2]*2)                    
                    if normalize == "logit":
                        ref = normalize_mat(ref, normalization_params)
                        alt = normalize_mat(alt, normalization_params)
                    if normalize == "spline":
                        ref = mc_spline(ref, spline_list)
                        alt = mc_spline(alt, spline_list)                 
                ref = np.nan_to_num(ref, nan=0)
                alt = np.nan_to_num(alt, nan=0)               
                ref = F.avg_pool1d(torch.tensor(ref), ref.shape[2])
                alt = F.avg_pool1d(torch.tensor(alt), alt.shape[2])        
                ref = np.squeeze(ref).numpy()
                alt = np.squeeze(alt).numpy()

            if mode == "max":
                if kernel == "PWM":
                    ref = ref.view(ref.shape[0], ref.shape[1]//2, ref.shape[2]*2)
                    alt = alt.view(alt.shape[0], alt.shape[1]//2, alt.shape[2]*2)
                ref = np.nan_to_num(ref, nan=0)
                alt = np.nan_to_num(alt, nan=0)
                ref = F.max_pool1d(torch.tensor(ref), ref.shape[2])
                alt = F.max_pool1d(torch.tensor(alt), alt.shape[2])
                if kernel == "PWM":
                    if normalize == "logit":
                        ref = normalize_mat(ref, normalization_params)
                        alt = normalize_mat(alt, normalization_params)
                    if normalize == "spline":
                        ref = mc_spline(ref, spline_list)
                        alt = mc_spline(alt, spline_list)
                ref = np.squeeze(ref).numpy()
                alt = np.squeeze(alt).numpy()
        
        if diff_score == "FABIAN":                 
            ref = F.max_pool1d(ref, ref.shape[2]).numpy()
            alt = F.max_pool1d(alt, alt.shape[2]).numpy()
            ref = np.max(ref.reshape(ref.shape[0],-1,2), axis=2) # separates the convolutions from the original kernel and the reverse complement kernel into two differetn columns AND then keeps the maximum between those two
            alt = np.max(alt.reshape(alt.shape[0],-1,2), axis=2)
        
        outRef[i1:i2, :motif.nmotifs] = ref 
        outAlt[i1:i2, :motif.nmotifs] = alt

    print("calculating diff...")
    if diff_score == "FABIAN":
        outRef = 1 - outRef/kernel_norms[np.newaxis,:]
        outAlt = 1 - outAlt/kernel_norms[np.newaxis,:]
        mask = outRef > outAlt
        f = np.empty_like(outRef)
        f[np.where(mask)] = -(1-outAlt[mask]+alpha)/(1-outRef[mask]+alpha)+1
        mask = outAlt >= outRef
        f[np.where(mask)] = (1-outRef[mask]+alpha)/(1-outAlt[mask]+alpha)-1
        outDiff = 2/(1+np.power(2, -2*f)) - 1
        
    
    if diff_score == "probNorm" or diff_score == "NONE":
        #outAlt[outAlt == 0] = 1e-7
        #outDiff = outRef/outAlt
        outDiff = outAlt-outRef
    
    
    #print(outDiff[:,0])
    #print(motif.names[0])
    
    print(f"Writing the results to {out_file}")
    #motif_names = []
    motif_names = motif.names

    write_output_diff(out_file+".alt", outAlt, motif_names, segments.names())
    write_output_diff(out_file+".ref", outRef, motif_names, segments.names())
    write_output_diff(out_file+".diff", outDiff, motif_names, segments.names())

if __name__ == "__main__":
    app()
