#!/usr/bin/env python
import argparse
import numpy as np
import os
from pysam import FastaFile
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from IPython import embed

torch.backends.cudnn.deterministic = True


def readvcf(filename):
    with open(filename,"r") as file:      
      data = file.readlines()      
      #TODO check for headers
    header=0

    while (data[header][0]=="#"):
        header=header+1
    print(str(header)+" header lines")
    data = [i.strip().split('\t') for i in data[0:]]
    chrs = [i[0] for i in data[header:]]
    start = [int(i[1]) for i in data[header:]]
    ref = [i[3] for i in data[header:]]
    alt = [i[4] for i in data[header:]]

    return np.array(chrs), np.array(start, dtype = int), np.array(ref), np.array(alt)


def returnonehot(string):
    string=string.upper()
    lookup = {'A':0, 'C':1, 'G':2, 'T':3}
    tmp = np.array(list(string))
    irow = np.where(tmp != 'N')[0]
    out = np.zeros((len(tmp),4))
    icol = np.array([lookup[i] for i in tmp[irow]])
    if len(icol)>0:
        out[irow,icol] = 1
    return np.asarray(out)

class MEME():
    def __init__(self):
        self.version = 0
        self.alphabet = ""
        self.strands = ""
        self.headers = []
        self.background = []
        self.names = []
        self.nmotifs = 0

    def parse(self, text, transform):
        with open(text,'r') as file:
            data = file.read()
        data = data.split("\n\n")
        data = data[:-1]
        out_channels = (len(data) - 4) * 2
        in_channels = 1
        lens = np.array([len(i.split('\n')[2:]) for i in data[4:]])
        height = np.max(lens)
        maximumpadding = height - np.min(lens)
        width = 4
        out = np.zeros((out_channels, in_channels, height, width))
        self.nmotifs = len(data) - 4
        self.version = int(data[0].split(' ')[-1])
        self.alphabet = data[1][10:].strip()
        self.strands = data[2][9:].strip()
        self.background = np.array(data[3].split('\n')[1].split(' ')[1::2],dtype=float)
        data = data[4:]
        for k, i in enumerate(data):
            tmp = i.split('\n')
            self.names.append(tmp[0].split()[-1])
            self.headers.append('\n'.join(tmp[:2]))
            kernel = np.array([j.split() for j in tmp[2:]],dtype=float)
            if (transform == "constant"):
                bg=np.repeat(0.25,4).reshape(1,4)
            if (transform == "local"):
                bg=np.average(kernel,0).reshape(1,4)
            if (transform != "none"):
                offset=np.min(kernel[kernel>0])
                bgMat=np.tile(bg,(kernel.shape[0],1))
                kernel=np.log((kernel+offset)/bgMat)
            out[2*k  , 0, :kernel.shape[0], :] = kernel
            out[2*k+1, 0, :kernel.shape[0], :] = kernel[::-1, ::-1]
        return torch.from_numpy(out)


class vcfDataREF:
    def __init__(self, vcf, batchsize, genome, windowsize):
        self.chrs, self.pos, self.ref, self.alt = readvcf(vcf)
        self.starts = self.pos - int(windowsize)
        self.ends = self.pos + int(windowsize)-1
        self.batchsize = batchsize
        self.n = len(self.chrs)
        self.seqs = FastaFile(genome)
        self.windowsize = windowsize
        refs = self.seqs.references
        lengths = self.seqs.lengths
        self.limits = {refs[i]: lengths[i] for i in range(len(refs))}
        self.out = open("coordinatesUsed.bed", "w")

        
    def __len__(self):
        return int(np.ceil(self.n / self.batchsize))

    def __getitem__(self, i):
        i1, i2 = i*self.batchsize, (i+1)*self.batchsize
        if i2 >= self.n: i2 = self.n
        batchsize = int(i2 - i1)
        height = self.windowsize*2-1 #np.max(self.ends[i1:i2] - self.starts[i1:i2])# + self.padding
        width = 4
        batch = np.zeros((batchsize, 1, height, width)) 
        stats = np.empty((batchsize, 4))
        for i, c, s, e, r, a in zip(range(i2-i1), self.chrs[i1:i2], self.starts[i1:i2], self.ends[i1:i2], self.ref[i1:i2], self.alt[i1:i2]):
            if s>0 and e<self.limits[c]:
                seg = self.seqs.fetch(c, s, e)
                seg=seg.upper()
                assert(seg[self.windowsize-1]==r or len(a)!=1 or len(r)!=1)
            else:
                seg = "N"*self.windowsize
#            print(seg)    
            batch[i, :, :height, :] = returnonehot(seg)
        return torch.from_numpy(batch)

    def __del__(self):
        self.out.close()


class vcfDataALT:
    def __init__(self, vcf, batchsize, genome, windowsize):
        self.chrs, self.pos, self.ref, self.alt = readvcf(vcf)
        self.starts = self.pos - int(windowsize)
        self.ends = self.pos + int(windowsize)-1
        self.batchsize = batchsize
        self.n = len(self.chrs)
        self.seqs = FastaFile(genome)
        self.windowsize = windowsize
        refs = self.seqs.references
        lengths = self.seqs.lengths
        self.limits = {refs[i]: lengths[i] for i in range(len(refs))}
        self.out = open("coordinatesUsed.bed", "w")

    def __len__(self):
        return int(np.ceil(self.n / self.batchsize))

    def __getitem__(self, i):
        i1, i2 = i*self.batchsize, (i+1)*self.batchsize
        if i2 >= self.n: i2 = self.n
        batchsize = int(i2 - i1)
        height = self.windowsize*2-1 #np.max(self.ends[i1:i2] - self.starts[i1:i2])# + self.padding
        width = 4
        batch = np.zeros((batchsize, 1, height, width)) 
        stats = np.empty((batchsize, 4))
        for i, c, s, e, r, a in zip(range(i2-i1), self.chrs[i1:i2], self.starts[i1:i2], self.ends[i1:i2], self.ref[i1:i2], self.alt[i1:i2]):
            
            if s>0 and e<self.limits[c]:
                seg = self.seqs.fetch(c, s, e)
                seg=seg.upper()
                if len(r)==1 and len(a)==1:
                    segA = seg[:(self.windowsize-1)] + a + seg[(self.windowsize):]
                else:
                    segA=seg
                if False:
                    print("REF="+r+" ALT="+a)
                    print(seg)
                    print(segA)
                    print("\n")
            
            else:
                seg = "N"*self.windowsize
            batch[i, :, :height, :] = returnonehot(segA)
        return torch.from_numpy(batch)

    def __del__(self):
        self.out.close()

        
        
def write_output(filename, mat, names):
    with open(filename, "w") as file:
        for i in range(len(names)):
            file.write(names[i])
            file.write("\t")
        file.write("\n")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                file.write("{:.8f}".format(mat[i,j]))
                if j != mat.shape[1]-1:
                    file.write("\t")
            file.write("\n")

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
    kernels = motif.parse(args.meme, args.transform)

    print(f"Reading variants from {args.vcf}")
    print(f"Genome: {args.genome}")


    segments = vcfDataREF(args.vcf, args.batch, args.genome,kernels.shape[2])
    outRef = np.empty((segments.n, motif.nmotifs))
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        mat = segments[i]
        tmp = F.conv2d(mat, kernels)
        tmp = F.max_pool2d(tmp, (tmp.shape[2], 1)).numpy()
        outRef[i1:i2, :motif.nmotifs] = np.max(tmp.reshape(tmp.shape[0],-1,2), axis=2)

    segments = vcfDataALT(args.vcf, args.batch, args.genome,kernels.shape[2])
    outAlt = np.empty((segments.n, motif.nmotifs))
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        mat = segments[i]
        tmp = F.conv2d(mat, kernels)
        tmp = F.max_pool2d(tmp, (tmp.shape[2], 1)).numpy()
        outAlt[i1:i2, :motif.nmotifs] = np.max(tmp.reshape(tmp.shape[0],-1,2), axis=2)
        
    print(f"Writing the results to {args.out}")
    write_output(args.out+".alt", outAlt, motif.names)
    write_output(args.out+".ref", outRef, motif.names)
    write_output(args.out+".diff", outAlt-outRef, motif.names)

main()
