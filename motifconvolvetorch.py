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

def readbed(filename, up):
    with open(filename,"r") as file:
        data =file.readlines()
    data = [i.split('\t') for i in data]
    chrs = [i[0] for i in data]
    start = [int(i[1]) for i in data]
    end = [int(i[2]) for i in data]
    if(len(data[0])>4): #get the strand
        print("Strand detected")
        up=int(np.floor(up))
        strand=[i[4].strip() for i in data]
        #adjust the regions to acccount for strand and up
        start=[start[i]-up if strand[i]=="+" else start[i] for i in range(len(start))]
        end=[end[i]+up if strand[i]=="-" else end[i] for i in range(len(start))]
#        print(strand)
#        print(start)
#        print(end)
    return np.array(chrs), np.array(start, dtype = int), np.array(end, dtype = int)

def stringstats(string):
    string=string.upper()
    tmp = np.array(list(string))
    gc_count = np.sum(np.logical_or(tmp == 'C', tmp == 'G'))/len(tmp)
    gc_pattern = string.count("GC")/(len(tmp)-1)
    cg_pattern = string.count("CG")/(len(tmp)-1)
    return np.array([gc_count, gc_pattern, cg_pattern])

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

class SegmentData:
    def __init__(self, bed, batchsize, genome, windowsize, up):
        self.chrs, self.starts, self.ends = readbed(bed, up)
        self.midpoints = np.asarray(np.ceil((self.starts + self.ends)/2),dtype=int)
#        print(self.midpoints)
        self.starts = self.midpoints - windowsize
        self.ends = self.midpoints + windowsize
        self.batchsize = batchsize
        self.n = len(self.chrs)
        self.seqs = FastaFile(genome)
        self.padding = windowsize
        refs=self.seqs.references
        lengths=self.seqs.lengths
        self.limits= {refs[i]: lengths[i] for i in range(len(refs))}
        self.out=open("coordinatesUsed.bed", "w")
    def __len__(self):
        return int(np.ceil(self.n / self.batchsize))

    def __getitem__(self, i):
        i1, i2 = i*self.batchsize, (i+1)*self.batchsize
        if i2 >= self.n: i2 = self.n
        batchsize = int(i2 - i1)
        nchannel = 1
        height = np.max(self.ends[i1:i2] - self.starts[i1:i2]) + self.padding
        width = 4
        batch = np.zeros((batchsize, nchannel, height, width)) 
        #batch[:] = np.NaN
        stats = np.empty((batchsize, 3))
        for i, c, s, e in zip(range(i2-i1), self.chrs[i1:i2], self.starts[i1:i2], self.ends[i1:i2]):
            self.out.write(c+"\t"+str(s)+"\t"+str(e)+"\n")
            if s>0 and e<self.limits[c]:
                seg = self.seqs.fetch(c, s, e)
            else:
                seg="N"*(self.padding*2)
            stats[i] = stringstats(seg)
            batch[i, :, :(e-s), :] = returnonehot(seg)
        return torch.from_numpy(batch), stats

def write_output(filename, mat, names):
    with open(filename, "w") as file:
        for i in range(len(names)):
            file.write(names[i])
            file.write("\t")
        file.write("GC_ratio\t")
        file.write("GC_pattern\t")
        file.write("CG_pattern\n")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                file.write("{:.4f}".format(mat[i,j]))
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
    parser.add_argument('-b', '--bed', required = True, help="bed file for the peaks")
    parser.add_argument('--batch', default=128, type=int, help="batch size")
    parser.add_argument('-o','--out', required = True, help="output directory")
    parser.add_argument('-w','--window', default=240, type=int, help="window size")
    parser.add_argument('-u','--up', default=0, type=int, help="add upstream")
    parser.add_argument('-t','--transform', default="none", help="transform probabilities to scoring matrix")
    args = parser.parse_args()

    motif = MEME()
    print(f"Reading the motifs from {args.meme}")
    kernels = motif.parse(args.meme, args.transform)
    print(f"Reading peaks from {args.bed}")
    print(f"Genome: {args.genome}")
    print(f"Total window: {args.window+args.up}")

    segments = SegmentData(args.bed, args.batch, args.genome, int(args.window+args.up/2), args.up)
    out = np.empty((segments.n, motif.nmotifs+3))
    print(f"Batch size: {args.batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*args.batch, (i+1)*args.batch
        if i2 >= segments.n: i2 = segments.n
        mat, out[i1:i2, motif.nmotifs:] = segments[i]
        tmp = F.conv2d(mat, kernels)
        tmp = F.max_pool2d(tmp, (tmp.shape[2], 1)).numpy()
        out[i1:i2, :motif.nmotifs] = np.max(tmp.reshape(tmp.shape[0],-1,2), axis=2)
    print(f"Writing the results to {args.out}")
    write_output(args.out, out, motif.names)

main()
