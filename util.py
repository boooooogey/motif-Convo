import numpy as np
import torch
import pandas as pd
from pysam import FastaFile
import time

def number_of_headers(filename):
    header=0
    with open(filename,"r") as file:      
        while True:
            line = file.readline()      
            if line.startswith("#"):
                header=header+1
            else:
                break
    return header

def readvcf(filename):
    nh = number_of_headers(filename)
    if nh > 1:
        data = pd.read_csv(filename, header=list(range(nh)), sep="\t")
        data.columns = pd.MultiIndex.from_tuples([tuple(i[1:] for i in data.columns[0])] +list(data.columns)[1:])
    elif nh == 1:
        data = pd.read_csv(filename, header=0, sep="\t")
        data.columns = [data.columns[0][1:]] + data.columns.to_list()[1:]
    else:
        data = pd.read_csv(filename, header=None, sep="\t")
    return data  

def readbed(filename, up):
    data = pd.read_csv(filename, sep = "\t", header = None)
    chrs = data[0].to_numpy()
    start = data[1].to_numpy(dtype=int)
    end = data[2].to_numpy(dtype=int)
    if(data.shape[1] > 5): #get the strand
        print("Strand detected")
        up = int(np.floor(up))
        strand = data[5].to_numpy()
        #adjust the regions to acccount for strand and up
        start = start - (strand == "+") * up #[start[i]-up if strand[i]=="+" else start[i] for i in range(len(start))]
        end = end + (strand == "-") * up #[end[i]+up if strand[i]=="-" else end[i] for i in range(len(start))]
    return chrs, start, end

def returnonehot(string):
    string = string.upper()
    lookup = {'A':0, 'C':1, 'G':2, 'T':3}
    tmp = np.array(list(string))
    icol = np.where(tmp != 'N')[0]
    out = np.zeros((4,len(tmp)), dtype = np.float32)
    irow = np.array([lookup[i] for i in tmp[icol]])

    if len(icol)>0:
        out[irow,icol] = 1

    return out

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
        in_channels = 4
        lens = np.array([len(i.split('\n')[2:]) for i in data[4:]])
        height = np.max(lens)
        maximumpadding = height - np.min(lens)
        out = np.zeros((out_channels, in_channels, height), dtype=np.float32)
        mask = torch.zeros((out_channels, 1, height), dtype=torch.uint8)
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
            if transform == "constant":
                bg=np.repeat(0.25,4).reshape(1,4)
            if transform == "local":
                bg=np.average(kernel,0).reshape(1,4)
            if transform != "none":
                offset=np.min(kernel[kernel>0])
                bgMat=np.tile(bg,(kernel.shape[0],1))
                kernel=np.log((kernel+offset)/bgMat)
            out[2*k  , :, :kernel.shape[0]] = kernel.T
            out[2*k+1, :, :kernel.shape[0]] = kernel[::-1, ::-1].T
            mask[2*k  , :, :kernel.shape[0]] = 1
            mask[2*k+1, :, :kernel.shape[0]] = 1
        return torch.from_numpy(out), mask

class vcfData:
    def __init__(self, vcf, batchsize, genome, windowsize):
        data = readvcf(vcf)
        self.headers = data.columns.to_list()
        
        self.chrs = data.iloc[:,0].to_numpy()
        self.starts = data.iloc[:,1].to_numpy() - int(windowsize)
        self.ends = data.iloc[:,1].to_numpy() + int(windowsize) - 1
        self.ref = data.iloc[:,3].to_numpy()
        self.alt = data.iloc[:,4].to_numpy()

        self.batchsize = batchsize
        self.n = data.shape[0] 
        self.seqs = FastaFile(genome)
        self.windowsize = windowsize
        refs = self.seqs.references
        lengths = self.seqs.lengths
        self.limits = {refs[i]: lengths[i] for i in range(len(refs))}
        self.out = open("coordinatesUsed.bed", "w")
        self.lookup = {'A':0, 'C':1, 'G':2, 'T':3}
        
    def __len__(self):
        return int(np.ceil(self.n / self.batchsize))

    def __getitem__(self, i):
        i1, i2 = i*self.batchsize, (i+1)*self.batchsize
        if i2 >= self.n: i2 = self.n
        batchsize = int(i2 - i1)
        height = self.windowsize*2-1 #np.max(self.ends[i1:i2] - self.starts[i1:i2])# + self.padding
        width = 4
        batch = np.zeros((batchsize, width, height), dtype=np.float32) 
        altbatch = np.zeros((batchsize, width, height), dtype=np.float32) 
        stats = np.empty((batchsize, 4))
        for i, c, s, e, r, a in zip(range(i2-i1), self.chrs[i1:i2], self.starts[i1:i2], self.ends[i1:i2], self.ref[i1:i2], self.alt[i1:i2]):
            if s>0 and e<self.limits[c]:
                seg = self.seqs.fetch(c, s, e)
                seg=seg.upper()
                #print(f"Sequence: {seg[:self.windowsize-1]} {seg[self.windowsize-1]} {seg[self.windowsize:]}")
                #print(f"a: ({a}, {self.lookup[a]}), r: ({r}, {self.lookup[r]}), Target: {seg[self.windowsize-1]}")
                assert(seg[self.windowsize-1]==r or len(a)!=1 or len(r)!=1)
                batch[i, :, :height] = returnonehot(seg)
                altbatch[i, :, :height] = batch[i, :, :] 
                altbatch[i, self.lookup[r], self.windowsize-1] = 0
                altbatch[i, self.lookup[a], self.windowsize-1] = 1
        return torch.from_numpy(batch), torch.from_numpy(altbatch) #torch.from_numpy(batch)

    def return_mask(self):
        a = torch.zeros((1,self.windowsize*2-1), dtype=torch.uint8)
        a[:, self.windowsize-1] = 1
        return a

    def __del__(self):
        self.out.close()

def countlowercase(arr):
    return sum([1 for c in arr if c.islower()])

def stringstats(string):
    lowercaseratio = countlowercase(string)/len(string)
    string = string.upper()
    tmp = np.array(list(string))
    gccount = np.sum(np.logical_or(tmp == 'C', tmp == 'G'))/len(tmp)
    gcpattern = string.count("GC")/(len(tmp)-1)
    cgpattern = string.count("CG")/(len(tmp)-1)
    return np.array([gccount, gcpattern, cgpattern, lowercaseratio])

class SegmentData:
    def __init__(self, bed, batchsize, genome, windowsize, up):
        self.chrs, self.starts, self.ends = readbed(bed, up)
        self.midpoints = np.asarray(np.ceil((self.starts + self.ends)/2),dtype=int)
        self.starts = self.midpoints - windowsize
        self.ends = self.midpoints + windowsize
        self.batchsize = batchsize
        self.n = len(self.chrs)
        self.seqs = FastaFile(genome)
        self.padding = windowsize
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
        height = np.max(self.ends[i1:i2] - self.starts[i1:i2]) + self.padding
        width = 4
        batch = np.zeros((batchsize, width, height), dtype=np.float32) 
        stats = np.empty((batchsize, 4), dtype=np.float32)
        for i, c, s, e in zip(range(i2-i1), self.chrs[i1:i2], self.starts[i1:i2], self.ends[i1:i2]):
            self.out.write(c+"\t"+str(s)+"\t"+str(e)+"\n")
            if s>0 and e<self.limits[c]:
                seg = self.seqs.fetch(c, s, e)
            else:
                seg = "N"*(self.padding*2)
            stats[i] = stringstats(seg)
            batch[i, :, :(e-s)] = returnonehot(seg)
        return torch.from_numpy(batch), stats

    def __del__(self):
        self.out.close()

if __name__ == "__main__":
    motif = MEME()
    kernels = motif.parse("TestInput/Test.meme", "none")
    segments = vcfData("TestInput/TestHg38.vcf", 128, "/data/genomes/human/Homo_sapiens/UCSC/hg38/Sequence/WholeGenomeFasta/genome.fa", kernels.shape[2])
    print(segments.headers)
    start = time.time()
    for i in range(len(segments)):
        orig, alt = segments[i]
    end = time.time()
    print(f"other took {end-start}")

