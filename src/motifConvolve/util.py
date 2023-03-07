import numpy as np
import torch
import pandas as pd
from pysam import FastaFile
import time
import itertools
import xml.etree.ElementTree as ET
import os
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import regex as re

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

def kmers_count(seq, k=2):
    lookup = {"".join(i):0 for i in itertools.product(["A","C","G","T"], repeat=k)}
    mers = [seq[i:i+2] for i in range(len(seq)-k+1)]
    for i in mers:
        if i in lookup:
            lookup[i] += 1
    for i in lookup:
        lookup[i] /= (len(seq)-k+1)
    return list(lookup.values())

def kmers(k=2):
    return ["".join(i) for i in itertools.product(["A","C","G","T"], repeat=k)]

def logit(x, a, b):
    return 1/(1 + np.exp(-a * x - b))

def logit_torch(x, a, b):
    return 1/(1 + torch.exp(-a * x - b))

def init_dist(dmin, dmax, dp, weights, probs):
    out = np.zeros(int(np.round((dmax-dmin)/dp)+1))
    ii = np.array(np.round((weights-dmin)/dp), dtype=int)
    for i in range(len(probs)):
        out[ii[i]] = out[ii[i]] + probs[i]
    return out

def scoreDist(pwm, nucleotide_prob=None, gran=None, size=1000):
    if nucleotide_prob is None:
        nucleotide_prob = np.ones(4)/4
    if gran is None:
        if size is None:
            raise ValueError("provide either gran or size. Both missing.")
        gran = (np.max(pwm) - np.min(pwm))/(size - 1)
    pwm = np.round(pwm/gran)*gran
    pwm_max, pwm_min = pwm.max(axis=1), pwm.min(axis=1)
    distribution = init_dist(pwm_min[0], pwm_max[0], gran, pwm[0], nucleotide_prob[0])
    for i in range(1, pwm.shape[0]):
        kernel = init_dist(pwm_min[i], pwm_max[i], gran, pwm[i], nucleotide_prob[i])
        distribution = np.convolve(distribution, kernel)
    support_min = pwm_min.sum()
    ii = np.where(distribution > 0)[0]
    support = support_min + (ii) * gran
    return support, distribution[ii]

def return_coef_for_normalization(pwms, nucleotide_prob=None, gran=None, size=1000):
    params = []
    for i in range(0,pwms.shape[0],2):
        pwm = pwms[i].numpy().T
        pwm = pwm[pwm.sum(axis=1) != 0, :]
        #prob = np.exp(pwm).sum(axis=0)/np.exp(pwm).sum()
        prob = np.exp(pwm)
        s, d = scoreDist(pwm, prob, gran, size)
        param, _ = curve_fit(logit, np.exp(s), np.cumsum(d), maxfev=5000)
        #f = interp1d(np.exp(s), np.cumsum(d))
        #print(curve_fit(logit, np.exp(s), np.cumsum(d), maxfev=5000))
        #params.append(param)
        params.append(param)
    return params

def return_coef_for_normalization_diff(pwms, nucleotide_prob=None, gran=None, size=1000, length_correction=1):
    params = []
    for i in range(0,pwms.shape[0],2):
        pwm = pwms[i].numpy().T
        pwm = pwm[pwm.sum(axis=1) != 0, :]
        #prob = pwm.sum(axis=0)/pwm.sum()
        prob = np.sum(np.exp(pwm) / np.exp(pwm).sum(axis=1).reshape(-1,1), axis=0)/np.sum(np.exp(pwm) / np.exp(pwm).sum(axis=1).reshape(-1,1))
        s, d = scoreDist(pwm, prob, gran, size, diff=True)
        param, _ = curve_fit(logit, s, np.power(np.cumsum(d), length_correction))
        params.append(param)
    return params

def normalize_mat(mat, params):
    out = torch.empty_like(mat)
    assert mat.shape[1] == len(params)
    for i in range(len(params)):
        #out[:,i] = logit(mat[:,i], *params[i])
        #tmp = np.clip(mat[:,i],params[i].x.min(), params[i].x.max())
        #tmp = params[i](tmp)
        out[:,i] = logit_torch(mat[:,i], *params[i])
    return out

#def readvcf(filename):
#    nh = number_of_headers(filename)
#    if nh > 1:
#        data = pd.read_csv(filename, header=list(range(nh)), sep="\t")
#        data.columns = pd.MultiIndex.from_tuples([tuple(i[1:] for i in data.columns[0])] +list(data.columns)[1:])
#    elif nh == 1:
#        data = pd.read_csv(filename, header=0, sep="\t")
#        data.columns = [data.columns[0][1:]] + data.columns.to_list()[1:]
#    else:
#        data = pd.read_csv(filename, header=None, sep="\t")
#    return data  

def readvcf(filename):
    nh = number_of_headers(filename)
    if nh > 1:
        data = pd.read_csv(filename, skiprows=nh, header=None, sep="\t")
        #data.columns = pd.MultiIndex.from_tuples([tuple(i[1:] for i in data.columns[0])] +list(data.columns)[1:])
    elif nh == 1:
        data = pd.read_csv(filename, skiprows=1, header=None, sep="\t")
        #data.columns = [data.columns[0][1:]] + data.columns.to_list()[1:]
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

def returnonehot(string, dinucleotide=False):
    string = string.upper()
    tmp = np.array(list(string))

    if dinucleotide:
        lookup = {"".join(i):n for n,i in enumerate(itertools.product(["A","C","G","T"], repeat=2))}
        icol = np.where(tmp == 'N')[0]
        #icol = np.unique(icol // 2)
        #icol = np.where(np.logical_not(np.isin(np.arange(len(tmp)//2), icol)))[0]
        icol = np.unique(np.clip(np.concatenate([icol, icol-1]), 0, len(tmp)-2))
        icol = np.where(np.logical_not(np.isin(np.arange(len(tmp)-1), icol)))[0]
        tmp = np.array([tmp[i] + tmp[i+1] for i in range(len(tmp)-1)])
        irow = np.array([lookup[i] for i in tmp[icol]])
    else:
        lookup = {'A':0, 'C':1, 'G':2, 'T':3}
        icol = np.where(tmp != 'N')[0]
        irow = np.array([lookup[i] for i in tmp[icol]])

    out = np.zeros((len(lookup),len(tmp)), dtype = np.float32)

    if len(icol)>0:
        out[irow,icol] = 1

    return out

def read_TFFM(file):
    tree = ET.parse(file)
    root = tree.getroot()
    data = []
    for state in root[0].iterfind("state"):
        discrete = state[0]
        if "order" in discrete.attrib:
            data.append(discrete.text.split(","))
    return np.array(data, dtype=float)

def transform_kernel(kernel, smoothing, background):
    out = np.log(kernel / background + smoothing)
    c = out.max(axis=1)
    out = out - c[:, np.newaxis]
    norm = out.min(axis=1).sum()
    return out, norm

class MEME():
    def __init__(self, precision=1e-7, smoothing=0.02, background=None):
        self.version = 0
        self.alphabet = ""
        self.strands = ""
        #self.headers = []
        self.background = []
        self.names = []
        self.nmotifs = 0
        self.precision=1e-7
        self.smoothing = smoothing
        if background is None:
            self.background = np.ones(4)*0.25
        else:
            self.background = background

    def parse(self, text):
        precision = self.precision
        with open(text,'r') as file:
            data = file.read()
        self.version = re.compile(r'MEME version ([\d+\.*]+)').match(data).group(1)
        self.names = re.findall(r"MOTIF (.*)\n", data)
        self.background = re.findall(r"Background letter frequencies.*\n(A .* C .* G .* T .*)\n", data)[0]
        self.strands = re.findall(r"strands: (.*)\n", data)[0].strip()
        self.alphabet = re.findall(r"ALPHABET=(.*)\n", data)[0].strip()
        letter_probs = re.findall(r"(letter-probability.*\n([ \t]*\d+\.?\d*[ \t]+\d+\.?\d*[ \t]+\d+\.?\d*[ \t]+\d+\.?\d*[ \t]*\n)+)", data)
        assert len(letter_probs) == len(self.names)
        self.nmotifs = len(letter_probs)
        out_channels = self.nmotifs * 2
        in_channels = 4
        matrices = []
        length = 0
        for i in range(len(letter_probs)):
            matrix = letter_probs[i][0].split("\n")
            if len(matrix[-1]) == 0:
                matrix = matrix[1:-1]
            else:
                matrix = matrix[1:]
            matrices.append(np.array([i.split() for i in matrix], dtype=float))
            if matrices[-1].shape[0] > length:
                length = matrices[-1].shape[0]
        out = np.zeros((out_channels, in_channels, length), dtype=np.float32)
        mask = torch.zeros((out_channels, 1, length), dtype=torch.uint8)
        for k, kernel in enumerate(matrices):
            #if transform == "constant":
            #    bg=np.repeat(0.25, in_channels).reshape(1,4)
            #if transform == "local":
            #    bg=np.average(kernel,0).reshape(1,4)
            #if transform != "none":
            #    offset=np.min(kernel[kernel>0])
            #    bgMat=np.tile(bg,(kernel.shape[0],1))
            #    kernel=np.log((kernel+offset)/bgMat)
            kernel[kernel == 0] = self.precision
            kernel = np.log(kernel)
            out[2*k  , :, :kernel.shape[0]] = kernel.T
            out[2*k+1, :, :kernel.shape[0]] = kernel[::-1, ::-1].T
            mask[2*k  , :, :kernel.shape[0]] = 1
            mask[2*k+1, :, :kernel.shape[0]] = 1
        return torch.from_numpy(out), mask

class MEME_with_Transformation():
    def __init__(self, precision=1e-7, smoothing=0.02, background=None):
        self.version = 0
        self.alphabet = ""
        self.strands = ""
        #self.headers = []
        self.background = []
        self.names = []
        self.nmotifs = 0
        self.precision=1e-7
        self.smoothing = smoothing
        if background is None:
            self.background_prob = np.ones(4)*0.25
        else:
            self.background_prob = background

    def parse(self, text):
        precision = self.precision
        with open(text,'r') as file:
            data = file.read()
        self.version = re.compile(r'MEME version ([\d+\.*]+)').match(data).group(1)
        self.names = re.findall(r"MOTIF (.*)\n", data)
        self.background = re.findall(r"Background letter frequencies.*\n(A .* C .* G .* T .*)\n", data)[0]
        self.strands = re.findall(r"strands: (.*)\n", data)[0].strip()
        self.alphabet = re.findall(r"ALPHABET=(.*)\n", data)[0].strip()
        letter_probs = re.findall(r"(letter-probability.*\n([ \t]*\d+\.?\d*[ \t]+\d+\.?\d*[ \t]+\d+\.?\d*[ \t]+\d+\.?\d*[ \t]*\n)+)", data)
        assert len(letter_probs) == len(self.names)
        self.nmotifs = len(letter_probs)
        out_channels = self.nmotifs * 2
        in_channels = 4
        matrices = []
        length = 0
        for i in range(len(letter_probs)):
            matrix = letter_probs[i][0].split("\n")
            if len(matrix[-1]) == 0:
                matrix = matrix[1:-1]
            else:
                matrix = matrix[1:]
            matrices.append(np.array([i.split() for i in matrix], dtype=float))
            if matrices[-1].shape[0] > length:
                length = matrices[-1].shape[0]
        out = np.zeros((out_channels, in_channels, length), dtype=np.float32)
        mask = torch.zeros((out_channels, 1, length), dtype=torch.uint8)
        motif_norms = np.zeros(self.nmotifs, dtype=np.float32)
        for k, kernel in enumerate(matrices):
            kernel, motif_norms[k] = transform_kernel(kernel, self.smoothing, self.background_prob)
            out[2*k  , :, :kernel.shape[0]] = kernel.T
            out[2*k+1, :, :kernel.shape[0]] = kernel[::-1, ::-1].T
            mask[2*k  , :, :kernel.shape[0]] = 1
            mask[2*k+1, :, :kernel.shape[0]] = 1
        return torch.from_numpy(out), mask, motif_norms

class TFFM():
    def __init__(self):
        self.names = []
        self.nmotifs = 0

    def parse(self, directory):
        self.names = os.listdir(directory)
        self.nmotifs = len(self.names)
        in_channels = 16
        out_channels = self.nmotifs
        data = []
        height = 0
        for i in self.names:
            tffm = read_TFFM(os.path.join(directory, i))
            data.append(tffm)
            if tffm.shape[0] > height:
                height = tffm.shape[0]
        out = np.zeros((out_channels, in_channels, height), dtype=np.float32)
        mask = torch.zeros((out_channels, 1 , height), dtype=torch.uint8)
        for n, tffm in enumerate(data):
            out[n, :, :tffm.shape[0]] = tffm.T
            mask[n, :, :tffm.shape[0]] = 1
        return torch.from_numpy(out), mask

class vcfData:
    def __init__(self, vcf, batchsize, genome, windowsize):
        data = readvcf(vcf)
        self.headers = data.columns.to_list()
        
        self.ref = data.iloc[:,3].to_numpy()
        self.alt = data.iloc[:,4].to_numpy()

        f = np.vectorize(len)

        self.reflength = f(self.ref)
        self.altlength = f(self.alt)

        self.chrs = data.iloc[:,0].to_numpy()

        self.refstarts = data.iloc[:,1].to_numpy() - int(windowsize)
        self.refends = data.iloc[:,1].to_numpy() + self.reflength - 1 + int(windowsize) - 1

        self.altstarts = data.iloc[:,1].to_numpy() - int(windowsize)
        self.altends = data.iloc[:,1].to_numpy() + self.altlength - 1 + int(windowsize) - 1

        self.pos = data.iloc[:,1].to_numpy()

        self.variant_names = data.iloc[:, 2].to_numpy()

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

    def names(self):
        return self.variant_names

    def __getitem__(self, i):
        i1, i2 = i*self.batchsize, (i+1)*self.batchsize
        if i2 >= self.n: i2 = self.n
        batchsize = int(i2 - i1)
        targetlength = max(np.max(self.reflength[i1:i2]), np.max(self.altlength[i1:i2]))
        height = (self.windowsize-1)*2 + targetlength #np.max(self.ends[i1:i2] - self.starts[i1:i2])# + self.padding
        width = 4
        batch = np.zeros((batchsize, width, height), dtype=np.float32) 
        mask = torch.zeros((batchsize, 1, height), dtype=torch.uint8)
        altbatch = np.zeros((batchsize, width, height), dtype=np.float32) 
        altmask = torch.zeros((batchsize, 1, height), dtype=torch.uint8)
        stats = np.empty((batchsize, 4))
        for i, c, refs, refe, alts, alte, r, a, lenr, lena in zip(range(i2-i1), self.chrs[i1:i2], self.refstarts[i1:i2], self.refends[i1:i2], self.altstarts[i1:i2], self.altends[i1:i2], self.ref[i1:i2], self.alt[i1:i2], self.reflength[i1:i2], self.altlength[i1:i2]):
            if refs>0 and refe<self.limits[c]:
                seg = self.seqs.fetch(c, refs, refe)
                seg=seg.upper()
                #print(f"Sequence: {seg[:self.windowsize-1]} {seg[self.windowsize-1]} {seg[self.windowsize:]}")
                #print(f"a: ({a}, {self.lookup[a]}), r: ({r}, {self.lookup[r]}), Target: {seg[self.windowsize-1]}")
                #assert(seg[self.windowsize-1]==r or len(a)!=1 or len(r)!=1)
                assert(seg[self.windowsize-1:-(self.windowsize-1)]==r)
                batch[i, :, :(refe-refs)] = returnonehot(seg)
                mask[i, :, (self.windowsize-1):(refe-refs-self.windowsize+1)] = 1
                altbatch[i, :, :(alte-alts)] = returnonehot(seg[:self.windowsize-1] + a + seg[-(self.windowsize-1):])
                altmask[i, :, (self.windowsize-1):(alte-alts-self.windowsize+1)] = 1
        return torch.from_numpy(batch), mask, torch.from_numpy(altbatch), altmask #torch.from_numpy(batch)

def countlowercase(arr):
    return sum([1 for c in arr if c.islower()])

def stringstats(string):
    lowercaseratio = countlowercase(string)/len(string)
    string = string.upper()
    tmp = np.array(list(string))
    gccount = np.sum(np.logical_or(tmp == 'C', tmp == 'G'))/len(tmp)
    #gcpattern = string.count("GC")/(len(tmp)-1)
    #cgpattern = string.count("CG")/(len(tmp)-1)
    patterns = kmers_count(string)
    return np.array([gccount, lowercaseratio, *patterns], dtype=np.float32)

class SegmentData:
    def __init__(self, bed, batchsize, genome, windowsize, up, dinucleotide=False):
        self.chrs, self.starts, self.ends = readbed(bed, up)
        self.id = ["_".join([c, str(s), str(e)]) for c, s, e in zip(self.chrs, self.starts, self.ends)]
        self.midpoints = np.asarray(np.ceil((self.starts + self.ends)/2),dtype=int)
        self.starts = self.midpoints - windowsize
        self.ends = self.midpoints + windowsize
        self.batchsize = batchsize
        self.n = len(self.chrs)
        self.seqs = FastaFile(genome)
        self.padding = windowsize
        refs = self.seqs.references
        lengths = self.seqs.lengths
        self.additional = 4 * 4 + 2
        self.limits = {refs[i]: lengths[i] for i in range(len(refs))}
        self.out = open("coordinatesUsed.bed", "w")
        self.dinucleotide = dinucleotide

    def names(self):
        return self.id

    def __len__(self):
        return int(np.ceil(self.n / self.batchsize))

    def __getitem__(self, i):
        i1, i2 = i*self.batchsize, (i+1)*self.batchsize
        if i2 >= self.n: i2 = self.n
        batchsize = int(i2 - i1)
        if self.dinucleotide:
            height = np.max(self.ends[i1:i2] - self.starts[i1:i2])-1# + self.padding
            width = 16
        else:
            height = np.max(self.ends[i1:i2] - self.starts[i1:i2])# + self.padding
            width = 4
        batch = np.zeros((batchsize, width, height), dtype=np.float32) 
        stats = np.empty((batchsize, self.additional), dtype=np.float32)
        for i, c, s, e in zip(range(i2-i1), self.chrs[i1:i2], self.starts[i1:i2], self.ends[i1:i2]):
            self.out.write(c+"\t"+str(s)+"\t"+str(e)+"\n")
            if s>0 and e<self.limits[c]:
                seg = self.seqs.fetch(c, s, e)
            else:
                seg = "N"*(self.padding*2)
            stats[i] = stringstats(seg)
            if self.dinucleotide:
                batch[i, :, :(e-s)-1] = returnonehot(seg, dinucleotide=True)
            else:
                batch[i, :, :(e-s)] = returnonehot(seg)
        return torch.from_numpy(batch), stats

    def __del__(self):
        pass
        #self.out.close()

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

