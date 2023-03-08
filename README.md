# motif-Convo
Extracts features using motif convolution
## Installation
```
python -m pip install git+https://github.com/boooooogey/motif-Convo.git
```
## How to use
To run it:
```
motifConvolve tfextract --meme meme_file --bed bed_file --genome genome.fa --mode max/average --out matrix_out_file
```
- ```meme_file```  is a MEME motif file.
- ```bed_file```  is a bed file that contains ATAC peaks. 
- ```genome.fa``` is the FASTA file for the genome. 
- After it calculates convolutions of motifs and peaks, it pools them either with ```max``` or ```average```. You can specify this with ```--mode```. The default is ```max```.
- Once the execution is finished, it writes the resulting matrix to matrix_out_file . Rows of the matrix correspond to peaks, and columns correspond to motifs. There are 4 additional columns: ```GC_ratio```, ```GC_pattern```, ```CG_pattern```, and ```Masked_ratio```. ```GC_ratio``` is the ratio of the total number of G and C to the length of a peak. ```GC_pattern``` is the ratio of the GC pattern and ```CG_pattern``` is the ratio of the CG pattern within a peak. ```Masked_ratio``` is the ratio of the repressed sites to the length.
-  You can change the batch size with ```--batch```. It may affect the execution time. The default is ```128```. 
-  There is ```--window``` flag. Using it, you can change the window size around peaks. The default is ```240```.
