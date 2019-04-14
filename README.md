# Introduction

This is MTFN-RR, source code of Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking.  



## Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking 

1. Multi-modal Tensor Fusion Network (MTFN) efficiently encode bilinear interactions between visual and textual global feature representations via tensor-based tucker decomposition on image-text fusion and text-text fusion. It directly learns an accurate image-text similarity function from the two fusion branches and naturally combined with the widely used ranking loss to distinguish between the ground-truth and negative image-text pairs without seeking a common embedding space.

2. Re-ranking (RR) scheme is a simple but effective step to further improve the matching performance, though it has rarely been explored in the community. It is designed to simultaneously refine the initial sentence retrieval and image retrieval results obtained by our MTFN based on their mutual information.

   ![framework](https://github.com/submissionwithsupp/MTFN-RR_Code/blob/master/fig/framework_all__.jpg)




## Proposed Model (MTFN-RR) 


- ### MTFN
  <img src="https://github.com/submissionwithsupp/MTFN-RR_Code/blob/master/fig/framework_all_1.jpg" width="60%" height="60%" />

- ### Re-ranking scheme


  - ### I2T Re-ranking
    
    <img src="https://github.com/submissionwithsupp/MTFN-RR_Code/blob/master/fig/i2t_rerank.jpg" width="50%" height="50%" />



  - ### T2I Re-ranking
    
    <img src="https://github.com/submissionwithsupp/MTFN-RR_Code/blob/master/fig/t2i_rerank.jpg" width="50%" height="50%" />





## Examples
![example](https://github.com/submissionwithsupp/MTFN-RR_Code/blob/master/fig/example.jpg)



## Data Download

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The precomputed image features of MS-COCO are from [here](https://github.com/peteanderson80/bottom-up-attention). The precomputed image features of Flickr30K are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from:

```
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip
```

We refer to the path of extracted files for `data.zip` as `$DATA_PATH` and files for `vocab.zip` to `./vocab` directory. Alternatively, you can also run vocab.py to produce vocabulary files. For example,

```
python vocab.py --data_path data --data_name f30k_precomp
python vocab.py --data_path data --data_name coco_precomp
```



## Usage

- Training stage:
  - The optional parameter setting is in option/MTFN_RR.yaml
  - Run train.py (Note to modify the default settings at the top of python file)
- Testing stage:
  - save the similarity matrix obtained in training stage
  - Run rerank.py (Please note to modify the similarity matrix path)



## Requirements

We recommend the following dependencies：

- Python 3.6
- [PyTorch](http://pytorch.org/) 0.3.1
- [NumPy](http://www.numpy.org/) (>1.12.1)
- [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
- click
- [skipthougths](https://github.com/Cadene/skip-thoughts.torch)



