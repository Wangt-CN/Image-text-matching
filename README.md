# Introduction

This is building code for image-text matching.  



## Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking 

1. Multi-modal Tensor Fusion Network (MTFN) efficiently encode bilinear interactions between visual and textual global feature representations via tensor-based tucker decomposition on image-text fusion and text-text fusion. It directly learns an accurate image-text similarity function from the two fusion branches and naturally combined with the widely used ranking loss to distinguish between the ground-truth and negative image-text pairs without seeking a common embedding space.

2. Re-ranking (RR) scheme is a simple but effective step to further improve the matching performance, though it has rarely been explored in the community. It is designed to simultaneously refine the initial sentence retrieval and image retrieval results obtained by our MTFN based on their mutual information.

   ![framework](https://github.com/Wangt-CN/Image-text-matching/blob/master/fig/framework_new.png)




## Motivation

The existing two main methods for image-text matching: the **embedding-based** and **classification-based**.
<br>

 <img src="https://github.com/Wangt-CN/Image-text-matching/blob/master/fig/intro_example_new2.png" width="50%" />


- Embedding-based : higher accuracy **but**
  - Construct the whole embedding space
  - difficult
  - time consuming
- Classification-based : fast **but**
  - binary classification problem
  - "match" or "mismatch" 
  - low accuracy

So we propose our MTFN-RR to *combine the advantages of these two paradigms, that is balancing retrieval performance and model efficiency in both training and testing stage*.

<br>



## More examples



- ### I2T Re-ranking
    
    <img src="https://github.com/Wangt-CN/Image-text-matching/blob/master/fig/i2t_rerank.png" width="70%"/>
<br>


- ### T2I Re-ranking

    <img src="https://github.com/Wangt-CN/Image-text-matching/blob/master/fig/t2i_rerank.png" width="70%" />
<br>

- ### Retrieval Examples
<br>

![example](https://github.com/Wangt-CN/Image-text-matching/blob/master/fig/example_.png)
<br>
<br>

![demo](https://github.com/Wangt-CN/Image-text-matching/blob/master/fig/demo.gif)




## Data Download

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The precomputed image features of MS-COCO are from [here](https://github.com/peteanderson80/bottom-up-attention). The precomputed image features of Flickr30K are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from:

```
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
wget https://scanproject.blob.core.windows.net/scan-data/vocab.zip
```

We refer to the path of extracted files for `data.zip` as `$DATA_PATH` and files for `vocab.zip` to `./vocab` directory. Alternatively, you can also run vocab.py to produce vocabulary files. For example,

```
python vocab.py --data_path data --data_name coco_precomp
```



## Usage

### Code Structure

```
├── MTFN-RR/
|   ├── engine.py           /* Files contain train/validation code
|   ├── model.py            /* Files for the model layer
|   ├── data.py             /* Files for construct dataset
|   ├── utils.py            /* Files for tools
|   ├── train.py             /* Files for main code
|   ├── re_rank.py         /* Files for re-ranking in testing stage
|   ├── vocab.py           /* Files for construct vocabulary
|   ├── seq2vec.py         /* Files for sentence to vector
|   ├── readme.md
│   ├── option/
| 	├── MTFN_RR.yaml/                 /* setting file
```



- Training stage:
  - The optional parameter setting is in option/*.yaml
  - Run train.py (Note to modify the default settings at the top of python file)
- Testing stage:
  - save the similarity matrix obtained in training stage
  - Run rerank.py (Please note to modify the similarity matrix path)
