# Introduction

This is building code for image-text matching.  



<br>

## Retrieval Examples

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
