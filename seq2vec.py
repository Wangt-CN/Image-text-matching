import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import skipthoughts

def factory(vocab_words, opt):
    if opt['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           vocab_words,
                           dropout=opt['dropout'],
                           fixed_emb=opt['fixed_emb'])


    else:
        raise NotImplementedError
    return seq2vec