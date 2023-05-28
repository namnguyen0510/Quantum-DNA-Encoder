from primitives import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch

from visualize import *

def QDNAEncoder(gstr,ops):
    dual = dual_dna(gstr)
    enc_gsstr = one_hot_encoding_dna(gstr)
    enc_gastr = one_hot_encoding_dna(dual)
    w = np.random.random(size=[3])/2*np.pi
    akernel = ASymmetricDNAEncoder(w)
    skernel = SymmetricDNAEncoder(w)
    qemb_s = [np.matmul(skernel,enc_gsstr[i]) for i in range(enc_gsstr.shape[0])]
    qemb_a = [np.matmul(akernel,enc_gastr[i]) for i in range(enc_gastr.shape[0])]
    qemb_s = np.vstack(qemb_s)
    qemb_a = np.vstack(qemb_a)
    if ops == 'linear_comb':
        genc = (qemb_a + qemb_s)/2
    elif ops == 'concat_depth':
        genc = np.concatenate([qemb_s,qemb_a], axis = 1)
    elif ops == 'concat_width':
        genc = np.concatenate([qemb_s,qemb_a], axis = 0)
    elif ops == 'inner_prod':
        genc = np.matmul(qemb_a.T,qemb_s) - np.matmul(qemb_s.T,qemb_a)
    elif ops == 'outer_prod':
        genc = np.matmul(qemb_a,qemb_s.T) - np.matmul(qemb_s,qemb_a.T)
    return genc
