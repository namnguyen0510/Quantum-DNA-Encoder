from primitives import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch
from model import *
import pandas as pd
import tqdm

ops = 'concat_depth'

gstr = 'ATGCTCGA'


df = pd.read_csv('Achilles_guide_efficacy.csv')

opses = ['linear_comb','concat_depth','concat_width','inner_prod','outer_prod']
_grna = []
_linear_comb = []
_concat_depth = []
_concat_width = []
_inner_prod = []
_outer_prod = []

OPS = [_linear_comb,_concat_depth,_concat_width,_inner_prod,_outer_prod]

for i in tqdm.tqdm(range(11)):
    for no, ops in enumerate(opses):
        grna = df['gRNA'][i]
        pam_p2 = grna[-8:]
        #print(grna)
        #print(pam_p2)
        OPS[no].append(QDNAEncoder(pam_p2,ops))
        _grna.append(grna)

print(OPS)
_df = pd.DataFrame([])
_df['gRNA'] = _grna
_df['linear_comb']=_linear_comb
_df['concat_depth']=_concat_depth
_df['concat_width']=_concat_width
_df['inner_prod']=_inner_prod
_df['outer_prod']=_outer_prod

print(_df)
_df.to_csv('meta_feaures.csv', index = False)
