from primitives import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch
from model import *
import pandas as pd
import tqdm
df = pd.read_csv('Achilles_guide_efficacy.csv')

opses = ['linear_comb','concat_depth','concat_width','inner_prod','outer_prod']
_grna = []
_linear_comb = []
_concat_depth = []
_concat_width = []
_inner_prod = []
_outer_prod = []
OPS = [_linear_comb,_concat_depth,_concat_width,_inner_prod,_outer_prod]
for i in tqdm.tqdm(range(len(df))):
    for no, ops in enumerate(opses):
        grna = df['gRNA'][i]
        OPS[no].append(QDNAEncoder(grna,ops))
        _grna.append(grna)

feat_dict = dict(zip(opses, OPS))
import pickle
# Save dictionary
with open('FeatureSets.pkl', 'wb') as file:
    pickle.dump(feat_dict, file)