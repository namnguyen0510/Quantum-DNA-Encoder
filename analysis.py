import pickle
import pandas as pd
from primitives import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch
from sklearn.feature_selection import mutual_info_regression
from visualize import *
import plotly as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio   
pio.kaleido.scope.mathjax = None
from scipy import stats

df = pd.read_csv('Achilles_guide_efficacy.csv')
opses = ['OHE','linear_comb','concat_depth','concat_width','outer_prod']
plot_names = ['OHE', 'QDE (MA)', 'QDE (CD)', 'QDE (CW)', 'QDE (CO)']
fig = make_subplots(1,5)

OHE=[]
linear_comb=[]
concat_depth=[]
concat_width=[]
outer_prod=[]

f_dict = [OHE,linear_comb,
concat_depth,
concat_width,
outer_prod]

score_dict = dict(zip(opses, f_dict))


for i,ops in enumerate(opses):
    df = df.sort_values(by = 'efficacy', ascending=False)#[:3000]
    idx = df.index
    print(df)
    print(idx)
    # Open the pickle file in binary mode
    with open('FeatureSets.pkl', 'rb') as file:
        # Load the dictionary from the file
        dictionary = pickle.load(file)

    file.close()

    if ops == 'OHE':
        H = [one_hot_encoding_dna(g).flatten() for g in df['gRNA']]

    else:
        # Now you can use the loaded dictionary
        H = [np.abs(dictionary[ops][i]) for i in idx]
        H = [h.flatten() for h in H]
    

    H = np.vstack(H)
    print(H.shape)
    score = mutual_info_regression(H,df['efficacy'])
    print(score)
    score_dict[ops].append(score)

    fig.add_trace(go.Box(y = score, name = plot_names[i]), row = 1, col = i+1)


for o_0 in opses:
    for o_1 in opses:
        if o_0 != o_1:
            ptest = stats.ttest_ind(np.array(score_dict[o_0]),
                                    np.array(score_dict[o_1]), axis = 1,
                                    equal_var = False,
                                    alternative='less',random_state = 42)
            print('Test Statistics (P-test): {} vs. {}, t-stat: {}, p-value: {}'.format(o_0,o_1,ptest[0],ptest[1]))



fig.update_yaxes(range=[0,0.02])
fig.update_layout(yaxis_title = '<b>Mutual Information</b><br>(Relevancy)')
fig.update_layout(showlegend = False)
fig.update_layout(font = dict(size = 19), width = 1960, height = 500)
fig.write_image('Box.pdf')
fig.show()
