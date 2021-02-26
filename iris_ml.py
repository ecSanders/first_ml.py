#%% modules and data 
import pandas as pd
import numpy as np
import altair as alt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
            
# %% Visualize
chart = (alt.Chart(df)
    .encode(
        alt.X('sepal length (cm)'),
        alt.Y('petal length (cm)'),
        alt.Color('target')
    ).mark_circle()
)
chart

chart = (alt.Chart(df)
    .encode(
        alt.X('sepal width (cm)'),
        alt.Y('petal width (cm)'),
        alt.Color('target')
    ).mark_circle()
)
chart





# %% Prep NN
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

#%% Create model
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x