from copy import deepcopy
import random
import os
from os.path import isfile
from pathlib import Path
import pickle
from tqdm.notebook import tqdm

from IPython.display import display
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, mean_squared_error, classification_report
from sklearn.metrics import mean_squared_error, f1_score


from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
import dgl.nn as gnn
import dgl.function as fn

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
dgl.seed(2)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
