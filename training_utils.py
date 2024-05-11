import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import dgl.function as fn
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import torch.nn.functional as F
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import dgl.nn as gnn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import faiss


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False, device=None):
        if device is None:
            device = g.device
        self.weights = g.in_degrees().float().to(device) ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g):
        src, _ = g.edges()
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return dgl.graph((src, dst), num_nodes=g.num_nodes())


class CrossEntropyLoss(nn.Module):
    def forward(self, h, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = h
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = h
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss
    

def train(model, sampler, graph, writer, num_epochs,
          nfeat, criterion, optimizer, require_improvements, best_loss=10_000):
    for epoch in range(num_epochs):
        neg_graph = sampler(graph)
        pred = model(graph, nfeat)
        loss = criterion(pred, graph, neg_graph)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)
        if loss.item() < best_loss:
            best_loss = loss.item()
            last_improvement = 0
            best_state = model.state_dict()
        else:
            last_improvement += 1
        if last_improvement > require_improvements:
            print(f"No improvement found during the {require_improvements} last iterations, stopping optimization. on epoch = {epoch}")
            print(f"Best loss = {best_loss}")
            model.load_state_dict(best_state)
            break
    print(f"Best loss = {best_loss}")
  

def fix_okved(o: str) -> str:
    if o is None:
        return

    while o[-1] == '0' and '.' in o:
        if o[-2] == '.':
            o = o[:-2]
        else:
            o = o[:-1]
    return o

def eval_model(version_name, bert_use=True):
    okved_data = pd.read_csv('../data/okved2/okved_2014_w_sections.csv', index_col=0)
    sections = okved_data['section_id'].values

    idx_to_okved = {0: 'root'}
    idx_to_okved |= okved_data['native_code'].to_dict()
    okved_to_idx = {v: k for k, v in idx_to_okved.items()}

    embeddings_bert = np.load('../data/okved2/okved_embeddings.npy')
    # creating mapping for gos description
    embeddings_model = np.load(f'../results/embs/{version_name}.npy')
    okved_consumption = pd.read_csv('../data/stats/okved_consumption.csv')

    prov_indices = okved_consumption['okved_provider'].map(fix_okved).map(okved_to_idx)
    cons_indices = okved_consumption['okved_consumer'].map(fix_okved).map(okved_to_idx)
    X_bert = np.column_stack((embeddings_bert[prov_indices], embeddings_bert[cons_indices]))
    X_bert = StandardScaler().fit_transform(X_bert)
    X_model = np.column_stack((embeddings_model[prov_indices], embeddings_model[cons_indices]))
    X_model = StandardScaler().fit_transform(X_model)
    y = okved_consumption['normalized_consumption']
    model = LinearRegression()

    bert_scores = []
    linear_regression_score_bert, svm_score_bert = 0, 0
    if bert_use:
        model = LinearRegression().fit(X_bert, y)
        linear_regression_score_bert = model.score(X_bert, y)
        model = SVR().fit(X_bert, y)
        svm_score_bert = model.score(X_bert, y)
        for i in tqdm(range(100), leave=False):
            model = MLPRegressor().fit(X_bert, y)
            score = model.score(X_bert, y) 
            bert_scores.append(score)
    model_scores = []
    model = LinearRegression().fit(X_model, y)
    linear_regression_score_model = model.score(X_model, y)
    model = SVR().fit(X_model, y)
    svm_score_model = model.score(X_model, y)
    for i in tqdm(range(100), leave=False):
        model = MLPRegressor().fit(X_model, y)
        score = model.score(X_model, y) 
        model_scores.append(score)
    return bert_scores, linear_regression_score_bert, svm_score_bert, model_scores, linear_regression_score_model, svm_score_model


def get_embs_and_plot(embeddings_model, model_name, sections):        
    embeddings_model_2d = TSNE(n_components=2).fit_transform(embeddings_model)
    embeddings_bert = np.load('../data/okved2/okved_embeddings.npy')
    embeddings_bert_2d = TSNE(n_components=2).fit_transform(embeddings_bert)
    
    fig, axs = plt.subplots(figsize=(10, 5), ncols=2)
    axs[0].scatter(embeddings_bert_2d[:, 0], embeddings_bert_2d[:, 1], c=sections)
    axs[0].set_title('TSNE of BERT embeddings')
    axs[1].scatter(embeddings_model_2d[:sections.shape[0], 0], embeddings_model_2d[:sections.shape[0], 1], c=sections)
    axs[1].set_title(f'TSNE of {model_name} embeddings')
    plt.savefig(f"../results/tsne/{model_name}.png")
    np.save(file=f"../results/embs/{model_name}.npy", arr=embeddings_model)
    np.save(file=f"../results/embs/{model_name}_2d.npy", arr=embeddings_model_2d)


def cut(s: str, ln: int = 10) -> str:
    if len(s) >= ln:
        return s[:ln - 3] + '...'
    return s
    
def draw_2d(embeddings_2d: np.array,
            names: list,
            colors: list,
            xlim: tuple = None,
            ylim: tuple = None,
            figsize: tuple = (15, 5),
            annotate: bool = False,
            hide_spins: tuple = None,
            node_size: int = 5) -> None:
    """
    Рисует проекции эмбеддингов на плоскости
    Args:
        embeddings_2d (np.array): массив эмбеддингов узлов
        g (dgl.DGLHeteroGraph): граф для обучения
        okved_data (pd.DataFrame): таблица с информацией об ОКВЭД
        okved_to_section (dict): маппинг код раздела ОКВЭД - номер кода
        idx_to_okved (dict): маппинг номер кода ОКВЭД - код
        xlim (tuple, optional): ограничения по оси х
        ylim (tuple, optional): ограничения по оси у
        figsize (tuple, optional): размер фигуры
        annotate (bool, optional): True, если нужно добавить подписи к точкам (лучше не применять на полном датасете)
        name_len (int, optional): максимальная длина названия кода
        hide_spins (tuple, optional): какие рамки скрывать
        node_size (int, optional): размер точки
    """
    fig, ax = plt.subplots(figsize=figsize)
    if xlim:
        x_mask = (embeddings_2d[:, 0] >= xlim[0]) & (embeddings_2d[:, 0] <= xlim[1])
    else:
        x_mask = np.ones(len(embeddings_2d)).astype(bool)
    if ylim:
        y_mask = (embeddings_2d[:, 1] >= ylim[0]) & (embeddings_2d[:, 1] <= ylim[1])

    else:
        y_mask = np.ones(len(embeddings_2d)).astype(bool)

    mask = x_mask & y_mask
    embs = embeddings_2d[mask]

    colors = colors[mask]
    names = names[mask]
    ax.scatter(embs[:, 0], embs[:, 1], s=node_size, c=colors)
    ax.set_xlabel('$h_0(v)$')
    ax.set_ylabel('$h_1(v)$')
    if annotate:
        for (x, y), txt in zip(embs, names):
            ax.text(x, y, txt, rotation=45)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if hide_spins is not None:
        for spin in hide_spins:
            ax.spines[spin].set_visible(False)
    ax.set_title('Fragment of an illustration of t-SNE on OKVED code embeddings ', y=1.8, pad=-14)