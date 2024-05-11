# %%
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
from itertools import combinations
import dgl
import torch as th
from typing import Union
import joblib
from copy import deepcopy
from collections import Counter

GRAPH_ROOT = Path('C:/Users/egrli/FU_Projects/Курсач/intelligent_okved_embeddings/data/graph')
OKVED_ROOT = Path('C:/Users/egrli/FU_Projects/Курсач/intelligent_okved_embeddings/data/okved2')

# %%
def fix_okved(o: str) -> str:
    """Removes extra zeros at the end of the code

    Args:
        o (str): raw OKVED code

    Returns:
        str: OKVED code with extra zeros at the end removed
    """
    if o is None:
        return

    while o[-1] == '0' and '.' in o:
        if o[-2] == '.':
            o = o[:-2]
        else:
            o = o[:-1]
    return o

def create_okveds_list(main: str, extra: list[str]) -> list[str]:
    """Create a flat list containing both main company OKVED and extra OKVEDs.

    Args:
        main (str): main company OKVED
        extra (list[str]): extra company OKVEDs

    Returns:
        list[str]: flat list containing all company OKVEDS
    """
    all_okveds = [main, *extra]
    return [fixed for o in all_okveds if (fixed := fix_okved(o)) in okved_to_idx]

def read_okved_embeddings(path: Union[Path, str], add_root_first: bool = True) -> th.Tensor:
    """Reads OKVED embeddings from the file.

    Args:
        path (Path | str): path to npy file containing OKVED embeddings
        add_root_first (bool, optional): whether to add theextra row containing
                                         zeros at the beginning. Defaults to True.

    Returns:
        th.Tensor: tensor containing OKVED embeddings
    """
    okved_embeddings = np.load(path)
    if add_root_first:
        root_emb = np.zeros((1, okved_embeddings.shape[-1]))
        okved_embeddings = np.r_[root_emb, okved_embeddings]
    okved_embeddings = th.from_numpy(okved_embeddings)
    return okved_embeddings

def push_reverse_eid_to_end(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Сортирует ребра так, чтобы взаимообратные ребра находились на расстоянии len(edge df) // 2

    Args:
        edge_df (pd.DateFrame): таблица с ребрами

    Returns:
        pd.DataFrame: таблица c ребрами отсортированными так, чтобы взаимообратные ребра
        находились на расстоянии len(edge df) // 2
    """
    edge_df = edge_df.copy()
    edges_sort = {}
    curr_idx = 0
    edge_to_eid = {(u, v): eid for eid, (u, v) in enumerate(edge_df[['u', 'v']].values)}
    for (u, v) in edge_df[['u', 'v']].values:
        if edge_to_eid[(u, v)] not in edges_sort:
            edges_sort[edge_to_eid[(u, v)]] = curr_idx
            edges_sort[edge_to_eid[(v, u)]] = curr_idx + len(edge_df) // 2
            curr_idx += 1

    edge_df['order'] = edge_df.index.map(edges_sort)
    edge_df = edge_df.sort_values('order').reset_index(drop=True).drop(columns='order')
    return edge_df

def create_cc_edges(edata: pd.DataFrame,
                    company_to_okved_flat: pd.Series,
                    okved_to_idx: dict[str, int]) -> pd.DataFrame:
    """Creates edges based on the information about companies relationships.

    Args:
        edata (pd.DataFrame): links data from listorg preprocessed by the `build_listorg_dataset` script
        company_to_okved_flat (pd.Series): series containing all company OKVEDs row by row
        okved_to_idx (dict[str, int]): a mapping from OKVED code to its ID

    Returns:
        pd.DataFrame: The resulting dataframe contains the following columns: u_code (str), v_code (str),
                      u (int), v (int), weight (int), direction (str), type = 0.
    """
    # only company-company edges
    comp_comp_edges = edata[(edata['u'].str.startswith('company')) &
                            (edata['v'].str.startswith('company'))].copy()
    # replace url with company INN
    comp_comp_edges['u'] = comp_comp_edges['u'].map(company_inn)
    comp_comp_edges['v'] = comp_comp_edges['v'].map(company_inn)
    # there could be companies without any data collected
    comp_comp_edges.dropna(inplace=True)
    cc_edges_data = (comp_comp_edges
                    .merge(company_to_okved_flat, left_on='u', right_index=True)
                    .merge(company_to_okved_flat, left_on='v', right_index=True)
                    .loc[:, ['u', 'okved_code_x', 'v', 'okved_code_y', 'type']]
                    )
    # пока считаем все типы одинаковыми и считаем кол-во ребер между парами ОКВЭДов
    cc_edges = cc_edges_data.groupby(['okved_code_x', 'okved_code_y']).size()

    # убираем петли и редкие связи
    cc_edges = cc_edges[cc_edges.index.get_level_values(1) != cc_edges.index.get_level_values(0)]
    cc_edges = cc_edges.to_frame(name='weight').reset_index()
    cc_edges = cc_edges.query('weight > 1').reset_index(drop=True)
    # маппим названия ОКВЭД на целые числа и получаем веса ребер
    cc_edata = (cc_edges.loc[:, ['okved_code_x', 'okved_code_y', 'weight']]
                        .rename(columns={'okved_code_x': 'u_code',
                                         'okved_code_y': 'v_code'}))

    cc_edata['u'] = cc_edata['u_code'].map(okved_to_idx)
    cc_edata['v'] = cc_edata['v_code'].map(okved_to_idx)
    # Company-Company relationship type ID = 0
    cc_edata['type'] = 0
    cc_edata['direction'] = 'forward'
    # sort edges so the reverse edges are in the second half of the dataframe
    cc_edata = push_reverse_eid_to_end(cc_edata)
    half = len(cc_edata) // 2
    cc_edata.loc[half:, 'direction'] = 'reverse'
    # sanity check
    assert np.all(cc_edata[:half][['u', 'v']].values == cc_edata[half: ][['v', 'u']].values)
    return cc_edata

def create_clf_edges(okved_data: pd.DataFrame,
                     okved_to_idx: dict[str, int]) -> pd.DataFrame:
    """Creates edges based on the OKVED tree.

    Args:
        okved_data (pd.DataFrame): merged OKVED dataset generated by `build_okved_dataset.prepocess_okved_data`
        okved_to_idx (dict[str, int]): a mapping from OKVED code to its ID

    Returns:
        pd.DataFrame: The resulting dataframe contains the following columns: u_code (str), v_code (str),
                      u (int), v (int), weight (int), direction (str), type = 1.
    """
    # build okved codes tree as a dict
    digit_code_to_okved = (okved_data.loc[:, ['code_okved', 'native_code']]
                                     .set_index('code_okved')
                                     .loc[:, 'native_code']
                                     .to_dict())
    # add extra root node which is not real OKVED
    digit_code_to_okved['000000'] = 'root'
    assert len(digit_code_to_okved) - 1 == len(okved_data)

    classifier_edges = okved_data[['code_okved', 'parent_code']].copy()
    classifier_edges['code_okved'] = classifier_edges['code_okved'].map(digit_code_to_okved)
    classifier_edges['parent_code'] = classifier_edges['parent_code'].map(digit_code_to_okved)
    # add reverse edges at the end of DF s.t. the graph is not tree anymore
    classifier_edges_rev = classifier_edges.iloc[:, ::-1]
    classifier_edata = pd.DataFrame(np.concatenate([classifier_edges.values,
                                                    classifier_edges_rev.values],
                                                axis=0),
                                    columns=['u_code', 'v_code'])
    # маппим названия ОКВЭД на целые числа и получаем веса ребер из классификатора
    classifier_edata['weight'] = 1 # не очень понятный аспект
    classifier_edata['u'] = classifier_edata['u_code'].map(okved_to_idx)
    classifier_edata['v'] = classifier_edata['v_code'].map(okved_to_idx)
    # Classifier relationship type ID = 1 поправил, было 0!!!
    classifier_edata['type'] = 1
    classifier_edata['direction'] = 'forward'
    half = len(classifier_edata) // 2
    # no need to sort edges as in the other functions
    classifier_edata.loc[half:, 'direction'] = 'reverse'
    # sanity check
    assert np.all(classifier_edata[:half][['u', 'v']].values == classifier_edata[half:][['v', 'u']].values)
    return classifier_edata

def create_extra_edges(company_to_okveds_lst: dict[str, list],
                       okved_to_idx: dict[str, int]) -> pd.DataFrame:
    """Creates edges based on the information about different OKVED codes for the same companies.

    Args:
        company_to_okveds_lst (dict[str, list]): dict containing a list of possible OKVEDs for each company
        okved_to_idx (dict[str, int]): a mapping from OKVED code to its ID

    Returns:
        pd.DataFrame: The resulting dataframe contains the following columns: u_code (str), v_code (str),
                      u (int), v (int), weight (int), direction (str), type = 2.
    """
    # take each pair of codes for the same company and count the pairs' frequencies
    extra_edges = []
    for okveds in tqdm(company_to_okveds_lst.values()):
        okveds = [fix_okved(o) for o in okveds]
        # sorted is required s.t. we can be sure that for (u, v) pair from one company
        # there won't be any (v, u) pairs from another company
        extra_edges.extend(combinations(sorted(okveds), r=2))
    extra_edges_data = pd.DataFrame(extra_edges, columns=['u_code', 'v_code'])
    extra_edges = extra_edges_data.groupby(['u_code', 'v_code']).size()

    # убираем петли и редкие связи
    extra_edges = extra_edges[(extra_edges.index.get_level_values(1) !=
                               extra_edges.index.get_level_values(0))]
    extra_edges = extra_edges.to_frame(name='weight').reset_index()
    # edges between OKVEDs with the same class are too frequent so we ignore them
    not_same_class = (extra_edges['u_code'].str.slice(0, 3) !=
                      extra_edges['v_code'].str.slice(0, 3))
    # the sample is big enough to cut edges with low weights
    often_enough = extra_edges['weight'] > 10
    extra_edges = extra_edges[not_same_class & often_enough].reset_index(drop=True)
    extra_edges_rev = extra_edges.loc[:, ['v_code', 'u_code', 'weight']]
    extra_edges_rev.columns = ['u_code', 'v_code', 'weight']
    # add reverse edges to the end
    extra_edges = pd.concat([extra_edges, extra_edges_rev],
                            axis=0,
                            ignore_index=True)

    extra_edges['u'] = extra_edges['u_code'].map(okved_to_idx)
    extra_edges['v'] = extra_edges['v_code'].map(okved_to_idx)
    # Company-Company relationship type ID = 2
    extra_edges['type'] = 2
    extra_edges['direction'] = 'forward'
    # sort edges so the reverse edges are in the second half of the dataframe
    # probably unnecessary here
    extra_edges = push_reverse_eid_to_end(extra_edges)
    half = len(extra_edges) // 2
    extra_edges.loc[half:, 'direction'] = 'reverse'
    # sanity check
    assert np.all(extra_edges[:half][['u', 'v']].values == extra_edges[half:][['v', 'u']].values)
    return extra_edges

def create_gc_edges(start_prices, okved_to_idx):
    """
    Creates edges for gos contracts sum between okveds
    """
    start_prices = start_prices.rename(columns={
        "executor_okved" : "u_code",
        "customer_okved" : "v_code",
        "avg_gos" : "weight"
    }).drop(columns=[
        "count_gos", "sum_gos"
    ])
    start_prices['type'] = 3
    start_prices['direction'] = 'forward'
    start_prices = start_prices[start_prices['u_code'].apply(lambda x: x in okved_to_idx) & start_prices['v_code'].apply(lambda x: x in okved_to_idx)]
    start_prices['u'] = start_prices['u_code'].apply(lambda x: okved_to_idx[x])
    start_prices['v'] = start_prices['v_code'].apply(lambda x: okved_to_idx[x])
    start_prices = start_prices[start_prices['weight']>0]
    return start_prices[["u_code", "v_code", "weight", "u", "v", "type","direction"]]

def build_graph(okved_embeddings: th.FloatTensor,
                # gos_contracts_embeddings : th.FloatTensor,
                *edge_dfs: tuple[pd.DataFrame]) -> dgl.graph:
    """Creates DGL Graph based on edges from edge_dfs.

    Args:
        okved_embeddings (th.FloatTensor): tensor containing OKVED embeddings

    Returns:
        dgl.graph: OKVEDs graph.
    """
    # loop over directions then over different types
    # s.t. reverse edges are at the end of the table
    # after concat
    directions = ('forward', 'reverse')
    parts = [df[df['direction']==direction]
                 for direction in directions
                 for df in edge_dfs]
    all_edges = pd.concat(parts, axis=0, ignore_index=True)
    # build the graph from df
    src = all_edges['u'].values
    dst = all_edges['v'].values
    weight = th.from_numpy(all_edges['weight'].values)
    type_edges = th.from_numpy(all_edges['type'].values)

    g = dgl.graph((src, dst), num_nodes=len(idx_to_okved))
    # add extra features on nodes and edges
    g.ndata['feat'] = okved_embeddings # HERE ADD INFO EMBS
    # g.ndata['gos'] = okved_embeddings # HERE ADD INFO EMBS
    # g.ndata['feat_2'] = gos_contracts_embeddings
    g.edata['weight'] = weight
    g.edata['type'] = type_edges

    # create train mask s.t. if the edge (u, r, v) is in train set
    # then the edge (v, r, u) will be in train set, too.
    train_mask = th.zeros(g.num_edges()).bool()
    half_all_edges = len(all_edges) // 2
    types = all_edges['type'].unique()
    g.num_edges_type = {}
    g.num_rels = len(types)
    for type_ in types:
        typed = all_edges.query('type == @type_')
        half_typed_edges = len(typed) // 2
        train_forward = (typed.iloc[:half_typed_edges]
                              .sample(frac=0.8)
                              .index)
        train_reverse = train_forward + half_all_edges
        train_mask[train_forward] = True
        train_mask[train_reverse] = True
        g.num_edges_type[type_] = len(typed)
    g.edata['train_mask'] = train_mask

    # normalize edge weights within the subgraph
    # induced by the relevant edge type
    # and add `norm` feature
    norm = dgl.nn.EdgeWeightNorm()
    g.edata['norm'] = th.zeros_like(g.edata['weight'], dtype=th.float32)
    for type_ in types:
        edges_of_type = (g.edata['type'] == type_).nonzero().flatten()
        edge_subgraph = dgl.edge_subgraph(graph=g,
                                          edges=edges_of_type,
                                          store_ids=True)
        eids = edge_subgraph.edata['_ID']
        g.edata['norm'][eids] = norm(edge_subgraph, edge_subgraph.edata['weight'].float())

    return g

# %%
if __name__ == '__main__':
    companies_ndata = pd.read_csv(GRAPH_ROOT / 'companies_ndata.csv',
                                  usecols=['url', 'INN'],
                                  dtype={'INN': str}) # информация об узлах компании- 
    ################################################### name,url,founders_amount,registration_date,okved_code,type
    company_links = pd.read_csv(GRAPH_ROOT / 'edata.csv') # ИНФО О РЕБРАХ
    okved_data = pd.read_csv(OKVED_ROOT / 'okved_2014_w_sections.csv',
                             index_col=0,
                             dtype={'code': str,
                                    'code_okved': str,
                                    'parent_code': str,
                                    'native_code': str})
    start_prices = pd.read_csv(OKVED_ROOT / "start_prices.csv", index_col=0)
    with open(OKVED_ROOT / 'company_inn_okveds_rmsp.pickle', 'rb') as fp:
        company_to_okved = pickle.load(fp) # оквэд-классификация ОКВЭДОВ (тоже граф кстати)
    okved_embeddings = read_okved_embeddings(OKVED_ROOT / 'okved_embeddings.npy') # Эмбеддинги названий ОКВЭД
    company_inn = companies_ndata.set_index('url')['INN']
    inns = set(company_inn)
    gos_contracts_embeddings = np.load(OKVED_ROOT / "okved_contact_emb.npy")
    okved_to_number_mapping = joblib.load(filename=OKVED_ROOT / 'okved_to_number_mapping.pickle')
    okved_embeddings = th.from_numpy(np.concatenate((okved_embeddings, gos_contracts_embeddings)))
    idx_in_extra_nodes = np.where(gos_contracts_embeddings[:, 0]!=0)[0]      
    extra_nodes_clear = list(okved_to_number_mapping.keys())
    extra_nodes = list(
        map(
            lambda x: x + '_',
            extra_nodes_clear
        )
    )
    idx_to_okved = {0: 'root'}
    idx_to_okved |= okved_data['native_code'].to_dict()
    # for i, okved in enumerate(extra_nodes, len(idx_to_okved)):
    #     idx_to_okved[i] = okved
    okved_to_idx = {v: k for k, v in idx_to_okved.items()}
    company_to_okveds_lst = {inn: create_okveds_list(**ok)
                                for inn, ok in tqdm(company_to_okved.items())
                                if inn in inns} # компания к ОКВЭДУ
    company_to_okved_flat = (pd.Series(company_to_okveds_lst)
                               .explode()
                               .rename('okved_code')
                               .map(fix_okved)) # ВТОРАЯ НОРМАЛЬНАЯ ФОРМА
    # company_to_okved_flat = pd.concat(
    #     (
    #         company_to_okved_flat,
    #         company_to_okved_flat[company_to_okved_flat.apply(lambda x: x in extra_nodes_clear)] + '_'
    #     )
    # )
    # extra_nodes_clear_joined = list(
    #     map(
    #         lambda x: ''.join(x.split('.')),
    #         extra_nodes_clear
    #     )
    # )
    # okved_data_extra = deepcopy(okved_data[okved_data['code_okved'].apply(lambda x: x in extra_nodes_clear_joined) | okved_data['parent_code'].apply(lambda x: x in extra_nodes_clear_joined)])
    # extra_nodes_okved_data = []


    # okved_data_extra = deepcopy(okved_data[okved_data['native_code'].apply(lambda x: x in extra_nodes_clear)])
    # okved_data_extra['native_code'] += '_'
    # okved_data_extra['code_okved'] += '_'
    # okved_data = pd.concat((okved_data, okved_data_extra)).reset_index(drop=True)
    # for k in company_to_okveds_lst:
    #     lst = company_to_okveds_lst[k]
    #     for el in lst:
    #         if el in extra_nodes_clear:
    #             lst.append(el + '_')
    #     company_to_okveds_lst[k] = lst
    # extra_nodes_okved_data_gos = []
    # okved_data_extra_gos = start_prices[start_prices['executor_okved'].apply(lambda x: x in extra_nodes_clear)|start_prices['customer_okved'].apply(lambda x: x in extra_nodes_clear)]
    # for _, row in okved_data_extra_gos.iterrows():
    #     if row['executor_okved'] in extra_nodes_clear and row['executor_okved'] in extra_nodes_clear:
    #         element1 = deepcopy(row)
    #         element2 = deepcopy(row)
    #         element3 = deepcopy(row)
    #         element1['executor_okved'] += '_'
    #         element2['customer_okved'] += '_'
    #         element3['executor_okved'] += '_'
    #         element3['customer_okved'] += '_'
    #         extra_nodes_okved_data_gos.extend([element1, element2, element3])
    #     elif row['executor_okved'] in extra_nodes_clear and row['executor_okved'] not in extra_nodes_clear:
    #         element1 = deepcopy(row)
    #         element1['executor_okved'] += '_'
    #         extra_nodes_okved_data_gos.append(element1)
    #     else:
    #         element1 = deepcopy(row)
    #         element1['customer_okved'] += '_'
    #         extra_nodes_okved_data_gos.append(element1)     
    # extra_nodes_okved_data_gos = pd.DataFrame(extra_nodes_okved_data_gos)
    cc_edges = create_cc_edges(company_links, company_to_okved_flat, okved_to_idx) # ЭТО Веса ребер по числу компаний

    clf_edges = create_clf_edges(okved_data, okved_to_idx)

    extra_edges = create_extra_edges(company_to_okveds_lst, okved_to_idx) # сколько компаний имеют два кода одновременно

    # gos_contracts_edges = create_gc_edges(start_prices=start_prices, okved_to_idx=okved_to_idx)

    codes = list(set([v['main'] for v in company_to_okved.values()]))
    codes_to_extra = {code : [] for code in codes}
    for v in company_to_okved.values():
        codes_to_extra[v['main']].extend(v['extra'])
    for code in codes_to_extra:
        codes_to_extra[code] = dict(Counter(codes_to_extra[code]))
    src, tgt, weights = [], [], []

    for k, v in codes_to_extra.items():
        try:
            node = okved_to_idx[k]
            counts, nodes_extra = [], []
            for el in v:
                try:
                    nodes_extra.append(okved_to_idx[el])
                    weights.append(v[el])
                except Exception:
                    pass
            src.extend([okved_to_idx[k]] * len(nodes_extra))
            tgt.extend(nodes_extra)
        except Exception:
            pass
    src, tgt, weights = list(map(lambda x: th.tensor(x), [src, tgt, weights]))
    mask = weights >= 3
    src, tgt, weights = src[mask], tgt[mask], weights[mask]
    extra_okved = pd.DataFrame([src.tolist(), tgt.tolist()], index=['u', 'v']).T
    extra_okved['weight'] = weights.tolist()
    extra_okved['type'] = 4
    extra_okved['direction'] = 'forward'
    # g = build_graph(okved_embeddings, cc_edges, clf_edges, extra_edges, gos_contracts_edges)


    g = build_graph(okved_embeddings,
                    cc_edges,
                    clf_edges,
                    extra_edges,
                    # gos_contracts_edges,
                    extra_okved)
    with open(GRAPH_ROOT / 'okved_graph_new_v2.pickle', 'wb') as fp:
        pickle.dump(g, fp)
