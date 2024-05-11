# %%
import json
import pandas as pd
from pathlib import Path
import numpy as np
data_root = Path('../../data')
URL_PREFIX = 'https://www.list-org.com/'
# %%
def capital_to_float(s: str) -> float: # а что если прикрутить нейронку???
    """Converts a string with the unit into a float

    Args:
        s (str): string like "10 тыс. руб."

    Returns:
        float: converted value.
    """
    if pd.isna(s):
        return s
    val, unit, _ = s.split(' ')
    assert unit in {'млн.', 'тыс.'}
    if unit == 'млн.':
        mul = 1_000_000
    elif unit == 'тыс.':
        mul = 1_000
    return float(val) * mul
# %%
def create_ndata(companies: list[dict], people: list[dict]) -> tuple[pd.DataFrame]:
    """Creates two DataFrames containing companies data and people data
    with a slight preprocessing.

    Args:
        companies (list[dict]): companies data
        people (list[dict]): people data

    Returns:
        tuple[pd.DataFrame]: people data and companies data as DataFrames
    """
    p_ndata = pd.DataFrame(people)
    p_ndata['url'] = p_ndata['url'].str.replace(URL_PREFIX,
                                                '',
                                                regex=False)
    p_ndata['type'] = 'person'
    p_ndata['boss'] = p_ndata['boss'].replace({'': 0, 'True': 1})
    companies_df = pd.DataFrame(companies)

    c_ndata = companies_df.drop(columns=['parents',
                                         'children',
                                         'head'])
    c_ndata['url'] = c_ndata['url'].str.replace(URL_PREFIX,
                                                '',
                                                regex=False)
    c_ndata[['INN', 'KPP']] = c_ndata['INN_KPP'].str.split(' / ', expand=True)
    c_ndata['capital'] = c_ndata['capital'].map(capital_to_float)
    c_ndata.rename({'NaN': 'welfare_benefits'}, axis=1, inplace=True)
    assert not c_ndata['boss'].replace('', np.nan).any()
    c_ndata = c_ndata.loc[c_ndata['fullname'].isna()]
    c_ndata.drop(columns=['boss', 'INN_KPP', 'inn', 'fullname'],
                 inplace=True)
    return p_ndata, c_ndata
# %%
def create_edata(companies: list[dict]) -> pd.DataFrame:
    """Creates DataFrame containing three columns: source node,
    target node and edge type.

    Args:
        companies (list[dict]): companies data

    Returns:
        pd.DataFrame: edges data as DataFrame
    """
    edata = []
    for company in companies:
        u = company['url'].replace('https://www.list-org.com/', '')
        parentage = []
        for field in ['parents', 'children']:
            if field in company:
                parentage.extend(company[field])
        for *_, v in parentage:
            if v:
                v = v.replace('https://www.list-org.com/', '')
                edata.append((u, v, 'parentage'))
                edata.append((v, u, 'parentage')) # why do we use undirected graph in this case??? (EGOR)

        if 'head' in company and company['head']:
            v = company['head'][-1].replace('https://www.list-org.com/', '')
            edata.append((u, v, 'management'))
            edata.append((v, u, 'management'))
    return pd.DataFrame(edata, columns=['u', 'v', 'type'])

# %%
if __name__ == '__main__':
    with open(data_root / 'companies.json', encoding='utf8') as fp:
        data = json.load(fp)
    with open(data_root / 'second_nodes.json', encoding='utf8') as fp:
        data.extend(json.load(fp))

    assert all(o['type'] in {'company', 'man'} for o in data)
    # there could be duplicates in two files
    companies = {o['url']: o for o in data if o['type']=='company'}.values()
    people = {o['url']: o for o in data if o['type']=='man'}.values()
    people_ndata, companies_ndata = create_ndata(companies, people) # создание данных для вершин
    edata = create_edata(companies)
    print(f"Found {len(companies_ndata)} companies, "
          f"{len(people_ndata)} people and "
          f"{len(edata)} edges.")
    people_ndata.to_csv(data_root / 'people_ndata.csv', index=False) # инфа о людях (вершины графа)
    companies_ndata.to_csv(data_root / 'companies_ndata.csv', index=False) # инфа о компаниях (вершины графа)
    edata.to_csv(data_root / 'edata.csv', index=False) # информация о ребрах: u -> v двух типов: управление, дочка
