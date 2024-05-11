# %%
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from pathlib import Path

okved_root = Path('../../data/okved2')

def prepocess_okved_data(okved_path: str,
                         sections_path: str,
                         save_path: str) -> None:
    """Reads two separate files with information about OKVED, merges them
    and reindexes rows starting from 1.

    Args:
        okved_path (str): path to csv file with OKVED info
        sections_path (str): path to csv file with OKVED sections info
        save_path (str): path to save result to
    """
    okved = pd.read_csv(okved_path,
                        dtype={'code': str,
                               'parent_code': str,
                               'native_code': str})
    okved_sec = pd.read_csv(sections_path)
    okved = (okved.merge(okved_sec,
                        left_on='section_id',
                        right_on='id',
                        suffixes=['_okved', '_section'])
                  .drop(columns=['id']))
    okved.index = range(1, len(okved) + 1)
    okved.to_csv(save_path)

def embed_bert_cls(text: str,
                   model: AutoModel,
                   tokenizer: AutoTokenizer,
                   truncation=True) -> np.ndarray:
    """A helper function to retrive the text embedding using specified model and tokenizer.

    Args:
        text (str): text to embed
        model (AutoModel): embedding model
        tokenizer (AutoTokenizer): tokenizer associated with the model

    Returns:
        np.ndarray: text embedding
    """
    t = tokenizer(text, padding=True, truncation=truncation, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**t)
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()

def get_okved_embeddings(okved: pd.DataFrame) -> np.ndarray:
    """Builds OKVED embeddings matrix using pretrained rubert model.

    Args:
        okved (pd.DataFrame): dataframe with information about OKVEDs

    Returns:
        np.ndarray: OKVED bert embeddings matrix
    """
    text_cols = ['name_section', 'name_okved', 'comment']
    okved['full_text'] = okved[text_cols].fillna('').agg(' '.join, axis=1)
    model_name = "cointegrated/rubert-tiny2" # эмбеддинги- 312 (неужели мы в узлы больше ничего не берем)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    okved_embeddings = []
    for text in tqdm(okved['full_text']):
        e = embed_bert_cls(text,
                           model,
                           tokenizer)
        okved_embeddings.append(e) # эмбеддинги описаний оквэдов
    return np.concatenate(okved_embeddings, axis=0)
# %%
if __name__ == '__main__':
    prepocess_okved_data(okved_root / 'okved_2014.csv', # по ощущениям это тоже самое
                         okved_root / 'okved_2014_sections.csv', # глобальная инфа
                         okved_root / 'okved_2014_w_sections.csv') # уточненная
    okved = pd.read_csv(okved_root / 'okved_2014_w_sections.csv',
                        index_col=0)
    okved_embeddings = get_okved_embeddings(okved)
    np.save(okved_root / 'okved_embeddings.npy', okved_embeddings) # эмбеддинги формы 2636, 312
