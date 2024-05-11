# %%
import requests
from tqdm import tqdm
from pathlib import Path
from os import remove


ROOT = r"C:\Users\egrli\FU_Projects\Курсач\intelligent_okved_embeddings\data\rsmp"



def download(url: str, fname: str) -> None:
    """Downloads file from url and saves as fname

    Args:
        url (str): url to download file from
        fname (str): file to save file to
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))

    try:
        with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f'There was an error while downloading {fname}')
        remove(fname)

# %%
if __name__ == '__main__':
    with open(f'{ROOT}/rsmp_urls', 'r', encoding='utf8') as fp:
        urls = fp.readlines()
    save_root = Path(ROOT)
    for url in urls:
        url = url.strip()
        file_name = Path(url).name
        save_path = save_root / file_name
        if save_path.exists():
            print(f'Skipping {url}...')
            continue
        download(url, str(save_path))
