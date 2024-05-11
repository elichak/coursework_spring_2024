# %%
from bs4 import BeautifulSoup
from pathlib import Path
import pickle
from tqdm import tqdm
from multiprocessing import Pool
# %%

ROOT = r"C:\Users\egrli\FU_Projects\Курсач\intelligent_okved_embeddings\data\rsmp"


def get_okveds_from_file(file: Path) -> dict[str, dict]:
    """Opens file from RSMP dataset, read all the OKVEDs for every company in it.

    Args:
        file (Path): file from RSMP dataset

    Returns:
        dict[str, dict]: dictionary with main and extra OKVED codes for company
    """
    with open(file, encoding='utf8') as fp:
        xml = fp.read()
    soup = BeautifulSoup(xml, 'xml')
    okveds = {}
    for doc in soup.find_all('Документ'):
        org = doc.find('ОргВклМСП')
        if org is not None:
            inn = org.attrs['ИННЮЛ']
            okved_main = main_tag.attrs['КодОКВЭД'] if (main_tag := doc.find('СвОКВЭДОсн')) is not None else None
            okved_extra = [o.attrs['КодОКВЭД'] for o in doc.find_all('СвОКВЭДДоп')]
            okveds[inn] = {"main": okved_main, "extra": okved_extra}
    return okveds
# %%
if __name__ == '__main__':
    okveds_all = {}
    dir_ = f'{ROOT}/data-10012022-structure-10082021'
    files = list(Path(dir_).iterdir())
    with Pool(processes=8) as pool, tqdm(total=len(files)) as pbar:
        # there are many files in the archive
        # thus OKVED extraction can be effectively parallelized
        results = [pool.apply_async(get_okveds_from_file,
                                    args=(f, ),
                                    callback=lambda _: pbar.update(1))
                       for f in files]
        okveds = {}
        for r in results:
            okveds |= r.get()
    with open('../data/okved2/company_inn_okveds_rmsp.pickle', 'wb') as fp:
        pickle.dump(okveds, fp)
