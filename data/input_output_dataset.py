import pandas as pd
from pathlib import Path
import numpy as np

STATS_ROOT = Path('../../data/stats')

if __name__ == '__main__':
    prod_stats = (pd.read_excel(STATS_ROOT / 'затраты_выпуск.xlsx',
                                skiprows=1,
                                sheet_name='Затраты_Выпуск_Плоская',
                                dtype=str)
                    .iloc[:, 1:]
                    .set_index('Коды')
                    )
    prod_stats.columns = [str(c).strip() for c in prod_stats.columns]
    # the matrix should have same cols and rows names
    assert np.all(prod_stats.columns == prod_stats.index)

    # for 2016 year OKVED and OKPD v1 are used
    okved_versions = pd.read_excel(STATS_ROOT / 'ОКВЭД2-ОКВЭД2007.xls', skiprows=1,
                                usecols=[0, 2],
                                names=['OKVED2', 'OKVED1'],
                                sheet_name='Лист1')
    old_to_new = okved_versions.dropna().set_index('OKVED1')['OKVED2'].to_dict()
    # first 4 symbols of the OKDP code match with the appropriate OKVED code
    # source file was preprocessed manually
    new_index_columns = prod_stats.columns.map(old_to_new)
    prod_stats.columns = new_index_columns
    prod_stats.index = new_index_columns
    prod_stats = prod_stats.loc[prod_stats.index.notna(), prod_stats.columns.notna()]
    # cast to float and normalize by row
    prod_stats_np = prod_stats.values.astype(float)
    prod_stats_np /= prod_stats_np.sum(axis=1).reshape(-1, 1)
    n = len(prod_stats)
    # for now, we use only the upper triangle of the matrix
    # so mask the lower triangle with -1
    prod_stats_np[np.tril_indices(n, k=-1)] = -1
    prod_stats_upper = (pd.DataFrame(prod_stats_np,
                                    columns=prod_stats.columns,
                                    index=prod_stats.index)
                           .dropna())
    okved_consumption = (prod_stats_upper.unstack()
                                        .reset_index()
                                        .rename(columns={"level_0": "okved_provider",
                                                         "level_1": "okved_consumer",
                                                         0: "normalized_consumption"}))
    # query to remove the lower triangle
    (okved_consumption.query('normalized_consumption >= 0')
                      .to_csv(STATS_ROOT / 'okved_consumption.csv', index=False))
