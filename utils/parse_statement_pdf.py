import tabula

def get_okved_from_pdf(path: str) -> list[str]:
    """Parses statement as PDF and tries to extract OKVED code by a set of euristics

    Args:
        path (str): path to the statement as PDF

    Returns:
        list[str]: list of OKVEDs extracted from the statement
    """
    tables = tabula.read_pdf(path, pages='all')
    codes = []
    for table in tables:
        mask = table.apply(lambda row: any('Код и наименование' in str(c) for c in row), raw=True, axis=1)
        if len(table[mask]):
            assert table.shape[-1] in {1, 3, 4}
            if table.shape[-1] == 1:
                codes_from_page = table[mask].iloc[:, 0].str.extract(r'деятельности\s([\d\.]+)\s').values.flatten().tolist()
            elif table.shape[-1] in {3, 4}:
                codes_from_page = table[mask].iloc[:, -1].str.extract(r'^([\d\.]+)\s').values.flatten().tolist()
            codes.extend(codes_from_page)
    return codes

# %%
if __name__ == '__main__':
    inn = '7714086422'
    print(get_okved_from_pdf("../../data/pdf/7714086422.pdf"))
