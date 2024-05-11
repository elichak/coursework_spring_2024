# %%
import datetime
import requests
from time import sleep
from random import randint
# %%
BASE_URL = 'https://egrul.nalog.ru/'
def download_statement_for_inn(inn: str, save_dir: str):
    """Downloads EGRUL statement for company found by search with specified INN

    Args:
        inn (str): company's INN
        save_dir (str): file to save the statement to
    """
    # first request to get cookie
    request_for_cookie = requests.get(BASE_URL)
    cookie = request_for_cookie.headers['Set-Cookie'].split(';')[0]
    # then request to get list of companies
    payload = {
        'vyp3CaptchaToken': '',
        'page': '',
        'query': inn,
        'nameEq': 'on',
        'PreventChromeAutocomplete':'',
    }
    headers = {'Cookie':cookie}
    request_for_token = requests.post(BASE_URL, data=payload, headers=headers)
    token_json = request_for_token.json()
    if 't' not in token_json:
        print(f"Skipping {inn}: token failed")
        return
    token = token_json['t']
    # then request to get result info
    time = int(datetime.datetime.now().timestamp())
    search_request = requests.get(f'{BASE_URL}search-result/{token}?r={time}&_={time}',
                                    headers=headers)


    search_json = search_request.json()
    if 'rows' not in search_json or not search_json['rows']:
        print(f"Skipping {inn}: search failed")
        return
    download_token = search_json['rows'][0]['t']
    # then request to start the statement creation process
    requests.get(f'{BASE_URL}vyp-request/{download_token}', headers=headers)
    sleep(1)
    # finally, download the statement as pdf
    pdf_request = requests.get(f'{BASE_URL}vyp-download/{download_token}',
                                headers={'Cookie': cookie,
                                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'})
    if pdf_request.status_code != 200:
        print(f"{inn}: {pdf_request.status_code} {pdf_request.text}")
    else:
        with open(f'{save_dir}/{inn}.pdf', 'wb') as fp:
            fp.write(pdf_request.content)

# %%
if __name__ == '__main__':
    inn = '7714086422'
    download_statement_for_inn(inn, '../../data/pdf')
    sleep(randint(15, 25))

# %%
