import requests
from bs4 import BeautifulSoup
x = requests.get('https://mebel.kz/sitemap.almaty.xml')
body = x.text
bs = BeautifulSoup(body, 'xml')
raw_urls = []
for url in bs.find_all('loc'):
    raw_urls.append(url.text)

category_urls = []
for url in raw_urls:
    if 'category' in url:
        category_urls.append(url)


def get_product_url(url:str, page:int=1):
    response = requests.get(url)

    tree = response.text
    soup = BeautifulSoup(tree, 'html.parser')
    unique = soup.find_all('div', class_= 'ProductCardMain-module__4dYtKq__wrapper')
    if len(unique) == 48:
        next_page = requests.get(url + '?page=' + str(page + 1))
        get_product_url(next_page.url, page + 1)
    print(len(unique))
    
get_product_url(category_urls[0])