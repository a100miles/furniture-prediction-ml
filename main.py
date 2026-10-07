import requests
import httpx
import pandas
import lxml
import seaborn
from bs4 import BeautifulSoup
from fastapi import FastAPI
from matplotlib.pyplot import plot as plt

url = "https://mebel.kz/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

response = requests.get(url, headers=headers)

print(response)