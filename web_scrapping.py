import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


url = "https://www.nobroker.in/blog/property-rates-in-hyderabad/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
rows = soup.find_all('tr')
land_data = []
for row in rows:
    cells = row.find_all('td')
    if len(cells) == 4:
        place_name = cells[0].get_text(strip=True)
        price_2021 = cells[1].get_text(strip=True)
        price_2020 = cells[2].get_text(strip=True)
        growth = cells[3].get_text(strip=True)
        land_data.append([place_name, price_2021, price_2020, growth])
df = pd.DataFrame(land_data, columns=['Place Name', 'Price 2021', 'Price 2020', 'Growth'])
df.to_csv('land_data.csv', index=False)