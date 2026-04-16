### This script aims to grab names from different websites depending on region, right now I am testing with common arabic / middle eastern names as it makes for a good test case. This script might have to change depending on which website is used


### In order to keep it simple, this website has a lot of names from different origins, and works with the current script: https://www.thebump.com/b/baby-name-origins
from bs4 import BeautifulSoup
import csv

# with open("page.html", "r", encoding="utf-8") as f:
#     soup = BeautifulSoup(f, "html.parser")
#
# cards = soup.find_all("li", class_="card-wrap")
#
# names = []
# for card in cards:
#     link = card.find("a", class_="name-card-header")
#     if link:
#         name = link.find(text=True, recursive=False).strip()
#         names.append(name)
#
# print(f"Found {len(names)} names")
#
# with open("names.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Name"])
#     writer.writerows([[n] for n in names])

### This code below was to grab arabic last names from a different site. Might not be needed again

from bs4 import BeautifulSoup
import csv

with open("page.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

div = soup.find("div", class_="has-global-padding")
names = [h3.text.strip() for h3 in div.find_all("h3")]

print(f"Found {len(names)} names")

with open("names.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Name"])
    writer.writerows([[n] for n in names])


### THE FOLLOWING CODE WAS USED TO GRAB THE GENDER SPECIFIC NAMES ###

import requests
import pandas as pd

def scrape_names(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    names_list_element = soup.find('ol')

    top_1000_names = [li.text.strip() for li in names_list_element.find_all('li')]

    top_1000_names = re.findall(r'[A-Z][a-z]+',top_1000_names[0])

    return top_1000_names

top_1000_female_names = scrape_names("https://www.whattoexpect.com/baby-names/list/top-baby-names-for-girls/")
top_1000_male_names = scrape_names("https://www.whattoexpect.com/baby-names/list/top-baby-names-for-boys/")

df_female = pd.DataFrame(top_1000_female_names,columns=["Female Names"])
df_male = pd.DataFrame(top_1000_male_names,columns=["Male Names"])

df_female.to_csv("../data/gender_names/top_female_names.csv",index=False)
df_male.to_csv("../data/gender_names/top_male_names.csv",index=False)
