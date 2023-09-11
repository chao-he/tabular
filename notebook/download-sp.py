import os
import pandas as pd
import requests
from glob import glob
from multiprocessing import Pool

def fetch_sp(doi):
    baseurl = 'https://widgets.figshare.com/public/files?institution=acs&limit=21&offset=0&collectionResourceDOI='
    r = requests.get(f"{baseurl}10.1021/{doi}")
    assert r.status_code == 200
    for f in r.json()["files"]:
        name, durl = f["name"], f["downloadUrl"]
        _, ext = os.path.splitext(name)
        if ext in ['.pdf', '.pdb'] or os.path.exists(f"data/{doi}.{name}"):
            continue
        r = requests.get(durl)
        assert r.status_code == 200
        with open(f"data/{doi}.{name}", "wb") as output:
            output.write(r.content)
        print(doi, name, durl, ext)


df = pd.read_csv("jmc-article.txt", sep='\t', header=None, names=["doi", "y", "h", "w", "subtype", "venue"])
for doi in df.doi:
    if not list(glob(f"./data/{doi}*.csv")):
        print(doi)
        fetch_sp(doi)

