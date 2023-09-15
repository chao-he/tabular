import os
import requests
import pandas as pd
from glob import glob
from multiprocessing import Pool


def fetch_sp(doi):
    url = f"https://widgets.figshare.com/public/files?institution=acs&limit=21&offset=0&collectionResourceDOI={doi}"
    r = requests.get(url)
    assert r.status_code == 200
    print(f"download {doi} -> {r.status_code}")
    for f in r.json()["files"]:
        fname, durl = f["name"], f["downloadUrl"]
        _, ext = os.path.splitext(fname)
        if ext in ['.xls', '.xlsx', '.zip', '.csv'] and not os.path.exists(f"data/{doi}.{fname}"):
            r = requests.get(durl)
            assert r.status_code == 200
            with open(f"data/{doi}.{fname}", "wb") as output:
                output.write(r.content)
            print(doi, fname, durl, ext)


if __name__ == "__main__":
    df = pd.read_csv("./jmc-article-stat.txt", sep='\t', header=None,
                     names=["doi", "y", "h", "w", "subtype", "venue"])

    cache = set()
    for f in glob("data/10.1021/*"):
        doi, ext = os.path.splitext(os.path.basename(f))
        doi = ".".join(doi.split(".")[:-1])
        cache.add(doi)

    seeds = []
    for _,row in df.iterrows():
        if doi in cache:
            continue
        doi = "10.1021/" + row["doi"]
        seeds.append(doi)

    print(len(df), len(seeds), len(cache))
    with Pool(10) as p:
        p.map(fetch_sp, seeds)
