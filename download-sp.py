import os
import re
import pandas as pd
import requests
from glob import glob

def read_csv(csv_file):
    try:
        return pd.read_excel(csv_file)
    except ValueError as e:
        pass

    for encoding in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(csv_file, on_bad_lines='skip', encoding=encoding, sep='[,;\t]', engine='python')
        except UnicodeDecodeError:
            pass
        except ValueError:
            pass
    return None


def convert(source, target):
    df = read_csv(source)
    if df is not None:
        df.to_csv(target)


def fetch_sp(doi):
    url = 'https://widgets.figshare.com/public/files?institution=acs&limit=21&offset=0&collectionResourceDOI=' + doi
    r = requests.get(url)

    files = {}
    if r.status_code == 200:
        for f in r.json()["files"]:
            name, durl = f["name"], f["downloadUrl"]
            if name.endswith('.csv'):
                r = requests.get(url)
                if r.status_code == 200:
                    files[name] = r.content
    return files


if __name__ == "__main__":
    df = pd.read_csv("jmc-article.txt", sep='\t', header=None,
                     names=["doi", "y", "h", "w", "subtype", "venue"])
    for doi in df.doi:
        try:
            files = fetch_sp(f"10.1021/{doi}")
            print(doi, len(files))
        except Exception as e:
            print(e)
