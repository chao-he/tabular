import os
import re
import pandas as pd
from glob import glob

def read_data(f):
    for encoding in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(f, on_bad_lines='skip', encoding=encoding, sep='[,;\t]', engine='python')
        except UnicodeDecodeError:
            pass
        except ValueError:
            pass
    return None

valid_sname_pat = re.compile('smile|struct', re.I)


db = {}

for sp in sorted(glob(f"./notebook/data/acs.jmedchem.*.csv")):
    doi = os.path.basename(sp)[:-21]
    db[doi] = {}

for sp in sorted(glob(f"./notebook/data/acs.jmedchem.*.csv")):
    doi = os.path.basename(sp)[:-21]
    df = read_data(sp)
    cname = df.columns[0]
    sname = [c for c in df.columns if valid_sname_pat.findall(c)]
    if sname:
        print(cname)

    continue

    for cid, smiles in zip(df[cname], df[sname]):
        db[doi][cid.strip()] = smiles.strip()
