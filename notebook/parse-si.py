import os
import re
import pandas as pd
from pandas.errors import EmptyDataError
import rdkit.Chem as Chem
from glob import glob

def read_data(f, sep=None):
    _, ext = os.path.splitext(f)
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(f)
    elif ext in [".csv"]:
        lines = open(f, encoding='latin-1').read().splitlines()
        try:
            return pd.read_csv(f, encoding='latin-1', on_bad_lines='skip', sep=sep, engine='python')
        except EmptyDataError as e:
            pass
    return None

valid_sname_pat = re.compile('smile|struct', re.I)

files = list(glob(f"data/*.csv")) + list(glob("data/*.xls*"))

db = {}
for sp in sorted(files):
    fname, ext = os.path.splitext(os.path.basename(sp))
    doi = ".".join(fname.split(".")[:-1])

    compounds = db.get(doi, {})

    ss = [",", ";"]

    lines = open(sp, encoding='latin-1').read().splitlines()
    if len(lines) > 2:
        a, b = len(lines[2].split(";")), len(lines[1].split(";"))
        if a == b and a > 2:
            ss = [";", ","]

    for sep in ss:
        sdf = read_data(sp, sep)
        if sdf is None:
            continue
        cname = sdf.columns[0]
        sname = [c for c in sdf.columns if valid_sname_pat.findall(c.lower())]
        if len(sname) > 0:
            scol = sname[0]
            for x in sname:
                if x.lower().find("smi") >= 0:
                    scol = x
            sdf = sdf[[cname, scol]].dropna()
            for cid, smiles in zip(sdf[cname], sdf[scol]):
                if isinstance(smiles, float):
                    continue
                if isinstance(cid, float) or isinstance(cid, int):
                    cid = str(round(cid))
                cid = " ".join(cid.split())
                smiles = str(smiles).strip().strip(";").strip(",")
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    compounds[cid] = smiles
        if compounds:
            break

    if compounds:
        db[doi] = compounds


for doi, compounds in db.items():
    for cid, smiles in compounds.items():
        print(f"{doi}\t{cid}\t{smiles}")
