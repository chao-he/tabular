import os
import re
import pandas as pd
from glob import glob

def read_data(f):
    try:
        return pd.read_excel(f)
    except ValueError as e:
        pass

    for encoding in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(f, on_bad_lines='skip', encoding=encoding, sep='[,;\t]', engine='python')
        except UnicodeDecodeError:
            pass
        except ValueError:
            pass
    return None

valid_sname_pat = re.compile('smi|struct', re.I)

slist = []
for pdf in glob("./tables/*"):
    compounds = {}
    doi = os.path.basename(pdf)
    for si in sorted(glob(f"../dev/tabular/si/10.1021/{doi}*.csv")):
        try:
            df = read_data(si)
            cname = df.columns[0]
            for sname in df.columns:
                if valid_sname_pat.findall(sname):
                    for cid, smiles in zip(df[cname], df[sname]):
                        compounds[cid.strip()] = smiles.strip()
        except Exception as e:
            pass
    if compounds:
        with open(f"./tables/{doi}/compounds.txt", "w") as output:
            for cid, smiles in compounds.items():
                output.write(f"{cid}\t{smiles}\n")
    print(doi, len(compounds))
