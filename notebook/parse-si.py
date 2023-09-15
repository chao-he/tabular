import os
import sys
import re
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from collections import defaultdict
from pandas.errors import EmptyDataError
from molvs import standardize_smiles
from glob import glob


def detect_line_sep(fn, enc='latin-1'):
    lines = open(fn, encoding=enc).read().splitlines()
    c = [np.random.randint(len(lines)) for i in range(min(8, len(lines)))]
    for ix in [0] + c:
        line = lines[ix].strip()
        a, b = line.count(","), line.count(";")
        if a == 0 or b == 0:
            break
    if a == 0 and b > 0:
        return ";"
    elif a > 0 and b == 0:
        return ","
    else:
        return "[,;]"


def like_smiles(s):
    smiles_pat = re.compile(r'([CHONPSFI]|Cl|Br|[1-9]|[=#@\/]|[\x28\x29\x5b\x5d]){5,}', re.I)
    if isinstance(s, str) and len(s) > 3:
        token = 0
        for c in s.upper():
            if c in ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'I', '=', '#']:
                token += 1
        if token > 8:
            return True

        m = smiles_pat.match(s)
        if m and m[0].count('C')>3:
            return True
    return False

def find_smiles_column(df):
    for word in ["smile", "simile", "structure", "molecular", "formula", "string"]:
        for j, column in enumerate(df.columns):
            if column.lower().find(word) >= 0:
                return j
    for i, row in df.iterrows():
        for j, s in enumerate(row):
            if like_smiles(s):
                return j
    return -1


def detect_csv_format(f):
    sep = detect_line_sep(f, 'latin-1')

    skip = 0
    for i, line in enumerate(open(f, encoding='latin-1')):
        line = line.lower()
        is_head = False
        for word in ["compound", "smiles"]:
            if line.find(word) >= 0:
                is_head = True
                break
        if is_head:
            skip = i

    df = pd.read_csv(f, encoding='latin-1', sep=sep, skiprows=skip, on_bad_lines='skip', engine='python')

    smiles_row_ix, smiles_col_ix = skip + 1, find_smiles_column(df)

    if smiles_col_ix >= 0:
        sname = df.columns[smiles_col_ix]
        for i, s in enumerate(df[sname]):
            if like_smiles(s):
                smiles_row_ix = skip + i + 1
                break

    content = open(f, encoding='latin-1').read().splitlines()
    columns = ['' for j in range(len(df.columns))]

    for i in range(smiles_row_ix):
        t = content[i].split(sep)
        for j in range(len(columns)):
            if j < len(t) and t[j]:
                columns[j] += " " + t[j]
    columns = [c.strip() for c in columns]
    return 'latin-1', sep, smiles_row_ix, smiles_col_ix, columns


def read_data(f):
    _, ext = os.path.splitext(f)

    magic = open(f, 'rb').read(4)
    if magic == b'PK\x03\x04' and ext == ".csv":
        ext = ".xlsx"

    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(f)

    elif ext == ".csv":
        try:
            enc, sep, smiles_row_ix, smiles_col_ix, columns = detect_csv_format(fn)
            # print(fn, smiles_row_ix, smiles_col_ix, columns)
            df = pd.read_csv(f, encoding=enc, sep=sep, skiprows=smiles_row_ix, on_bad_lines='skip', engine='python')
            df.columns = (columns + ['Unamed: {i}' for i in range(len(df.columns) - len(columns))])[:len(df.columns)]
            return df
        except pd.errors.EmptyDataError:
            return None


if __name__ == "__main__":
    root_dir = "../jmc-article-discovery-formal-table"
    doc_list = [os.path.basename(fn)[:-4] for fn in glob(f"{root_dir}/*.pdf")]
    for fn in glob("./data/10.1021/*.*"):
        df = read_data(fn)
        if df is None:
            continue
        ix = find_smiles_column(df)
        print(fn, df.columns[ix])
