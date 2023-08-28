import os
import re
import sys
import json
import jinja2
from glob import glob
from segment import process
from collections import defaultdict


tpl = jinja2.Template(open("template.html").read())

def normalize_column(t):
    return t.replace('IC5o', 'IC50')

if __name__ == "__main__":
    for doc in glob("./tables/*3c00133"):
        compounds = defaultdict(dict)
        doi = doc.split("/")[-1]
        for line in open(f"./{doi}.csv"):
            cid, smiles = line.strip().split()[:2]
            compounds[cid]["smiles"] = smiles

        attributes = []

        tables = []
        for imgfile in sorted(glob(f"{doc}/*.full.png")):
            result = process(imgfile, 10, False)
            if result is None:
                continue
            table, title, note, _ = result
            if not table:
                continue

            tables.append({"source": imgfile, "title": title, "content": table, "note": note})

            for i in range(len(table)):
                for j in range(len(table[i])):
                    table[i][j] = ';'.join(table[i][j])

            for attr in table[0]:
                attr = normalize_column(attr)
                if attr not in attributes:
                    attributes.append(attr)

            for i in range(1, len(table)):
                for j in range(len(table[i])):
                    cid, attr = table[i][0], normalize_column(table[0][j])
                    compounds[cid][attr] = table[i][j]


        attributes = [a for a in attributes if not re.match("^R[0-9']*$", a)]
        attributes.insert(1, "smiles")
        table = [attributes]
        for cid, attrs in compounds.items():
            if attrs is not None:
                attrs = [attrs.get(a, '') for a in attributes]
                attrs[1] = "<img width=\"100\" src=\"https://askcos.mit.edu/api/v2/draw/?smiles=" + attrs[1] + "\"></img>"
                table.append(attrs)

        tables.append({"source": '', "title": '', "content": table, "note": ''})

        html = tpl.render(tables=tables)
        with open(f"{doi}.html", "w") as output:
            output.write(html)
