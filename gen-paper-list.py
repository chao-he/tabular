import os
import jinja2
from glob import glob


meta = {}

for line in open("../dev/tabular/data/renova.meta.txt"):
    _, doi, mag, title = line.strip().split("\t")[:4]
    meta[doi] = title

data = []
for doc in glob("../papers/tables/*"):
    doi = os.path.basename(doc)
    title = meta.get(f"10.1021/{doi}", None)
    data.append([doi, title])

template = open("html/index.html").read()
tpl = jinja2.Template(template)
html = tpl.render(data=data)
output = open("index.html", "w")
output.write(html)
