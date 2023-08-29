import os
import re
import sys
import json
import jinja2
from glob import glob
from segment import process
from collections import defaultdict


def gen_html(path, tables):
    tpl = jinja2.Template(open("html/template.html").read())
    html = tpl.render(tables=tables)
    with open(path, "w") as output:
        output.write(html)

if __name__ == "__main__":
    for doc in glob("./tables/*"):
        doi = doc.split("/")[-1]
        tables = []
        for imgfile in sorted(glob(f"{doc}/*.full.png")):
            try:
                table, title, note, compounds = process(imgfile)
                tables.append({
                    "source": os.path.basename(imgfile.replace(".full.png", ".dbg.png")),
                    "title": title,
                    "content": table,
                    "note": note
                })
                print(imgfile, "done")
            except Exception as e:
                print(imgfile, e)
        gen_html(f"{doc}/index.html", tables)
