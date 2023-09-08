import re
import fitz
from glob import glob
from multiprocessing import Process

checklist1 = [
    (0, "Correction to"),
    (0, "Author Manuscript"),
    (0, "Just Accepted"),
    (1, "ACS Paragon Plus Environment"),
    (1, 'ACS Medicinal Chemistry Letters'),
    (1, 'Journal of Medicinal Chemistry'),
    (1, 'The Journal of Organic Chemistry'),
    (1, 'Journal of Chemical Information and Modeling'),
    (1, 'Journal of Natural Products'),
]

article_type = """
article
articles
brief article
brief articles
communications to the editor
drug annotation
editorial
featured article
innovations
letter
letters
microperspectives
note
notes
organic letters
patent highlight
perspective
viewpoint
""".splitlines()

#atype_pat = re.compile("^" + s + "$", re.I)
venue_pat = re.compile("(The\\s+)?Journal\\s+of\\s(.+)|ACS\\s+Medicinal\\s+Chemistry", re.I)

def parse_head_line(words):
    atype, venue = "-", "-"
    for s in words:
        if s.lower() in article_type:
            atype = s.capitalize()
        if venue_pat.findall(s):
            venue = s
    return atype, venue

def process(files, shard, np):
    output = open(f"stat.{shard}.txt", "w")
    for i in range(shard, len(files), np):
        pdf = files[i]
        try:
            doc = fitz.open(pdf)
            doi = pdf.split("/")[-1][:-4]

            page = doc[len(doc)//2]

            blocks = page.get_text('dict', sort=True)['blocks']

            r = blocks[0]["bbox"]
            t = page.get_text(clip=r)
            y, h, w = r[1], r[3]-r[1], r[2]-r[0]
            words = t.splitlines()

            atype, venue = parse_head_line(words)
            t = "\t".join(words)

            print(f'{doi}\t{y:.02f}\t{h:.02f}\t{w:.02f}\t{atype}\t{venue}', file=output)
        except Exception as e:
            print(f"{doi}\terror:{e}", file=output)


if __name__ == "__main__":
    files = glob("../papers/discovery/*.pdf")
    ps = []
    for i in range(20):
        p = Process(target=process, args=(files, i, 20))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
