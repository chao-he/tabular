import os, sys, json
import re
from glob import glob
from collections import defaultdict
import rdkit.Chem as Chem


def parse_smiles(smiles):
    sp_atom_pattern = r'\[.+?\]'
    sp_atoms = re.findall(sp_atom_pattern, smiles)
    smiles = re.sub(sp_atom_pattern, '[*]', smiles)
    return Chem.MolFromSmiles(smiles), sp_atoms


if __name__ == "__main__":
    for root in glob("./images/*"):
        doi = os.path.basename(root)
        data = defaultdict(lambda:defaultdict(list))
        for molfile in glob(f"{root}/*.txt"):
            r = json.load(open(molfile))
            if "data" not in r:
                continue
            smiles = r["data"]["smiles"]
            molfile, _ = os.path.splitext(os.path.basename(molfile))
            tix, tpe, pos = molfile.split(".")[-3:]
            if tix == 'cont':
                tix = molfile.split(".")[-4] + ".cont"
            x, y, w, h = map(int, pos.split("_"))
            data[tix][tpe].append([x,y,w,h,smiles])
        if data:
            data = json.dumps(data)
            print(f"{doi}\t{data}")
