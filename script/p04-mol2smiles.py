import os
import re
import sys
import json
import requests
import traceback
import rdkit.Chem as Chem
from glob import glob
from collections import defaultdict
from requests_toolbelt.multipart.encoder import MultipartEncoder
from config import DOC_ROOT, MOL_ROOT


def osra(imgfile):
    url = 'http://106.12.69.57:8078/api/ocsr/upload'
    headers = {
        "X-App-Key": "aDSGpMT9va",
        "X-App-Secret": "9XGM1ELlE5Pvmcfa",
    }
    molfile = imgfile.replace(".png", ".txt")
    if not os.path.exists(molfile):
        try:
            molimg = open(imgfile, 'rb')
            multipart_encoder = MultipartEncoder(fields={"file": ("test.png", molimg, "image/png")})
            headers['Content-Type'] = multipart_encoder.content_type
            r = requests.post(url, data=multipart_encoder, headers=headers)
            if r.status_code == 200 and r.json()["code"] == 200:
                with open(molfile, "w") as output:
                    json.dump(r.json(), output)
                print(imgfile, r.json()["code"])
            return r.json()
        except Exception:
            traceback.print_exc(file=sys.stderr)
    else:
        return json.load(open(molfile))


def parse_smiles(smiles):
    sp_atom_pattern = r'\[.+?\]'
    sp_atoms = re.findall(sp_atom_pattern, smiles)
    smiles = re.sub(sp_atom_pattern, '[*]', smiles)
    return Chem.MolFromSmiles(smiles), sp_atoms


if __name__ == "__main__":
    fpat = sys.argv[1] if len(sys.argv) > 1 else '*'
    for root in glob(f"{DOC_ROOT}/{fpat}"):
        for jsonf in glob(f"{root}/*.table.*.json"):
            prefix,ext = os.path.splitext(os.path.basename(jsonf))
            n = 0
            data = json.load(open(jsonf))
            for k in ["scaffold", "rgroup"]:
                for i, ((x,y,w,h),_) in enumerate(data.get(k, [])):
                    molfile = f"{MOL_ROOT}/{prefix}.{k[0]}.{x}_{y}_{w}_{h}.txt"
                    mol = json.load(open(molfile))
                    if "data" in mol:
                        smiles = mol["data"]["smiles"]
                        data[k][i][1] = smiles
                        n += 1
            if n > 0:
                json.dump(data, open(jsonf, "w"))
            print(jsonf, '\t', n)
