import os
import sys
import json
import requests
import traceback
from requests_toolbelt.multipart.encoder import MultipartEncoder
from glob import glob

url = 'http://106.12.69.57:8078/api/ocsr/upload'
headers = {
    "X-App-Key": "aDSGpMT9va",
    "X-App-Secret": "9XGM1ELlE5Pvmcfa",
}

for root in glob("./images/*"):
    for imgfile in glob(f"{root}/*.r.*.png"):
        molfile = imgfile.replace(".png", ".txt")
        if os.path.exists(molfile):
            continue
        try:
            img = open(imgfile, 'rb')
            multipart_encoder = MultipartEncoder(
                fields={
                    "file": ("test.png", img, "image/png")
                }
            )
            headers['Content-Type'] = multipart_encoder.content_type
            r = requests.post(url, data=multipart_encoder, headers=headers)
            with open(molfile, "w") as output:
                json.dump(r.json(), output)
            print(imgfile, r.json()["code"])
        except Exception:
            traceback.print_exc(file=sys.stderr)
