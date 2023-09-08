import os, sys, json
import fitz
from glob import glob
from fitz import Rect


pdf = sys.argv[1]
name = os.path.basename(pdf)[:-4]
newpdf = fitz.open()

doc = fitz.open(pdf)

for page in doc:
    width, height = page.rect.width, page.rect.height
    outpage = newpdf.new_page(width=width, height=height)

    paths = page.get_drawings(extended=True)
    for path in paths:
        items = path.pop("items", [])

        fill = path.pop('fill', None)
        color = path.pop("color", None)
        rect = path.pop('rect', None)

        if not color or color == (1.0, 1.0, 1.0) or color == (0.0, 0.0, 0.0) or fill is None:
            continue

        for i, item in enumerate(items):
            shape = outpage.new_shape()
            if item[0] == "l":
                shape.draw_line(item[1], item[2])

            elif item[0] == "re":
                shape.draw_rect(item[1])
            elif item[0] == "qu":
                shape.draw_quad(item[1])
            elif item[0] == "c":
                shape.draw_bezier(item[1], item[2], item[3], item[4])
            else:
                raise ValueError("unhandled drawing", item)

            shape.finish(even_odd=True, color=color, fill=fill[i],
                     fill_opacity=1, stroke_opacity=1.0,
                     lineCap = 1.0, lineJoin = 1.0,
                     width=1.0)

            shape.commit()
        print(path['rect'], color, fill, len(items))
newpdf.save(sys.argv[2])
