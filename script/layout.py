import os
import sys
import json
import numpy as np


def normalize(text):
    return text.split()[0].lower().strip(".").strip("#")


def find_first_of(tags, pat):
    for ix, tag in tags:
        if tag in pat:
            return ix

def check_table_position(table):
    group_of_tags = [
        ("compound", "compounds"),
        ("entry", "structure", "fragment"),
        ("comp", "compd", "cmpd", "cpd"),
        ("compa", "compb", "compc", "compda", "compdb", "compdc", "compdd"),
        ("id", "no",  "ex", "#"),
        ("assay", "species")
    ]

    left, right = 0, 0
    for _, _, row in table:
        left, right = min(left, row[0][0]), max(right, row[-1][1])
    table_width = right - left

    title_last_row, table_first_row, table_last_row = 0, -1, -1

    th = table[0][1] - table[0][0]
    tw = max(table[0][2][0][1] - table[0][2][0][0], table_width-120)
    for i, (ymin,ymax,row) in enumerate(table):
        xmin, xmax, txt = row[0]
        w, h = xmax-xmin, ymax-ymin
        if xmin < left + 2 * th and w > tw*0.55:
            title_last_row = i
        else:
            break

    for i, (ymin,ymax,row) in enumerate(table):
        if i > title_last_row and len(row) > 2:
            table_first_row = i

    if table_first_row == -1:
        for ymin,ymax,row in table:
            if i > title_last_row and len(row) > 1:
                table_first_row = i

    first_column = []
    for i, (ymin,ymax,row) in enumerate(table):
        if i > title_last_row and len(row) > 1:
            first_column.append([i, normalize(row[0][2])])

    for tags in group_of_tags:
        row = find_first_of(first_column, tags)
        if row is not None:
            table_first_row = row
            break

    table_last_row = table_first_row
    nr_col = len(table[table_first_row][2])
    for i, (ymin,ymax,row) in enumerate(table):
        if i > table_first_row:
            xmin, xmax, txt = row[0]
            if len(row) < 3 and xmin-xmax > 3 * table_width/nr_col:
                break
            if xmax-xmin > table_width*0.5 or len(txt.split()) > 8:
                break
            table_last_row = i

    return title_last_row, table_first_row, table_last_row + 1


def nonzero(flags):
    i, e = 0, len(flags)
    while i < len(flags):
        while i < e and flags[i] == 0:
            i += 1
        s = i
        while i < e and flags[i] == 1:
            i += 1
        if s < i:
            yield s,i

def aligment(boxes):
    rows, width, height = [], 0, 0
    for (x,y,w,h),t in boxes:
        width = max(width, x + w)
        height = max(height, y + h)

    flags = np.zeros(height)
    for (x,y,w,h),t in boxes:
        flags[y:y+h] = 1

    for ymin, ymax in nonzero(flags):
        cells, mask = [], np.zeros(width)
        for (x,y,w,h),t in boxes:
            if y>=ymin and y+h<=ymax:
                mask[x:x+w] = 1
                cells.append([x,y,w,h,t])
        columns = []
        for xmin, xmax in nonzero(mask):
            column = [(y, t) for x,y,w,h,t in cells if x>=xmin and x+w<=xmax]
            text = " ".join([t for y,t in sorted(column)])
            columns.append([xmin, xmax, text])
        rows.append([ymin, ymax, columns])
    return rows


def read_layout(fpath):
    if isinstance(fpath, list):
        data = fpath
    else:
        data = json.load(open(fpath))["text"]
    layout = aligment(data)
    title_ix, table_ix, note_ix = check_table_position(layout)
    t_boxes = []
    for y0, y1, row in layout[table_ix:note_ix]:
        for x0, x1, txt in row:
            t_boxes.append([x0, y0, x1-x0, y1-y0, txt])
    return layout, t_boxes, title_ix, table_ix, note_ix


if __name__ == "__main__":
    import sys
    layout, tboxes, tix, cid, nix = read_layout(sys.argv[1])
    for _,_, row in layout:
        print([t for _,_,t in row])

