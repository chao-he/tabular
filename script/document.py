import re
import json
import math
from collections import defaultdict
from fitz import Rect


def wrap_text(text):
    wrap = ""
    for line in text.splitlines():
        line = line.strip()
        if line.endswith(u'-'):
            line = line[:-1]
        else:
            line += ' '
        wrap += line
    return wrap


class TableBuilder:
    def __init__(self):
        self.tables = []
        self.current_name = None
        self.current_table = []
        self.hh = 0

    def start(self, bbox, text):
        if self.current_name:
            self.finish()
        self.current_name = ''.join(text.splitlines())
        self.current_table = [(bbox, text)]
        self.is_right_page = 0 if bbox.x0 < 300 else 1
        self.max_y = bbox.y1
        self.hh = min(bbox.width, bbox.height)

    def update(self, bbox, text):
        if self.is_right_page and bbox.x0 < 300:
            return
        self.current_table.append((bbox, text))
        self.max_y = max(self.max_y, bbox.y1)

    def finish(self):
        if not self.current_name:
            return
        rect, title = self.current_table[0]
        for ix, (bbox, text) in enumerate(self.current_table):
            if bbox.y0 < rect.y0:
                continue
            rect.include_rect(bbox)
        self.tables.append((self.current_name, rect))
        self.current_name = None
        self.current_table = None


class DocSplitter:
    def __init__(self, c_w=4, c_h=11.5):
        self.fig_pat = re.compile("^(Table|Figure|Scheme|Fig.) ", re.IGNORECASE)
        self.c_w = c_w
        self.c_h = c_h

    def __call__(self, doc):
        clips = []
        for pn, page in enumerate(doc):
            blocks = self.split_page(page, pn, self.c_w, self.c_h)
            blocks = list(sorted(blocks, key=lambda x: self.gen_key(x[0], page.rect.width/2)))
            clips.append(blocks)
        return clips

    def gen_key(self, rect, half_width):
        return int(rect.x0/half_width)*1000 + rect.y0

    def split_page(self, page, pn, c_w, c_h):
        blocks = []
        for blk in page.get_text("dict")["blocks"]:
            bbox, text_type, text = self.parse_block(blk, pn, c_w, c_h)
            blocks.append((bbox, text_type, text))

        paths = page.get_drawings(extended=True)
        for path in paths:
            items = path.pop("items", [])
            fill = path.pop('fill', None)
            color = path.pop("color", None)
            rect = path.pop('rect', None)

            if rect is None or rect.width < 220 or 70 < rect.x0 < 300:
                continue

            if color is None and fill is None:
                continue
            r,g,b = color if color else fill
            if r < 0.5 or (abs(r-g) < 0.01 and abs(r-b) < 0.01):
                continue
            blocks.append((rect, "break", ""))

        #for draw in page.get_drawings(extended=True):
        for draw in page.get_drawings():
            # kind, level = draw["type"], draw['level']
            bbox = draw.get('rect', draw.get('scissor', None))
            area = bbox.get_area()
            x, y, w, h = map(round, [
                bbox.x0, bbox.y0, bbox.width, bbox.height
            ])
            if area < 400 or h < 20 or w < 4:
                continue
            if area > page.rect.get_area() * 0.55:
                continue
            blocks.append((bbox, "draw", None))
        return blocks

    def parse_line(self, line, pn, c_w, c_h):
        return "".join([span["text"] for span in line["spans"]])

    def parse_block(self, blk, pn, c_w, c_h):
        bbox = Rect(blk["bbox"])
        if blk["type"] == 1:
            return bbox, "img", None

        text, text_area, text_type = [], 0, "unk"
        for line in blk["lines"]:
            s = self.parse_line(line, pn, c_w, c_h)
            text_area += c_w * c_h * len(s)
            text.append(s)

        text, nlines = "\n".join(text), len(blk["lines"])
        text_ratio = text_area / bbox.get_area()
        if pn == 0 and text.startswith('ABSTRACT:'):
            text_type = "normal"
        elif self.fig_pat.match(text):
            c = {"Fig.": "Figure", "TABLE": "Table"}
            text_type = text.split()[0]
            text_type = c.get(text_type, text_type)
        elif text.startswith(u'■'):
            text_type = "normal"
        elif 0.8 < text_ratio < 1.05:
            if nlines > 2 * bbox.height / c_h:
                text_type = "column"
            else:
                text_type = "normal"
        elif text_ratio > 1 and nlines >= 5:
            text_type = "normal"
        else:
            w, h = 0, 0
            for line in blk["lines"]:
                b = Rect(line["bbox"])
                w += b.width
                h += b.height
            if h / bbox.height > 1.2:
                text_type = "column"
        return bbox, text_type, text


def extract_blocks(doc):
    return DocSplitter()(doc)


def find_title(doc):
    title = doc.metadata["title"]
    if len(title.split()) > 3:
        return title
    page, block_size = doc[0], []
    for blk in page.get_text("dict")["blocks"]:
        if blk["type"] == 0:
            size = round(blk["lines"][0]["spans"][0]["size"])
        block_size.append(0 if blk["type"] == 1 else size)
    max_size, text_map = 0, defaultdict(list)
    for size, blk in zip(block_size, page.get_text("blocks")):
        if 12 < size < 24:
            text = blk[4].replace('\n', ' ').replace('- ', '').strip()
            text_map[size].append(text)
            max_size = max(max_size, size)
    return " ".join(text_map[max_size]).strip()


def parse_doc(doc):
    title, abstract, content = find_title(doc), "", []
    tables, schemes, figures, images = [], [], [], []

    cropbox = (doc[0].rect - 40).intersect(doc[0].rect + 40)
    for blk in doc[0].get_text('blocks'):
        if blk[-1] == 0 and blk[4].startswith("ABSTRACT:"):
            abstract = wrap_text(blk[4])
            break

    clips = extract_blocks(doc)
    for pn, blocks in enumerate(clips):
        figure_set = set()
        for idx, (bbox, tag, text) in enumerate(blocks):
            if tag in ("img", "draw"):
                if tag == "img":
                    images.append(["", pn, bbox])
                if idx+1 < len(blocks) and blocks[idx+1][1] == "Figure":
                    bbox.include_rect(blocks[idx+1][0])
                    figures.append([blocks[idx+1][2], pn, bbox])
                    figure_set.add(idx)
        tb = TableBuilder()
        for idx, (bbox, tag, text) in enumerate(blocks):
            if tag in ("Table"):
                if bbox.x1*2 > doc[pn].rect.width:
                    bbox.x1 = doc[pn].rect.x1 - 20
                tb.start(bbox, text)
            elif tb.current_name:
                if len(tb.current_table) <= 1:
                    tb.update(bbox, text)
                elif tag in ("img", "draw") and idx not in figure_set:
                    tb.update(bbox, text)
                elif tag in ("column", "unk"):
                    tb.update(bbox, text)
                elif tag == "break":
                    tb.update(bbox, text)
                    tb.finish()
                elif abs(tb.max_y - bbox.y1) < 50:
                    tb.update(bbox, text)
                #else:
                #    tb.finish()
        tb.finish()

        for name, bbox in tb.tables:
            if name.startswith("Table"):
                tables.append([name, pn, bbox, tb.hh])
            else:
                schemes.append([name, pn, bbox, tb.hh])

    for pn, blocks in enumerate(clips):
        for bbox, tag, text in blocks:
            if not text or bbox.height <= 10 or not cropbox.contains(bbox):
                continue
            text = text.strip()
            if text and tag in ("normal", "unk"):
                content.append(text.replace('\n', ' '))
    content = " ".join(content)
    return title, abstract, content, tables, figures, schemes, images


class TextBuffer:
    def __init__(self, w, h=10):
        self.page_width = w/2
        self.line_height = h
        self.line_width = 90
        self.buffer = []

    def draw(self, x, y, text):
        row = int(math.floor(y / self.line_height))
        col = int(math.floor(x / self.page_width))
        d = max(0, row + 1 - len(self.buffer))
        for i in range(d):
            self.buffer.append([[], []])
        self.buffer[row][col].append([x, text])

    def render(self, line, cw=4):
        offset, text = self.page_width, []
        for x, t in line:
            offset = min(offset, x)
            text.append(t)
        if offset >= self.page_width:
            offset -= self.page_width
        return ' ' * int(round(offset/cw)) + ' '.join(text)

    def show(self, output, trim=6):
        for left, right in self.buffer:
            left, right = self.render(left), self.render(right)
            padsize = self.line_width - len(left)
            if padsize > 0:
                left += ' ' * padsize
            left = left[trim:self.line_width] + right
            output.write(left)
            output.write("\n")


def pdf2txt(doc, output):
    for page in doc:
        tb = TextBuffer(page.rect.width, int(page.rect.height/10))
        for blk in page.get_text("dict")["blocks"]:
            if blk["type"] == 1:
                continue
            for line in blk["lines"]:
                for span in line["spans"]:
                    r = Rect(span["bbox"]).round()
                    font, size = span["font"], round(span["size"])
                    text = span["text"].strip()
                    tb.draw(r.x0, r.y0 + size/2, text)
        output.write("-"*150)
        output.write("\n")
        tb.show(output)
