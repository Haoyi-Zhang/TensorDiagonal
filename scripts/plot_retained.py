"""Redraw retained evaluation tables as vector PDF; this is not an experiment.

Adapted from paper/figures/draw_evaluation.py without changing geometry.
Explicit input/output directories isolate this optional route from the paper.
Font files and metrics default to TeX Live; no font binaries are redistributed. CSV identities are checked against the retained source snapshot before
drawing. No project runner, evaluator, compiler, or network request is invoked.
"""
from pathlib import Path
from decimal import Decimal
import csv
import io
import math
import subprocess
import argparse
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

ARTIFACT = Path(__file__).resolve().parents[1]
HERE = ARTIFACT / "figures"
INPUT = ARTIFACT / "results" / "plot-data"
AFM_DIR = None
PFB_DIR = None


def tex_file(name):
    directory = AFM_DIR if name.endswith(".afm") else PFB_DIR
    if directory is not None:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    result = subprocess.run(["kpsewhich", name], check=True,
                            capture_output=True, text=True)
    path = Path(result.stdout.strip())
    if not result.stdout.strip() or not path.is_file():
        raise FileNotFoundError(name)
    return str(path)


def register_fonts(family):
    stems = {
        "times": [("Body", "utmr8a"), ("Variable", "utmri8a")],
        "newtx": [("Body", "ztmr"), ("Variable", "ztmri")],
        "cm": [("Body", "cmr10"), ("Variable", "cmmi10")],
    }[family]
    for alias, stem in stems:
        face = pdfmetrics.EmbeddedType1Face(tex_file(stem + ".afm"),
                                          tex_file(stem + ".pfb"))
        pdfmetrics.registerTypeFace(face)
        pdfmetrics.registerFont(pdfmetrics.Font(alias, face.name, "WinAnsiEncoding"))


def read_rows(path):
    text = (INPUT / Path(path).name).read_text(encoding="utf-8-sig")
    def parse(raw):
        if path.endswith(".dat"):
            rows = [line.split() for line in raw.splitlines() if line.strip()]
            return [dict(zip(rows[0], row)) for row in rows[1:]]
        return list(csv.DictReader(io.StringIO(raw)))
    rows = parse(text)
    expected = parse(SNAPSHOTS[path])
    if rows != expected:
        raise ValueError("Retained CSV changed: " + path)
    return rows


def points(spec):
    rows = read_rows(spec["csv"])
    result = []
    for i, row in enumerate(rows):
        raw_x = row[spec["x"]]
        x = Decimal(i + 1) if spec.get("categorical") else Decimal(raw_x)
        if "domain" in spec and not spec["domain"][0] <= x <= spec["domain"][1]:
            continue
        if row[spec["y"]] == "":
            raise ValueError("Missing executed value must not become a point.")
        y = Decimal(row[spec["y"]])
        result.append((float(x), float(y), row))
    return result


class Plot:
    def __init__(self, cfg, canvas_instance=None):
        self.cfg = cfg
        w, h = cfg["size"]
        self.c = canvas_instance or canvas.Canvas(str(HERE / (cfg["name"] + ".pdf")),
                               pagesize=(w, h), invariant=True,
                               initialFontName="Body")
        self.fontsize = cfg.get("font", 9)
        self.c.setFont("Body", self.fontsize)
        self.left, self.bottom, self.width, self.height = cfg["box"]
        self.axes()

    def xy(self, x, y):
        def ratio(v, bounds, logarithmic):
            if logarithmic:
                v, bounds = math.log10(v), [math.log10(a) for a in bounds]
            return (v - bounds[0]) / (bounds[1] - bounds[0])
        return (self.left + self.width * ratio(x, self.cfg["xlim"], self.cfg.get("xlog", False)),
                self.bottom + self.height * ratio(y, self.cfg["ylim"], self.cfg.get("ylog", False)))

    def label(self, x, y, text, anchor="center", angle=0, font="Body"):
        c = self.c
        c.saveState()
        c.translate(x, y)
        c.rotate(angle)
        c.setFillGray(0)
        c.setFont(font, self.fontsize)
        {"center": c.drawCentredString, "right": c.drawRightString,
         "left": c.drawString}[anchor](0, 0, text)
        c.restoreState()

    def axes(self):
        c, cfg = self.c, self.cfg
        x0, y0 = cfg["xlim"][0], cfg["ylim"][0]
        c.setStrokeGray(.82)
        c.setLineWidth(.3)
        for x, label in cfg["xticks"]:
            px, _ = self.xy(x, y0)
            if cfg.get("grid", True):
                c.line(px, self.bottom, px, self.bottom + self.height)
            rotation = cfg.get("xrotation", 0)
            if isinstance(label, dict) and "power" in label:
                self.power_label(px, self.bottom - 14, label.get("base", "2"), label["power"])
            else:
                self.label(px, self.bottom - 12, label,
                           anchor="right" if rotation else "center", angle=rotation)
        for y, label in cfg["yticks"]:
            _, py = self.xy(x0, y)
            if cfg.get("grid", True):
                c.line(self.left, py, self.left + self.width, py)
            if cfg.get("hide_ylabels"):
                continue
            if isinstance(label, dict) and "power" in label:
                base = "10"
                exponent = str(label["power"])
                c.saveState()
                c.setFillGray(0)
                c.setFont("Body", self.fontsize)
                ew = pdfmetrics.stringWidth(exponent, "Body", self.fontsize * .72)
                bw = pdfmetrics.stringWidth(base, "Body", self.fontsize)
                xx = self.left - 6 - ew - bw
                yy = py - self.fontsize * .32
                c.drawString(xx, yy, base)
                c.setFont("Body", self.fontsize * .72)
                c.drawString(xx + bw, yy + 4, exponent)
                c.restoreState()
            else:
                self.label(self.left - 6, py - self.fontsize * .32, label, anchor="right")
        c.setStrokeGray(0)
        c.setLineWidth(.5)
        if cfg.get("left_axes"):
            c.line(self.left, self.bottom, self.left + self.width, self.bottom)
            c.line(self.left, self.bottom, self.left, self.bottom + self.height)
        else:
            c.rect(self.left, self.bottom, self.width, self.height)
        if cfg.get("xlabel"):
            self.label(self.left + self.width / 2, cfg.get("xlabel_y", 5), cfg["xlabel"])
        if cfg.get("xlabel_runs"):
            self.rich_label(self.left + self.width / 2, cfg.get("xlabel_y", 5), cfg["xlabel_runs"])
        if cfg.get("ylabel_lines"):
            for n, label in enumerate(cfg["ylabel_lines"]):
                self.label(6 + n * 11, self.bottom + self.height / 2, label, angle=90)
        if cfg.get("ylabel"):
            self.label(cfg.get("ylabel_x", 9), self.bottom + self.height / 2,
                       cfg["ylabel"], angle=90)

    def power_label(self, x, y, base, exponent):
        c = self.c
        c.saveState()
        bwidth = pdfmetrics.stringWidth(base, "Body", self.fontsize)
        esize = self.fontsize * .72
        ewidth = pdfmetrics.stringWidth(str(exponent), "Body", esize)
        xx = x - (bwidth + ewidth) / 2
        c.setFont("Body", self.fontsize)
        c.drawString(xx, y, base)
        c.setFont("Body", esize)
        c.drawString(xx + bwidth, y + 4, str(exponent))
        c.restoreState()

    def rich_label(self, x, y, runs):
        c = self.c
        width = sum(pdfmetrics.stringWidth(t, f, self.fontsize) for t, f in runs)
        cursor = x - width / 2
        for text, font in runs:
            c.setFont(font, self.fontsize)
            c.drawString(cursor, y, text)
            cursor += pdfmetrics.stringWidth(text, font, self.fontsize)
        c.setFont("Body", self.fontsize)

    def errorbars(self, values, low_key, high_key):
        c = self.c
        c.setDash()
        c.setStrokeGray(0)
        c.setLineWidth(.6)
        for x, y, row in values:
            low = Decimal(row[low_key])
            high = Decimal(row[high_key])
            if low < 0 or high < 0:
                raise ValueError("Negative retained quartile distance.")
            a, b = self.xy(x, y - float(low)), self.xy(x, y + float(high))
            c.line(*a, *b)
            for px, py in (a, b):
                c.line(px - 2.2, py, px + 2.2, py)

    def line(self, values, dash=(), gray=0, marker=None, markers_every=1, thickness=.8):
        c = self.c
        c.setStrokeGray(gray)
        c.setLineWidth(thickness)
        c.setDash(dash)
        p = c.beginPath()
        for i, (x, y, _) in enumerate(values):
            (p.moveTo if i == 0 else p.lineTo)(*self.xy(x, y))
        c.drawPath(p)
        c.setDash()
        if marker:
            for i, (x, y, _) in enumerate(values):
                if i % markers_every == 0:
                    self.mark(*self.xy(x, y), marker, gray)

    def mark(self, x, y, marker="circle", gray=0, size=1.9):
        c = self.c
        c.setDash()
        c.setLineWidth(.65)
        c.setStrokeGray(gray)
        c.setFillGray(1 if marker in ("open", "open_square", "open_triangle") else gray)
        if marker in ("circle", "open"):
            c.circle(x, y, size, fill=1)
        elif marker in ("square", "open_square"):
            c.rect(x-size, y-size, 2*size, 2*size, fill=1)
        elif marker in ("triangle", "open_triangle"):
            p = c.beginPath()
            p.moveTo(x, y+size)
            p.lineTo(x-size, y-size)
            p.lineTo(x+size, y-size)
            p.close()
            c.drawPath(p, fill=1)
        c.setFillGray(0)

    def reference(self, x=None, y=None):
        c = self.c
        c.setStrokeGray(.35)
        c.setLineWidth(.6)
        c.setDash(3, 2)
        if x is not None:
            px, _ = self.xy(x, self.cfg["ylim"][0])
            c.line(px, self.bottom, px, self.bottom + self.height)
        else:
            _, py = self.xy(self.cfg["xlim"][0], y)
            c.line(self.left, py, self.left + self.width, py)
        c.setDash()

    def bars(self, values, width, shift=0, gray=.3, annotation=None):
        c = self.c
        baseline = self.cfg["ylim"][0] if self.cfg.get("ylog") else 0
        for x, y, row in values:
            px, py = self.xy(x, y)
            _, bottom = self.xy(x, baseline)
            c.setStrokeGray(0)
            c.setFillGray(gray)
            c.setLineWidth(.45)
            c.rect(px+shift-width/2, bottom, width, py-bottom, fill=1)
            c.setFillGray(0)
            if annotation:
                label = row[annotation] if annotation != "_2dp" else format(y, ".2f")
                self.label(px+shift, py+5, label)

    def legend(self, items, x, y, columns=1, step=12):
        c = self.c
        widths = [30 + pdfmetrics.stringWidth(i["label"], "Body", self.fontsize) for i in items]
        column_width = max(widths) + 10
        for n, item in enumerate(items):
            xx, yy = x + (n % columns) * column_width, y - (n // columns) * step
            c.setStrokeGray(item.get("gray", 0))
            c.setLineWidth(item.get("thickness", .8))
            if item.get("bar"):
                c.setFillGray(item.get("gray", .3))
                c.rect(xx, yy-2, 16, 5, fill=1)
            else:
                c.setDash(item.get("dash", ()))
                c.line(xx, yy, xx+18, yy)
                c.setDash()
                if item.get("marker"):
                    self.mark(xx+9, yy, item["marker"], item.get("gray", 0))
            self.label(xx+24, yy-2.5, item["label"], anchor="left")
        c.setFillGray(0)

    def finish(self):
        self.c.showPage()
        self.c.save()


def render(cfg, canvas_instance=None, finish=True):
    plot = Plot(cfg, canvas_instance)
    for series in cfg.get("series", []):
        values = points(series)
        style = series.get("style", {})
        if series.get("kind") == "bar":
            plot.bars(values, **style)
        elif series.get("kind") == "scatter":
            for x, y, row in values:
                px, py = plot.xy(x, y)
                plot.mark(px, py, size=2.3)
                plot.label(px+5, py-2.5, row[series["label"]], anchor="left")
        elif series.get("kind") == "interval":
            for x, y, row in values:
                low, high = Decimal(row["lower"]), Decimal(row["upper"])
                a, b = plot.xy(float(low), y), plot.xy(float(high), y)
                plot.c.setStrokeGray(0)
                plot.c.setLineWidth(1.4)
                plot.c.line(*a, *b)
                for px, py in (a, b):
                    plot.c.line(px, py-3, px, py+3)
                plot.mark(*plot.xy(float(Decimal(row["observed"])), y))
        else:
            if "error_minus" in series:
                plot.errorbars(values, series["error_minus"], series["error_plus"])
            plot.line(values, **style)
    for ref in cfg.get("references", []):
        plot.reference(**ref)
    if "legend" in cfg:
        plot.legend(**cfg["legend"])
    if finish:
        plot.finish()
    return plot


FAMILY = "cm"
SNAPSHOTS = {
  "figures/noise-3.dat": "scale error radius\n0.0009765625 0.00178526815624 0.200290813829\n6.103515625e-05 0.000113996751547 0.0122820109006\n3.81469726562e-06 7.13425071832e-06 0.000766720295075\n",
  "figures/noise-5.dat": "scale error radius\n0.0009765625 0.00589434624071 0.594690098669\n6.103515625e-05 0.000364159963002 0.0359704377077\n3.81469726562e-06 2.27436805152e-05 0.00224362563858\n",
  "figures/noise-7.dat": "scale error radius\n0.0009765625 0.00615651351692 1.46797235622\n6.103515625e-05 0.000390147713862 0.0859547552368\n3.81469726562e-06 2.44056044977e-05 0.0053509985757\n"
}
CHARTS = [
  {
    "name": "noise-bounds",
    "size": [
      344,
      198
    ],
    "box": [
      42,
      36,
      89,
      141
    ],
    "xlim": [
      0.0000025,
      0.0014
    ],
    "ylim": [
      0.000003,
      3
    ],
    "xlog": True,
    "ylog": True,
    "font": 9.96,
    "xticks": [
      [
        0.00000381469726562,
        {
          "power": -18
        }
      ],
      [
        0.00006103515625,
        {
          "power": -14
        }
      ],
      [
        0.0009765625,
        {
          "power": -10
        }
      ]
    ],
    "yticks": [
      [
        0.00001,
        {
          "power": -5
        }
      ],
      [
        0.001,
        {
          "power": -3
        }
      ],
      [
        0.1,
        {
          "power": -1
        }
      ]
    ],
    "hide_ylabels": False,
    "ylabel": "Diagonal error or radius",
    "ylabel_x": 9,
    "series": [
      {
        "csv": "figures/noise-3.dat",
        "x": "scale",
        "y": "radius",
        "style": {
          "marker": "open"
        }
      },
      {
        "csv": "figures/noise-3.dat",
        "x": "scale",
        "y": "error",
        "style": {
          "marker": "open",
          "dash": [
            3,
            2
          ]
        }
      }
    ],
    "legend": {
      "items": [
        {
          "label": "Certified radius",
          "marker": "open"
        },
        {
          "label": "Actual error",
          "marker": "open",
          "dash": [
            3,
            2
          ]
        }
      ],
      "x": 99,
      "y": 188,
      "columns": 2
    }
  },
  {
    "name": "noise-bounds",
    "size": [
      344,
      198
    ],
    "box": [
      145,
      36,
      89,
      141
    ],
    "xlim": [
      0.0000025,
      0.0014
    ],
    "ylim": [
      0.000003,
      3
    ],
    "xlog": True,
    "ylog": True,
    "font": 9.96,
    "xticks": [
      [
        0.00000381469726562,
        {
          "power": -18
        }
      ],
      [
        0.00006103515625,
        {
          "power": -14
        }
      ],
      [
        0.0009765625,
        {
          "power": -10
        }
      ]
    ],
    "yticks": [
      [
        0.00001,
        {
          "power": -5
        }
      ],
      [
        0.001,
        {
          "power": -3
        }
      ],
      [
        0.1,
        {
          "power": -1
        }
      ]
    ],
    "hide_ylabels": True,
    "ylabel": "",
    "ylabel_x": 9,
    "series": [
      {
        "csv": "figures/noise-5.dat",
        "x": "scale",
        "y": "radius",
        "style": {
          "marker": "open_square"
        }
      },
      {
        "csv": "figures/noise-5.dat",
        "x": "scale",
        "y": "error",
        "style": {
          "marker": "open_square",
          "dash": [
            3,
            2
          ]
        }
      }
    ]
  },
  {
    "name": "noise-bounds",
    "size": [
      344,
      198
    ],
    "box": [
      248,
      36,
      89,
      141
    ],
    "xlim": [
      0.0000025,
      0.0014
    ],
    "ylim": [
      0.000003,
      3
    ],
    "xlog": True,
    "ylog": True,
    "font": 9.96,
    "xticks": [
      [
        0.00000381469726562,
        {
          "power": -18
        }
      ],
      [
        0.00006103515625,
        {
          "power": -14
        }
      ],
      [
        0.0009765625,
        {
          "power": -10
        }
      ]
    ],
    "yticks": [
      [
        0.00001,
        {
          "power": -5
        }
      ],
      [
        0.001,
        {
          "power": -3
        }
      ],
      [
        0.1,
        {
          "power": -1
        }
      ]
    ],
    "hide_ylabels": True,
    "ylabel": "",
    "ylabel_x": 9,
    "series": [
      {
        "csv": "figures/noise-7.dat",
        "x": "scale",
        "y": "radius",
        "style": {
          "marker": "open_triangle"
        }
      },
      {
        "csv": "figures/noise-7.dat",
        "x": "scale",
        "y": "error",
        "style": {
          "marker": "open_triangle",
          "dash": [
            3,
            2
          ]
        }
      }
    ]
  }
]
def main():
    global HERE, INPUT, AFM_DIR, PFB_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-afm-dir", type=Path)
    parser.add_argument("--font-pfb-dir", type=Path)
    args = parser.parse_args()
    INPUT = args.input_dir.resolve()
    HERE = args.output_dir.resolve()
    AFM_DIR = args.font_afm_dir
    PFB_DIR = args.font_pfb_dir
    for path in SNAPSHOTS:
        read_rows(path)
    protected = (ARTIFACT / "results").resolve()
    paper = (ARTIFACT.parent / "paper").resolve()
    if HERE == INPUT or HERE == protected or protected in HERE.parents or HERE == paper or paper in HERE.parents:
        raise ValueError("Output must be separate from input, retained results, and paper")
    HERE.mkdir(parents=True, exist_ok=True)
    register_fonts(FAMILY)
    c = canvas.Canvas(str(HERE / "noise-bounds.pdf"), pagesize=(344, 198),
                      invariant=True, initialFontName="Body")
    plots = [render(cfg, c, finish=False) for cfg in CHARTS]
    # The native manuscript caption identifies n=3,5,7 from left to right.
    plots[1].rich_label(189.5, 5, [
        ("Dyadic perturbation scale ", "Body"), ("2", "Body")])
    # Native math-italic exponent, placed as a superscript of the last base 2.
    runs = [("Dyadic perturbation scale ", "Body"), ("2", "Body")]
    total = sum(pdfmetrics.stringWidth(t, f, 9.96) for t, f in runs)
    end = 189.5 + total / 2
    c.setLineWidth(.45)
    c.line(end + .5, 11.5, end + 4, 11.5)
    c.setFont("Variable", 7.2)
    c.drawString(end + 4.4, 9, "p")
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
