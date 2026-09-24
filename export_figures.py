#!/usr/bin/env python3
"""Create publication plot inputs and the numerical table from saved fractions."""
from __future__ import annotations
import argparse, csv, math
from fractions import Fraction as Q
from pathlib import Path

def export(results: Path, out: Path) -> None:
    with (results/'noise.csv').open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['amplitude']=='1']
    if len(rows)!=9 or len({(r['n'],r['noise_exponent']) for r in rows})!=9:
        raise ValueError('expected the nine distinct declared noise configurations')
    out.mkdir(parents=True,exist_ok=True)
    for n in (3,5,7):
        lines=['scale error radius']
        for r in rows:
            if int(r['n'])==n:
                values=(2.**(-int(r['noise_exponent'])),math.sqrt(float(Q(r['error_squared']))),float(Q(r['diagonal_radius'])))
                lines.append(' '.join(f'{v:.12g}' for v in values))
        (out/f'noise-{n}.dat').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{rrrrrr}',r'\toprule',r'$n$ & $s$ & $p$ & $\gamma$ & Actual error & $R$\\',r'\midrule']
    for r in rows:
        lines.append(f"{r['n']} & {r['s']} & {r['noise_exponent']} & {float(Q(r['gamma'])):.3f} & {math.sqrt(float(Q(r['error_squared']))):.3e} & {float(Q(r['diagonal_radius'])):.3e} \\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'noise-table.tex').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--out',type=Path)
    a=p.parse_args(); export(a.results,a.out or a.results/'plot-data')
if __name__=='__main__':main()
