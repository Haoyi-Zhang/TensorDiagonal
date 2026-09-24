"""Exact corruption-distance and pointwise minimum-disagreement decoding.

The completion must first have been checked for consistency. Parameters on
separate free gain components have disjoint coordinate supports.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction as Q
from exact import solve, kernel_graph

def decode_diagonal(h, observed_diagonal):
    n=len(h);d=list(map(Q,observed_diagonal))
    if len(d)!=n:raise ValueError('diagonal dimension mismatch')
    cert=solve(h)
    if cert['status']=='inconsistent':return {'status':'inconsistent','completion':cert}
    x=list(map(Q,cert['particular']));g=kernel_graph(h)
    zs=[list(map(Q,z))for z in g['null_basis']]
    used={i for z in zs for i,v in enumerate(z)if v}
    rigid_errors=[i for i in range(n)if i not in used and d[i]!=x[i]]
    components=[];minimum=len(rigid_errors);number=1;thresholds=[]
    for z in zs:
        inds=[i for i,v in enumerate(z)if v]
        counts=Counter((d[i]-x[i])/z[i] for i in inds)
        largest=max(counts.values());modes=sorted(a for a,v in counts.items()if v==largest)
        minimum+=len(inds)-largest;number*=len(modes);thresholds.append(largest)
        components.append({'vertices':inds,'direction':[str(v)for v in z],
                           'values':[[str(a),count]for a,count in sorted(counts.items())],
                           'modes':[str(a)for a in modes],'maximum_multiplicity':largest})
    estimate=x[:]
    for component in components:
        alpha=Q(component['modes'][0]);z=list(map(Q,component['direction']))
        estimate=[a+alpha*b for a,b in zip(estimate,z)]
    # Compact list-size generating polynomial, indexed by extra disagreement.
    polynomial=[1]
    for component in components:
        largest=component['maximum_multiplicity'];p=[0]*(largest+1)
        for _,mult in component['values']:p[largest-mult]+=1
        new=[0]*(len(polynomial)+len(p)-1)
        for i,a in enumerate(polynomial):
            for j,b in enumerate(p):new[i+j]+=a*b
        polynomial=new
    return {'status':'decoded','completion':cert,'components':components,
            'rigid_errors':rigid_errors,'minimum_errors':minimum,
            'minimum_list_size':number,'finite_list_polynomial':polynomial,
            'infinite_at_budget':minimum+min(thresholds) if thresholds else None,
            'distance':min((len(c['vertices'])for c in components),default=None),
            'estimate':[str(v)for v in estimate]}

def list_size(report,budget):
    if report['status']!='decoded':return 0
    if type(budget)is not int or budget<0:raise ValueError('nonnegative integer budget required')
    if budget<report['minimum_errors']:return 0
    threshold=report['infinite_at_budget']
    if threshold is not None and budget>=threshold:return 'infinite'
    return sum(report['finite_list_polynomial'][:budget-report['minimum_errors']+1])
