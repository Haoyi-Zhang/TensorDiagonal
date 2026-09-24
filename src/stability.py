"""Rational posterior certificates conditional on a declared observation model.

A certificate bounds ALL compatible real odeco tensors. It does not assert
that the approximate observation has a compatible tensor. No floating point
is used to produce the bound or the inverse witnesses.
"""
from __future__ import annotations
from fractions import Fraction as Q
from itertools import combinations
from math import isqrt,comb
from exact import equations,to_orbits

def supports(n,s,maximum=2048):
    if type(s)is not int or not 0<=s<=n:raise ValueError('invalid corruption budget')
    if sum(comb(n,k)for k in range(s+1))>maximum:raise ValueError('support enumeration exceeds declared implementation budget')
    return [tuple(c)for k in range(s+1)for c in combinations(range(n),k)]

def inverse(a):
    n=len(a);b=[list(row)+[Q(i==j)for j in range(n)]for i,row in enumerate(a)]
    for i in range(n):
        k=next((k for k in range(i,n)if b[k][i]),None)
        if k is None:return None
        b[i],b[k]=b[k],b[i];p=b[i][i];b[i]=[v/p for v in b[i]]
        for k in range(n):
            if k==i:continue
            p=b[k][i]
            if p:b[k]=[v-p*w for v,w in zip(b[k],b[i])]
    return [row[n:]for row in b]

def sqrt_bound(a,upper=True,bits=40):
    a=Q(a)
    if a<0:raise ValueError('negative square')
    scale=1<<bits;k=isqrt((a.numerator*scale*scale)//a.denominator)
    if upper and Q(k*k,scale*scale)<a:k+=1
    return Q(k,scale)

def dot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))

def normal_system(h):
    rows=[(row,rhs)for _,row,rhs in equations(h)];n=len(h)
    g=[[sum((row[i]*row[j]for row,_ in rows),Q(0))for j in range(n)]for i in range(n)]
    v=[sum((row[i]*rhs for row,rhs in rows),Q(0))for i in range(n)]
    return rows,g,v

def trimmed_candidate(h,observed_diagonal,s):
    """Exhaustive small-support least squares; not a scalable robust solver."""
    d=list(map(Q,observed_diagonal));n=len(h)
    if not 1<=n<=16:raise ValueError('dimension outside [1,16]')
    if len(d)!=n:raise ValueError('diagonal length')
    rows,g,v=normal_system(h);best=None
    for support in supports(n,s):
        bad=set(support);m=[[g[i][j]+Q(i==j and i not in bad)for j in range(n)]for i in range(n)]
        inv=inverse(m)
        if inv is None:continue
        rhs=[v[i]+(d[i]if i not in bad else 0)for i in range(n)]
        x=[dot(row,rhs)for row in inv]
        score=sum(((dot(row,x)-rhs)**2 for row,rhs in rows),Q(0))+sum(((x[i]-d[i])**2 for i in range(n)if i not in bad),Q(0))
        if best is None or score<best[0]:best=(score,support,x)
    if best is None:return None
    return {'objective':str(best[0]),'support':list(best[1]),'diagonal':[str(v)for v in best[2]]}

def certify(h,d,x,candidate_support,s,delta,epsilon):
    n=len(h);d=list(map(Q,d));x=list(map(Q,x));delta=Q(delta);epsilon=Q(epsilon)
    if not 1<=n<=16:raise ValueError('dimension outside [1,16]')
    shat=sorted(candidate_support)
    if len(d)!=n or len(x)!=n or delta<0 or epsilon<0:raise ValueError('invalid dimensions or noise radii')
    if len(set(shat))!=len(shat)or len(shat)>s or any(type(i)is not int or not 0<=i<n for i in shat):raise ValueError('invalid candidate support')
    rows,g,_=normal_system(h);inverse_witnesses=[];qmin=None
    for truth in supports(n,s):
        union=set(truth)|set(shat)
        m=[[g[i][j]+Q(i==j and i not in union)for j in range(n)]for i in range(n)]
        inv=inverse(m)
        if inv is None:return {'status':'no_certificate','reason':'singular_restricted_system'}
        trace=sum((inv[i][i]for i in range(n)),Q(0))
        q=1/trace
        qmin=q if qmin is None else min(qmin,q)
        inverse_witnesses.append({'truth_support':list(truth),'inverse':[[str(a)for a in row]for row in inv]})
    gamma=sqrt_bound(qmin,False)
    if gamma<=delta:return {'status':'no_certificate','reason':'noise_exceeds_certified_margin'}
    normx=sqrt_bound(dot(x,x));normh=sqrt_bound(sum((v*v for sl in h for row in sl for v in row),Q(0)))
    residual=sqrt_bound(sum(((dot(row,x)-rhs)**2 for row,rhs in rows),Q(0)))
    diagres=sqrt_bound(sum(((d[i]-x[i])**2 for i in range(n)if i not in shat),Q(0)))
    a=residual+delta*(normx+2*normh)+delta*delta;b=diagres+epsilon
    radius=sqrt_bound(a*a+b*b)/(gamma-delta)
    total=sqrt_bound(delta*delta+radius*radius)
    return {'status':'conditional_bound','n':n,'mixed_orbits':to_orbits(h),'observed_diagonal':[str(a)for a in d],
            'candidate_diagonal':[str(a)for a in x],'candidate_support':shat,'corruption_budget':s,
            'delta':str(delta),'epsilon':str(epsilon),'gamma':str(gamma),
            'norm_diagonal_upper':str(normx),'norm_mixed_upper':str(normh),'commutator_residual_upper':str(residual),
            'clean_diagonal_residual_upper':str(diagres),'diagonal_radius':str(radius),'tensor_radius':str(total),
            'inverse_witnesses':inverse_witnesses}
