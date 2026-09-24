"""Independent verifier: direct slice products, not the producer's equation builder.

No imports from exact.py. Acceptance certifies exact rational algebra only;
real orthogonal decomposability follows from the accompanying mathematical proof.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, permutations

F=Fraction

def _q(value):
    if isinstance(value,bool) or not isinstance(value,(int,str)) or len(str(value))>4096:
        raise ValueError('invalid rational encoding')
    return F(value)

def decode(instance):
    n=instance['n']
    if type(n) is not int or not 1<=n<=16:raise ValueError('dimension outside [1,16]')
    vals=instance['orbits'];keys=list(combinations_with_replacement(range(n),3))
    if len(vals)!=len(keys):raise ValueError('wrong orbit length')
    t=[[[F(0) for _ in range(n)]for _ in range(n)]for _ in range(n)]
    for key,v in zip(keys,vals):
        for i,j,k in set(permutations(key)):t[i][j][k]=_q(v)
    for i in range(n):t[i][i][i]=F(0)
    return t

def _comm(t,key):
    i,j,p,q=key;n=len(t)
    return sum((t[i][p][a]*t[j][a][q]-t[j][p][a]*t[i][a][q]for a in range(n)),F(0))

def _row(h,key):
    base=_comm(h,key);row=[]
    for a in range(len(h)):
        h[a][a][a]+=1
        row.append(_comm(h,key)-base)
        h[a][a][a]-=1
    return row,-base

def _rank(matrix):
    a=[list(row)for row in matrix]
    if not a:return 0
    r=0
    for c in range(len(a[0])):
        k=next((k for k in range(r,len(a))if a[k][c]),None)
        if k is None:continue
        a[r],a[k]=a[k],a[r];pivot=a[r][c]
        a[r]=[v/pivot for v in a[r]]
        for k in range(r+1,len(a)):
            b=a[k][c]
            a[k]=[x-b*y for x,y in zip(a[k],a[r])]
        r+=1
        if r==len(a):break
    return r

def verify(instance:dict,certificate:dict)->bool:
    try:
        h=decode(instance);n=len(h)
        keys=list((i,j,p,q)for i,j in combinations(range(n),2)for p,q in combinations(range(n),2))
        keyset=set(keys);status=certificate['status']
        if status=='inconsistent':
            entries=certificate['witness']
            if not entries or len(entries)>len(keys):return False
            row=[F(0)]*n;rhs=F(0);seen=set()
            for key,v in entries:
                key=tuple(key)
                if key not in keyset or key in seen:return False
                seen.add(key);a,b=_row(h,key);v=_q(v)
                row=[x+v*y for x,y in zip(row,a)];rhs+=v*b
            return not any(row) and rhs!=0 and rhs==_q(certificate['nonzero_rhs'])
        if status not in ('unique','ambiguous'):return False
        x=[_q(v)for v in certificate['particular']]
        null=[[_q(v)for v in z]for z in certificate['null_basis']]
        r=certificate['rank']
        if type(r)is not int or not 0<=r<=n or len(x)!=n or len(null)!=n-r:return False
        if any(len(z)!=n for z in null):return False
        if status!=('unique' if r==n else 'ambiguous'):return False
        # A positive certificate must satisfy every full commutator, not just pivots.
        for i in range(n):h[i][i][i]=x[i]
        if any(_comm(h,key) for key in keys):return False
        for z in null:
            for i in range(n):h[i][i][i]=x[i]+z[i]
            if any(_comm(h,key) for key in keys):return False
        for i in range(n):h[i][i][i]=F(0)
        if _rank(null)!=n-r:return False
        eqs=[tuple(k)for k in certificate['pivot_equations']]
        cols=certificate['pivot_columns']
        if len(eqs)!=r or len(set(eqs))!=r or any(k not in keyset for k in eqs):return False
        if len(cols)!=r or len(set(cols))!=r or any(type(i)is not int or not 0<=i<n for i in cols):return False
        minor=[[_row(h,key)[0][c]for c in cols]for key in eqs]
        return _rank(minor)==r
    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
        return False

def verify_stability(cert:dict)->bool:
    """Check a conditional bound with independently formed finite differences.

The unknown true tensor is NOT an input to this verifier. Its existence and
noise radii are hypotheses, not observations certified by the computation.
"""
    try:
        from math import comb
        if cert['status']!='conditional_bound':return False
        n=cert['n'];h=decode({'n':n,'orbits':cert['mixed_orbits']})
        s=cert['corruption_budget'];shat=cert['candidate_support']
        if type(s)is not int or not 0<=s<=n or sum(comb(n,k)for k in range(s+1))>2048:return False
        if len(shat)>s or len(set(shat))!=len(shat)or any(type(i)is not int or not 0<=i<n for i in shat):return False
        delta=_q(cert['delta']);eps=_q(cert['epsilon']);gamma=_q(cert['gamma'])
        if delta<0 or eps<0 or gamma<=delta:return False
        x=list(map(_q,cert['candidate_diagonal']));d=list(map(_q,cert['observed_diagonal']))
        if len(x)!=n or len(d)!=n:return False
        keys=[(i,j,p,q)for i,j in combinations(range(n),2)for p,q in combinations(range(n),2)]
        rows=[_row(h,key)for key in keys]
        gram=[[sum((a[i]*a[j]for a,_ in rows),F(0))for j in range(n)]for i in range(n)]
        expected=[tuple(c)for k in range(s+1)for c in combinations(range(n),k)]
        witnesses=cert['inverse_witnesses']
        if len(witnesses)!=len(expected):return False
        for truth,witness in zip(expected,witnesses):
            if tuple(witness['truth_support'])!=truth:return False
            inv=[[ _q(v)for v in row]for row in witness['inverse']]
            if len(inv)!=n or any(len(row)!=n for row in inv):return False
            union=set(truth)|set(shat)
            m=[[gram[i][j]+F(i==j and i not in union)for j in range(n)]for i in range(n)]
            if any(sum((m[i][k]*inv[k][j]for k in range(n)),F(0))!=F(i==j)for i in range(n)for j in range(n)):return False
            trace=sum((inv[i][i]for i in range(n)),F(0))
            if trace<=0 or gamma*gamma*trace>1:return False
        nx=_q(cert['norm_diagonal_upper']);nh=_q(cert['norm_mixed_upper'])
        res=_q(cert['commutator_residual_upper']);diagres=_q(cert['clean_diagonal_residual_upper'])
        rad=_q(cert['diagonal_radius']);total=_q(cert['tensor_radius'])
        if min(nx,nh,res,diagres,rad,total)<0:return False
        if nx*nx<sum((v*v for v in x),F(0)):return False
        if nh*nh<sum((v*v for sl in h for row in sl for v in row),F(0)):return False
        if res*res<sum(((sum((ai*xi for ai,xi in zip(a,x)),F(0))-b)**2 for a,b in rows),F(0)):return False
        if diagres*diagres<sum(((d[i]-x[i])**2 for i in range(n)if i not in shat),F(0)):return False
        a=res+delta*(nx+2*nh)+delta*delta;b=diagres+eps
        return rad*rad*(gamma-delta)**2>=a*a+b*b and total*total>=delta*delta+rad*rad
    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
        return False
