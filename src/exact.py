"""Exact diagonal completion of real symmetric cubic tensors.

All proof-critical arithmetic uses fractions.Fraction. The base field of the
mathematical conclusion is R, not Q: commuting real symmetric slices have a
real orthogonal decomposition, even when its factors are irrational.
"""
from __future__ import annotations
from fractions import Fraction as Q
from itertools import combinations, combinations_with_replacement, permutations
from typing import Iterator

Tensor = list[list[list[Q]]]

def from_orbits(n: int, values: list[str | int]) -> Tensor:
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 16:
        raise ValueError('dimension must be an integer in [1,16] for this implementation')
    keys = list(combinations_with_replacement(range(n), 3))
    if len(values) != len(keys):
        raise ValueError('wrong number of symmetric-orbit entries')
    t = [[[Q(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for key, value in zip(keys, values):
        if not isinstance(value,(str,int)) or isinstance(value,bool):
            raise ValueError('use exact integer or rational-string entries')
        if len(str(value)) > 4096:
            raise ValueError('rational encoding too long')
        v = Q(value)
        for i,j,k in set(permutations(key)):
            t[i][j][k] = v
    return t

def to_orbits(t: Tensor) -> list[str]:
    return [str(t[i][j][k]) for i,j,k in combinations_with_replacement(range(len(t)),3)]

def off_diagonal(t: Tensor) -> Tensor:
    h = [[row[:] for row in sl] for sl in t]
    for i in range(len(t)): h[i][i][i] = Q(0)
    return h

def complete(h: Tensor, x: list[Q]) -> Tensor:
    if len(x) != len(h): raise ValueError('diagonal length mismatch')
    t = [[row[:] for row in sl] for sl in h]
    for i, xi in enumerate(x): t[i][i][i] = Q(xi)
    return t

def equations(h: Tensor) -> Iterator[tuple[tuple[int,int,int,int], list[Q], Q]]:
    """Yield row a and rhs b in a*x=b for every independent commutator entry."""
    n=len(h)
    for i,j in combinations(range(n),2):
        for p,q in combinations(range(n),2):
            row=[Q(0)]*n
            row[i] = (int(p==i)-int(q==i))*h[j][p][q]
            row[j] = (int(q==j)-int(p==j))*h[i][p][q]
            rhs = -sum((h[i][p][k]*h[j][k][q]-h[j][p][k]*h[i][k][q] for k in range(n)), Q(0))
            yield (i,j,p,q), row, rhs

def _combine(dst: dict, src: dict, multiplier: Q) -> None:
    for k,v in src.items():
        dst[k]=dst.get(k,Q(0))+multiplier*v
        if not dst[k]: del dst[k]

def solve(h: Tensor) -> dict:
    """Streaming exact elimination, with a sparse inconsistency witness.

A returned certificate never equates full column rank with consistency.
Pivot equations plus pivot columns give a rank lower-bound witness; the
explicit null basis provides the matching upper bound.
"""
    n=len(h); basis=[]; pivot_ids=[]
    for key,row,rhs in equations(h):
        aug=row+[rhs]; provenance={key:Q(1)}
        for pivot,b,trace in basis:
            mult=aug[pivot]
            if mult:
                aug=[a-mult*c for a,c in zip(aug,b)]
                _combine(provenance,trace,-mult)
        pivot=next((i for i in range(n) if aug[i]),None)
        if pivot is None:
            if aug[n]:
                return {'status':'inconsistent','witness':[[list(k),str(v)] for k,v in sorted(provenance.items())],
                        'nonzero_rhs':str(aug[n])}
            continue
        divisor=aug[pivot]
        aug=[a/divisor for a in aug]
        provenance={k:v/divisor for k,v in provenance.items()}
        basis.append((pivot,aug,provenance));basis.sort(key=lambda v:v[0]);pivot_ids.append(list(key))
    # Back substitution works because each later pivot row is zero in all
    # earlier pivot columns, independent of the order in which pivots arose.
    # Make a reduced system explicitly to also handle nonmonotone pivot order.
    for k in range(len(basis)-1,-1,-1):
        p,b,trace=basis[k]
        for j in range(k):
            pj,a,tr=basis[j];mult=a[p]
            if mult:
                a=[x-mult*y for x,y in zip(a,b)];_combine(tr,trace,-mult)
                basis[j]=(pj,a,tr)
    pivots=[p for p,_,_ in basis];free=[i for i in range(n) if i not in pivots]
    x=[Q(0)]*n
    for p,b,_ in basis: x[p]=b[n]
    null=[]
    for f in free:
        v=[Q(0)]*n;v[f]=Q(1)
        for p,b,_ in basis:v[p]=-b[f]
        null.append(v)
    return {'status':'unique' if not free else 'ambiguous','particular':[str(v) for v in x],
            'null_basis':[[str(v) for v in z] for z in null],
            'rank':len(basis),'pivot_equations':pivot_ids,'pivot_columns':pivots}

def kernel_graph(h: Tensor) -> dict:
    """Exact gain-graph kernel calculation, independent of elimination."""
    n=len(h); pinned=set(); adj=[[] for _ in range(n)]
    for i,j,k in combinations(range(n),3):
        if h[i][j][k]: pinned.update((i,j,k))
    for i,j in combinations(range(n),2):
        a=h[i][i][j];b=h[i][j][j]
        if a and b:
            adj[i].append((j,-b/a));adj[j].append((i,-a/b))
        elif b: pinned.add(i)
        elif a: pinned.add(j)
    seen=set();free=[];components=[]
    for start in range(n):
        if start in seen:continue
        weights={start:Q(1)};stack=[start];seen.add(start);balanced=True
        while stack:
            i=stack.pop()
            for j,gain in adj[i]:
                desired=weights[i]*gain
                if j in weights:
                    if weights[j]!=desired:balanced=False
                else:
                    weights[j]=desired;stack.append(j);seen.add(j)
        vertices=sorted(weights)
        isfree=balanced and not (pinned & weights.keys())
        components.append({'vertices':vertices,'balanced':balanced,'pinned':sorted(pinned & weights.keys()),'free':isfree})
        if isfree:
            z=[Q(0)]*n
            for i in vertices:z[i]=weights[i]
            free.append(z)
    return {'components':components,'null_basis':[[str(q) for q in z] for z in free]}

def rational_orthogonal(vector: list[int]) -> list[list[Q]]:
    n=len(vector);d=sum(v*v for v in vector)
    if not d:raise ValueError('Householder vector must be nonzero')
    return [[Q(i==j)-Q(2*vector[i]*vector[j],d) for j in range(n)]for i in range(n)]

def synthesize(u: list[list[Q]], weights: list[Q]) -> Tensor:
    n=len(u);r=len(weights)
    if any(len(row)!=r for row in u):raise ValueError('factor/weight dimensions')
    return [[[sum((weights[a]*u[i][a]*u[j][a]*u[k][a] for a in range(r)),Q(0)) for k in range(n)] for j in range(n)]for i in range(n)]


def cauchy_family(nodes: list[Q], direction: list[Q], parameter: Q) -> Tensor:
    """A real odeco cubic family with a full-support diagonal ambiguity.

    Nodes must be distinct real rationals and direction coordinates nonzero.
    Every member is odeco; its rank is n except at parameter zero, where it is n-1.
    """
    n=len(nodes);nodes=list(map(Q,nodes));z=list(map(Q,direction));parameter=Q(parameter)
    if len(z)!=n or not 1<=n<=16 or len(set(nodes))!=n or any(not a for a in z):
        raise ValueError('require 1<=n<=16, distinct nodes and nonzero matching direction')
    t=[[[Q(0) for _ in range(n)]for _ in range(n)]for _ in range(n)]
    for i in range(n):
        t[i][i][i]=z[i]*(parameter-sum((1/(z[k]**2*(nodes[k]-nodes[i])) for k in range(n) if k!=i),Q(0)))
        for j in range(n):
            if i!=j:
                a=1/(z[j]*(nodes[j]-nodes[i]))
                t[i][i][j]=t[i][j][i]=t[j][i][i]=a
    return t

def rotate_tensor(t: Tensor, rotation: list[list[Q]]) -> Tensor:
    """Change coordinates in three bounded contractions, with exact arithmetic."""
    n=len(t);r=rotation
    if len(r)!=n or any(len(row)!=n for row in r):raise ValueError('rotation dimensions')
    a=[[[sum((r[i][p]*t[p][j][k]for p in range(n)),Q(0))for k in range(n)]for j in range(n)]for i in range(n)]
    b=[[[sum((r[j][p]*a[i][p][k]for p in range(n)),Q(0))for k in range(n)]for j in range(n)]for i in range(n)]
    return [[[sum((r[k][p]*b[i][j][p]for p in range(n)),Q(0))for k in range(n)]for j in range(n)]for i in range(n)]

def distance_family(n:int,m:int,parameter:Q=Q(1)) -> Tensor:
    """Full-rank n-dimensional examples with a one-dimensional fiber of distance m.

For m<n a scalar-identity anchor makes the full odeco rank equal to n. A
rational rotation of the anchored complement pins every complementary
coordinate. This is a constructive sharpness example, not a sampled workload.
"""
    if not 1<=m<=n<=16:raise ValueError('require 1<=m<=n<=16')
    if m==n:return cauchy_family([Q(i)for i in range(n)],[Q(1)]*n,parameter)
    small=cauchy_family([Q(i)for i in range(m)],[Q(1)]*m,parameter)
    t=[[[Q(0)for _ in range(n)]for _ in range(n)]for _ in range(n)]
    for i in range(m):
        for j in range(m):
            for k in range(m):t[i][j][k]=small[i][j][k]
        t[i][i][i]-=i
        t[i][i][m]=t[i][m][i]=t[m][i][i]=Q(1)
    t[m][m][m]=Q(1)
    for i in range(m+1,n):t[i][i][i]=Q(i-m+1)
    q=n-m;v=rational_orthogonal(list(range(1,q+1)))
    rotation=[[Q(i==j)for j in range(n)]for i in range(n)]
    for i in range(q):
        for j in range(q):rotation[m+i][m+j]=v[i][j]
    return rotate_tensor(t,rotation)
