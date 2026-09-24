#!/usr/bin/env python3
"""Generate the bounded, deterministic scientific campaign with one worker.

Usage: python run.py --suite all --out results
Individual suites are resumable by selecting their name. Existing files for
that suite are replaced only on successful completion. No network is used.
"""
from __future__ import annotations
import argparse,csv,itertools,json,os,random,resource,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as Q

if not __debug__:
    raise RuntimeError(
        "run.py uses assertions as fail-closed scientific invariants; "
        "optimized Python (-O/-OO) is rejected"
    )
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from exact import (from_orbits,to_orbits,off_diagonal,complete,equations,solve,kernel_graph,
                   cauchy_family,distance_family,rational_orthogonal,synthesize)
from decoding import decode_diagonal,list_size
from stability import sqrt_bound,trimmed_candidate,certify
from verify import verify,verify_stability,_rank,_comm

SUITES=('exhaustive','structured','decoding','noise','mutations')

def norm2(xs):return sum((x*x for x in xs),Q(0))
def instance(t):return {'n':len(t),'orbits':to_orbits(t)}
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def check_kernel(h):
    g=kernel_graph(h);rows=list(equations(h));zv=[list(map(Q,z))for z in g['null_basis']]
    rank=_rank([a for _,a,_ in rows])
    assert len(zv)==len(h)-rank
    assert _rank(zv)==len(zv)
    for z in zv:assert all(sum((a*b for a,b in zip(row,z)),Q(0))==0 for _,row,_ in rows)
    # A separate, directly expanded energy formula guards sign conventions.
    for z in ([Q(i+1)for i in range(len(h))], [Q((-1)**i)for i in range(len(h))]):
        direct=sum((sum((a*b for a,b in zip(row,z)),Q(0))**2 for _,row,_ in rows),Q(0))
        pair=sum(((h[i][j][j]*z[i]+h[i][i][j]*z[j])**2 for i,j in itertools.combinations(range(len(h)),2)),Q(0))
        triple=2*sum((h[i][j][k]**2*(z[i]**2+z[j]**2+z[k]**2)for i,j,k in itertools.combinations(range(len(h)),3)),Q(0))
        assert direct==pair+triple
    return g

def exhaustive(out):
    """Enumerate three declared Cartesian input families with exact arithmetic.

    The four-dimensional binary family is a complete enumeration of every
    mixed symmetric orbit in that dimension.  It is small enough to retain
    every input and certificate, while exercising the graph/nullspace logic
    beyond the three-coordinate setting used by the worked examples.
    """
    specs=(
        ('ternary',3,(-1,0,1),True),
        ('pair_grid',3,(-2,-1,0,1,2),False),
        ('binary4',4,(0,1),True),
    )
    counts={};profiles={};path=out/'exhaustive.jsonl';num=0;maxbits=0
    with path.open('w')as f:
        for family,n,alphabet,include_all_distinct in specs:
            count={'unique':0,'ambiguous':0,'inconsistent':0};profile={}
            keys=list(itertools.combinations_with_replacement(range(n),3))
            selected=[k for k in keys if k[0]!=k[-1] and (include_all_distinct or len(set(k))<3)]
            for index,values in enumerate(itertools.product(alphabet,repeat=len(selected))):
                d=dict(zip(selected,values));h=from_orbits(n,[d.get(k,0)for k in keys]);c=solve(h)
                assert verify(instance(h),c);count[c['status']]+=1;num+=1
                if c['status']!='inconsistent':
                    # Graph/elimination agreement is checked for every nonempty
                    # fiber.  Inconsistent inputs instead carry an independently
                    # verified left-null witness and have no realizable component
                    # geometry to classify.
                    g=check_kernel(h);supports=[]
                    for comp in g['components']:
                        if comp['free']:
                            supports.append(len(comp['vertices']))
                            for i,j in itertools.combinations(comp['vertices'],2):assert h[i][i][j]and h[i][j][j]
                    signature=','.join(map(str,sorted(supports))) if supports else 'rigid'
                    profile[signature]=profile.get(signature,0)+1
                    for v in c['particular']+sum(c['null_basis'],[]):
                        q=Q(v);maxbits=max(maxbits,abs(q.numerator).bit_length(),q.denominator.bit_length())
                f.write(json.dumps({'case':f'{family}-{index:05d}','input':instance(h),'certificate':c},separators=(',',':'))+'\n')
            counts[family]=count;profiles[family]=dict(sorted(profile.items()))
    return {'instances':num,'counts':counts,'consistent_support_profiles':profiles,
            'maximum_completion_coefficient_bits':maxbits,'failures':0}

def structured(out):
    records=[]
    for n in range(1,9):
        for m in range(1,n+1):
            t=distance_family(n,m);h=off_diagonal(t);c=solve(h);g=check_kernel(h)
            assert verify(instance(h),c)
            r=_rank([[t[i][j][k]for j in range(n)for k in range(n)]for i in range(n)])
            assert r==n and c['rank']==n-1 and len(g['null_basis'])==1
            distance=sum(Q(a)!=0 for a in g['null_basis'][0]);assert distance==m
            records.append({'case':f'distance-{n}-{m}','input':instance(h),'truth':to_orbits(t),'certificate':c,'tensor_rank':r,'distance':distance})
    for n in range(2,9):
        for direction in ([Q(1)]*n,[Q((-1)**i*(i+1))for i in range(n)]):
            for kappa in (0,1,2):
                t=cauchy_family([Q(i)for i in range(n)],direction,Q(kappa));h=off_diagonal(t);c=solve(h)
                assert verify(instance(h),c)
                r=_rank([[t[i][j][k]for j in range(n)for k in range(n)]for i in range(n)])
                assert r==n-int(kappa==0) and c['rank']==n-1
                records.append({'case':f'cauchy-{n}-{len(records)}','input':instance(h),'truth':to_orbits(t),'certificate':c,'tensor_rank':r,'kappa':kappa})
    for n in range(3,9):
        t=synthesize(rational_orthogonal(list(range(1,n+1))),[Q(i+1)for i in range(n)])
        h=off_diagonal(t);c=solve(h);assert verify(instance(h),c)
        if c['status']!='unique':
            assert n==3 and c['rank']==2
            records.append({'case':'householder-three-coordinate-ambiguity','input':instance(h),'truth':to_orbits(t),'certificate':c,'tensor_rank':n,'distance':3})
            continue
        assert list(map(Q,c['particular']))==[t[i][i][i]for i in range(n)]
        d=[t[i][i][i]+Q((1<<30)*(i+1))for i in range(n)]
        dec=decode_diagonal(h,d);assert dec['distance']is None and list(map(Q,dec['estimate']))==[t[i][i][i]for i in range(n)]
        records.append({'case':f'rigid-{n}','input':instance(h),'truth':to_orbits(t),'certificate':c,'tensor_rank':n,'distance':None})
    # Exact witnesses for the generic-rigidity theorem.  These deterministic
    # examples do not prove the almost-sure statement; they exercise its
    # all-distinct-entry pinning mechanism in dimensions 3 through 12.
    for n in range(3,13):
        t=synthesize(rational_orthogonal(list(range(1,n+1))),[Q(1<<i)for i in range(n)])
        h=off_diagonal(t);c=solve(h);assert verify(instance(h),c)
        triples=[t[i][j][k]for i,j,k in itertools.combinations(range(n),3)]
        assert triples and all(triples) and c['status']=='unique' and c['rank']==n
        assert list(map(Q,c['particular']))==[t[i][i][i]for i in range(n)]
        r=_rank([[t[i][j][k]for j in range(n)for k in range(n)]for i in range(n)])
        assert r==n
        records.append({'case':f'generic-rigid-{n}','input':instance(h),'truth':to_orbits(t),
                        'certificate':c,'tensor_rank':r,'distance':None,
                        'householder_vector':[str(i)for i in range(1,n+1)],
                        'weights':[str(1<<i)for i in range(n)],
                        'all_distinct_nonzero':len(triples),
                        'minimum_all_distinct_magnitude':str(min(map(abs,triples)))})
    write_json(out/'structured.json',records)
    return {'instances':len(records),'distance_cases':36,'cauchy_cases':42,'rigid_cases':5,
            'additional_ambiguous_case':1,'generic_rigid_witnesses':10,'failures':0}

def decoding(out):
    counts=0;records=[]
    # A one-component fiber and a fiber with two free pairs and a free singleton.
    cases=[]
    cases.append(cauchy_family([Q(0),Q(1),Q(2)],[Q(1)]*3,Q(1)))
    a=cauchy_family([Q(0),Q(1)],[Q(1),Q(2)],Q(1));b=cauchy_family([Q(0),Q(2)],[Q(1),Q(1)],Q(1))
    t=from_orbits(5,[0]*35)
    for i,j,k in itertools.product(range(2),repeat=3):t[i][j][k]=a[i][j][k];t[i+2][j+2][k+2]=b[i][j][k]
    # A singleton independent block is itself a free component, correctly kept.
    t[4][4][4]=Q(3);cases.append(t)
    for ci,t in enumerate(cases):
        h=off_diagonal(t);base=solve(h);x=list(map(Q,base['particular']));g=kernel_graph(h)
        zs=[list(map(Q,z))for z in g['null_basis']];n=len(t)
        for wi,word in enumerate(itertools.product((-1,0,1),repeat=n)):
            d=[x[i]+Q(word[i])for i in range(n)];report=decode_diagonal(h,d)
            allchoices=[]
            for z in zs:
                observed=sorted({(d[i]-x[i])/z[i]for i in range(n)if z[i]})
                unseen=max(observed)+1
                allchoices.append([(v,False)for v in observed]+[(unseen,True)])
            oracle={s:[]for s in range(n+1)};infinite={s:False for s in range(n+1)}
            for choice in itertools.product(*allchoices):
                candidate=[x[i]+sum((alpha*z[i]for(alpha,_),z in zip(choice,zs)),Q(0))for i in range(n)]
                cost=sum(a!=b for a,b in zip(candidate,d));has_unseen=any(flag for _,flag in choice)
                for s in range(cost,n+1):
                    if has_unseen:infinite[s]=True
                    else:oracle[s].append(candidate)
            for s in range(n+1):
                expected='infinite'if infinite[s]else len(oracle[s]);assert list_size(report,s)==expected;counts+=1
            records.append({'case':f'word-{ci}-{wi}','observed_diagonal':[str(a)for a in d],
                            'minimum_errors':report['minimum_errors'],'minimum_list_size':report['minimum_list_size'],
                            'infinite_at_budget':report['infinite_at_budget'],'list_sizes':[list_size(report,s)for s in range(n+1)]})
    # Unnormalized l1 has coordinate-dependent leverage; this is not a benchmark.
    z=[Q(100),Q(1),Q(1)];truth=Q(1);ratios=[Q(2),truth,truth]
    costs={str(a):str(sum((abs(zi*(a-r))for zi,r in zip(z,ratios)),Q(0)))for a in set(ratios)}
    assert Q(costs['2'])<Q(costs['1'])
    ablation={'direction':list(map(str,z)),'ratios':list(map(str,ratios)),'true_parameter':'1','modal_parameter':'1','unnormalized_l1_parameter':'2','l1_costs':costs}
    write_json(out/'decoding.json',{'cases':records,'normalization_ablation':ablation})
    return {'words':len(records),'budget_checks':counts,'normalization_cases':1,'failures':0}

def noise(out):
    records=[];certificates=[]
    for n,s in ((3,1),(5,1),(7,2)):
        t=distance_family(n,n);h=off_diagonal(t);truth=[t[i][i][i]for i in range(n)]
        for power in (10,14,18):
            rng=random.Random(907+n);keys=list(itertools.combinations_with_replacement(range(n),3))
            e=from_orbits(n,[str(Q(0)if i==k else Q(rng.randint(-2,2),1<<power))for i,j,k in keys])
            v=[Q(rng.randint(-2,2),1<<power)for _ in range(n)]
            hh=[[[h[i][j][k]+e[i][j][k]for k in range(n)]for j in range(n)]for i in range(n)]
            delta=sqrt_bound(norm2(a for sl in e for row in sl for a in row));epsilon=sqrt_bound(norm2(v))
            reference=None
            for amplitude in (1,1<<10,1<<20,1<<40):
                d=[truth[i]+v[i]+(Q(amplitude*(i+1))if i<s else 0)for i in range(n)]
                candidate=trimmed_candidate(hh,d,s);assert candidate is not None
                cert=certify(hh,d,candidate['diagonal'],candidate['support'],s,delta,epsilon)
                assert cert['status']=='conditional_bound'and verify_stability(cert)
                err2=norm2(Q(a)-b for a,b in zip(candidate['diagonal'],truth));assert err2<=Q(cert['diagonal_radius'])**2
                identity=(candidate['diagonal'],candidate['support'],cert['diagonal_radius'])
                if reference is None:reference=identity
                # Not assumed by the theorem; measured amplitude-insensitivity in this design.
                equal=identity==reference
                case=f'noise-{n}-{s}-{power}-{amplitude}'
                certificates.append({'case':case,'certificate':cert,'truth':to_orbits(t),'actual_error_squared':str(err2)})
                records.append({'case':case,'n':n,'s':s,'noise_exponent':power,'amplitude':amplitude,'selected_support':';'.join(map(str,candidate['support'])),
                                'delta':str(delta),'epsilon':str(epsilon),'gamma':cert['gamma'],'error_squared':str(err2),
                                'diagonal_radius':cert['diagonal_radius'],'tensor_radius':cert['tensor_radius'],'verified':True,'same_as_unit_amplitude':equal})
    with(out/'noise.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write_json(out/'noise-certificates.json',certificates)
    # Null controls are mathematical noncertifiability/noise-margin checks.
    t=distance_family(2,2);h=off_diagonal(t);d=[t[i][i][i]for i in range(2)]
    a=certify(h,d,d,[0],1,Q(0),Q(0));assert a['reason']=='singular_restricted_system'
    t=distance_family(3,3);h=off_diagonal(t);d=[t[i][i][i]for i in range(3)]
    b=certify(h,d,d,[0],1,Q(100),Q(0));assert b['reason']=='noise_exceeds_certified_margin'
    write_json(out/'negative-controls.json',{'distance_two_budget_one':a,'excessive_noise_bound':b})
    return {'instances':len(records),'verified':len(records),'same_as_unit_amplitude':sum(r['same_as_unit_amplitude']for r in records),'negative_controls':2,'failures':0}

def mutations(out):
    import copy
    t=distance_family(3,3);h=off_diagonal(t);i=instance(h);c=solve(h);assert verify(i,c)
    cases=[]
    def test(name,m):cases.append({'case':name,'rejected':not verify(i,m)});assert cases[-1]['rejected']
    m=copy.deepcopy(c);m['particular'][0]=str(Q(m['particular'][0])+1);test('altered-particular',m)
    m=copy.deepcopy(c);m['null_basis'][0][0]='0';test('truncated-direction',m)
    m=copy.deepcopy(c);m['rank']=3;test('false-full-rank',m)
    m=copy.deepcopy(c);m['pivot_equations'][0]=[0,0,0,0];test('invalid-equation',m)
    m=copy.deepcopy(c);m['particular'][0]='1/0';test('invalid-rational',m)
    # Independent inconsistency witness mutation.
    ih=from_orbits(3,[0,-1,-1,-1,-1,-1,0,-1,0,0]);bad=solve(ih)
    if bad['status']!='inconsistent':
        for vals in itertools.product((-1,0,1),repeat=7):
            ih=from_orbits(3,[0,*vals[:5],0,*vals[5:],0]);bad=solve(ih)
            if bad['status']=='inconsistent':break
    assert verify(instance(ih),bad);bad['nonzero_rhs']='0';assert not verify(instance(ih),bad)
    cases.append({'case':'false-inconsistency-sum','rejected':True})
    d=[t[i][i][i]for i in range(3)];st=certify(h,d,d,[0],1,Q(0),Q(0));assert verify_stability(st)
    for name,edit in [('altered-inverse',lambda m:m['inverse_witnesses'][0]['inverse'][0].__setitem__(0,'0')),
                      ('omitted-support',lambda m:m['inverse_witnesses'].pop()),
                      ('inflated-margin',lambda m:m.__setitem__('gamma','100')),
                      ('negative-noise',lambda m:m.__setitem__('delta','-1'))]:
        m=copy.deepcopy(st);edit(m);rejected=not verify_stability(m);assert rejected;cases.append({'case':name,'rejected':rejected})
    write_json(out/'mutations.json',cases);return {'instances':len(cases),'rejected':len(cases),'failures':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=(*SUITES,'all'),default='all');ap.add_argument('--out',type=Path,default=ROOT/'results');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.suite=='all':
        # Run one suite at a time in a fresh interpreter.  The exhaustive
        # campaign creates many short-lived Fraction objects; process
        # isolation releases them deterministically before the next suite and
        # avoids allocator/collector interactions in the documented command.
        # This is sequential execution, not a worker pool.
        for suite in SUITES:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--suite',suite,'--out',str(args.out)],check=True)
        return
    suite=args.suite
    temp=args.out/('.'+suite+'-working');temp.mkdir(exist_ok=True)
    if any(temp.iterdir()):raise RuntimeError('nonempty interrupted work directory; inspect before removing it')
    wall=time.perf_counter();cpu=time.process_time();result=globals()[suite](temp)
    result.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(temp/(suite+'-summary.json'),result)
    for path in temp.iterdir():os.replace(path,args.out/path.name)
    temp.rmdir();print(json.dumps({'suite':suite,**result},sort_keys=True),flush=True)
if __name__=='__main__':main()
