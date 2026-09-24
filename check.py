#!/usr/bin/env python3
"""Recheck saved scientific records; does not trust producer acceptance flags."""
from __future__ import annotations
import argparse,csv,json,sys,time,resource
from collections import Counter
from fractions import Fraction as Q
from itertools import combinations,combinations_with_replacement
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from verify import verify,verify_stability,decode,_comm,_rank

EXACT_SPECS={
    'ternary':(3,(-1,0,1),True),
    'pair_grid':(3,(-2,-1,0,1,2),False),
    'binary4':(4,(0,1),True),
}
EXACT_EXPECTED={
    'ternary':Counter({'unique':444,'ambiguous':73,'inconsistent':1670}),
    'pair_grid':Counter({'unique':0,'ambiguous':469,'inconsistent':15156}),
    'binary4':Counter({'unique':297,'ambiguous':230,'inconsistent':65009}),
}

def support_signature(record):
    """Independently check the disjoint-support/clique form of a saved basis."""
    cert=record['certificate'];n=record['input']['n']
    if cert['status']=='inconsistent':return None
    h=decode(record['input']);used=set();sizes=[]
    triples=list(combinations(range(n),3))
    for encoded in cert['null_basis']:
        z=list(map(Q,encoded));support={i for i,value in enumerate(z)if value}
        require(support,'empty null vector')
        require(not (used & support),'overlapping null-basis supports')
        used.update(support);sizes.append(len(support))
        for i,j in combinations(sorted(support),2):
            require(h[i][i][j] and h[i][j][j],'consistent free support is not a clique')
            require(h[i][j][j]*z[i]+h[i][i][j]*z[j]==0,'free-support gain mismatch')
        for triple in triples:
            if support.intersection(triple):
                require(h[triple[0]][triple[1]][triple[2]]==0,'all-distinct entry pins a declared free support')
    return ','.join(map(str,sorted(sizes))) if sizes else 'rigid'

def load(path):return json.loads(path.read_text())
def require(condition,reason):
    if not condition:raise ValueError(reason)

def full_tensor(n,orbits):
    # decode intentionally removes the diagonal; restore it explicitly.
    from itertools import combinations_with_replacement
    t=decode({'n':n,'orbits':orbits})
    for key,v in zip(combinations_with_replacement(range(n),3),orbits):
        if key[0]==key[2]:t[key[0]][key[0]][key[0]]=Q(v)
    return t

def householder_tensor(vector,weights):
    v=list(map(Q,vector));w=list(map(Q,weights));n=len(v)
    require(len(w)==n,'householder weight count')
    norm=sum((x*x for x in v),Q(0));require(norm>0,'zero householder vector')
    u=[[Q(i==j)-2*v[i]*v[j]/norm for j in range(n)]for i in range(n)]
    require(all(sum((u[i][a]*u[i][b]for i in range(n)),Q(0))==Q(a==b)
                for a in range(n)for b in range(n)),'nonorthogonal householder factor')
    return [[[sum((w[a]*u[i][a]*u[j][a]*u[k][a]for a in range(n)),Q(0))
              for k in range(n)]for j in range(n)]for i in range(n)]

def check(results):
    counts=Counter();families={};profiles={};family_indices={name:set()for name in EXACT_SPECS}
    seen=set();maxbits=0
    with (results/'exhaustive.jsonl').open()as f:
        for line in f:
            r=json.loads(line);require(r['case']not in seen,'duplicate finite case');seen.add(r['case'])
            require(verify(r['input'],r['certificate']),'invalid exact certificate '+r['case'])
            status=r['certificate']['status'];counts[status]+=1
            family,index_text=r['case'].rsplit('-',1);index=int(index_text)
            require(family in EXACT_SPECS,'unknown finite family')
            n,alphabet,include_all_distinct=EXACT_SPECS[family]
            keys=list(combinations_with_replacement(range(n),3))
            selected=[k for k in keys if k[0]!=k[-1] and (include_all_distinct or len(set(k))<3)]
            total=len(alphabet)**len(selected)
            require(0<=index<total,'out-of-range case index')
            require(index not in family_indices[family],'duplicate finite-family index')
            family_indices[family].add(index)
            require(r['case']==f'{family}-{index:05d}','noncanonical case identifier')
            number=index;digit_indices=[0]*len(selected)
            for position in range(len(selected)-1,-1,-1):
                number,remainder=divmod(number,len(alphabet));digit_indices[position]=remainder
            require(number==0,'case index decoding overflow')
            expected={key:alphabet[digit_indices[position]]for position,key in enumerate(selected)}
            require(r['input']['n']==n and [Q(v)for v in r['input']['orbits']]==[expected.get(k,0)for k in keys],
                    'case does not match declared Cartesian design')
            families.setdefault(family,Counter())[status]+=1
            signature=support_signature(r)
            if signature is not None:
                profiles.setdefault(family,Counter())[signature]+=1
                c=r['certificate']
                for value in c['particular']+sum(c['null_basis'],[]):
                    q=Q(value);maxbits=max(maxbits,abs(q.numerator).bit_length(),q.denominator.bit_length())
    require(families==EXACT_EXPECTED,'finite family classification mismatch')
    for family,(n,alphabet,include_all_distinct) in EXACT_SPECS.items():
        keys=list(combinations_with_replacement(range(n),3))
        selected=[k for k in keys if k[0]!=k[-1] and (include_all_distinct or len(set(k))<3)]
        require(family_indices[family]==set(range(len(alphabet)**len(selected))),'incomplete Cartesian family')
    require(counts==Counter({'unique':741,'ambiguous':772,'inconsistent':81835}),'finite classification mismatch')
    summary=load(results/'exhaustive-summary.json')
    require(summary['instances']==83348 and set(summary['counts'])==set(families)
            and all(Counter(summary['counts'][k])==v for k,v in families.items()),'finite summary mismatch')
    require(set(summary['consistent_support_profiles'])==set(profiles)
            and all(Counter(summary['consistent_support_profiles'][k])==v for k,v in profiles.items()),
            'finite support-profile summary mismatch')
    require(summary['maximum_completion_coefficient_bits']==maxbits,'finite coefficient-bit summary mismatch')
    structured=load(results/'structured.json');require(len(structured)==94,'structured case count')
    for r in structured:
        require(verify(r['input'],r['certificate']),'structured certificate '+r['case'])
        n=r['input']['n'];t=full_tensor(n,r['truth'])
        keys=[(i,j,p,q)for i,j in combinations(range(n),2)for p,q in combinations(range(n),2)]
        require(not any(_comm(t,k)for k in keys),'truth is not commuting')
        require(_rank([[t[i][j][k]for j in range(n)for k in range(n)]for i in range(n)])==r['tensor_rank'],'truth rank')
        if 'distance'in r:
            null=r['certificate']['null_basis']
            distance=min((sum(Q(a)!=0 for a in z)for z in null),default=None)
            require(distance==r['distance'],'structured distance')
        if r['case'].startswith('generic-rigid-'):
            require(r['certificate']['status']=='unique' and r['certificate']['rank']==n,'generic witness not rigid')
            expected=householder_tensor(r['householder_vector'],r['weights'])
            require(t==expected,'generic witness does not match declared construction')
            triples=[t[i][j][k]for i,j,k in combinations(range(n),3)]
            require(triples and all(triples),'generic witness has a zero all-distinct entry')
            require(r['all_distinct_nonzero']==len(triples),'generic witness triple count')
            require(Q(r['minimum_all_distinct_magnitude'])==min(map(abs,triples)),'generic witness magnitude')
            require(list(map(Q,r['certificate']['particular']))==[t[i][i][i]for i in range(n)],'generic witness diagonal')
    noise=load(results/'noise-certificates.json');require(len(noise)==36,'noise case count')
    with (results/'noise.csv').open(newline='')as f:csvrows={r['case']:r for r in csv.DictReader(f)}
    require(len(csvrows)==36,'noise csv count')
    for r in noise:
        c=r['certificate'];require(verify_stability(c),'invalid posterior witness '+r['case'])
        n=c['n'];t=full_tensor(n,r['truth']);h=decode({'n':n,'orbits':c['mixed_orbits']})
        truth=[t[i][i][i]for i in range(n)];x=list(map(Q,c['candidate_diagonal']))
        err=sum((a-b)**2 for a,b in zip(x,truth));row=csvrows[r['case']]
        require(err==Q(r['actual_error_squared'])==Q(row['error_squared']),'actual error mismatch')
        require(err<=Q(c['diagonal_radius'])**2,'truth outside radius')
        mixerr=sum((h[i][j][k]-t[i][j][k])**2 for i in range(n)for j in range(n)for k in range(n)if not i==j==k)
        require(mixerr<=Q(c['delta'])**2,'declared mixed bound violated')
        d=list(map(Q,c['observed_diagonal']));s=c['corruption_budget']
        require(sum((d[i]-truth[i])**2 for i in range(s,n))<=Q(c['epsilon'])**2,'clean diagonal bound violated')
        keys=[(i,j,p,q)for i,j in combinations(range(n),2)for p,q in combinations(range(n),2)]
        require(not any(_comm(t,k)for k in keys),'non-odeco noisy truth')
        for field in ('delta','epsilon','gamma','diagonal_radius','tensor_radius'):
            require(Q(row[field])==Q(c[field]),'csv/certificate discrepancy')
    groups={}
    for r in noise:
        c=r['certificate'];parts=r['case'].split('-');key=tuple(parts[1:4])
        value=(c['candidate_diagonal'],c['candidate_support'],c['diagonal_radius'])
        if key in groups:require(value==groups[key],'amplitude sensitivity differs from reported result')
        else:groups[key]=value
    d=load(results/'decoding.json')['cases'];require(len(d)==270,'decoding count')
    checks=0
    for r in d:
        ls=r['list_sizes'];checks+=len(ls)
        require(next(i for i,x in enumerate(ls)if x!=0)==r['minimum_errors'],'minimum list budget')
        require(ls[r['minimum_errors']]==r['minimum_list_size'],'minimum list size')
        require(next((i for i,x in enumerate(ls)if x=='infinite'),None)==r['infinite_at_budget'],'infinite threshold')
    require(checks==1566,'budget count')
    neg=load(results/'negative-controls.json')
    require(neg['distance_two_budget_one']['reason']=='singular_restricted_system','singular control')
    require(neg['excessive_noise_bound']['reason']=='noise_exceeds_certified_margin','noise-margin control')
    mutation=load(results/'mutations.json')
    require(len(mutation)==10 and all(r['rejected']for r in mutation),'mutation records')
    for suite in ('exhaustive','structured','decoding','noise','mutations'):
        require(load(results/(suite+'-summary.json'))['failures']==0,'reported failure')
    return {'exact_inputs':83348,'structured_inputs':94,'noisy_certificates':36,
            'decoding_words':270,'budget_records':1566,'mutation_records':10,'status':'verified'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=ROOT/'results');a=ap.parse_args()
    cpu=time.process_time();wall=time.perf_counter();report=check(a.results)
    report.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
