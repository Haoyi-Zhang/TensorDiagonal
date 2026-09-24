"""Exact boundary checks supplementary to, not replacements for, general proofs."""
import copy,sys,unittest
from fractions import Fraction as Q
from itertools import combinations,combinations_with_replacement,product
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exact import (from_orbits,to_orbits,off_diagonal,cauchy_family,solve,equations,
                   kernel_graph,complete,rotate_tensor,rational_orthogonal,synthesize)
from stability import inverse,certify,supports,normal_system
from verify import verify,verify_stability,_rank

class BoundaryChecks(unittest.TestCase):
    def test_dimension_one(self):
        h=from_orbits(1,[0]);c=solve(h)
        self.assertEqual(c['rank'],0);self.assertTrue(verify({'n':1,'orbits':['0']},c))
    def test_two_coordinate_cases(self):
        for a,b in product((-2,0,3),repeat=2):
            h=from_orbits(2,[0,a,b,0]);c=solve(h)
            self.assertEqual(c['status'],'ambiguous')
            self.assertLess(c['rank'],2)
            x=list(map(Q,c['particular']))
            self.assertEqual(b*x[0]+a*x[1],a*a+b*b)
            self.assertTrue(verify({'n':2,'orbits':to_orbits(h)},c))
    def test_full_rank_inconsistent(self):
        h=from_orbits(3,[0,0,0,0,1,0,0,0,0,0]);c=solve(h)
        self.assertEqual(_rank([a for _,a,_ in equations(h)]),3)
        self.assertEqual(c['status'],'inconsistent')
        self.assertTrue(verify({'n':3,'orbits':to_orbits(h)},c))
    def test_four_coordinate_binary_boundaries(self):
        keys=list(combinations_with_replacement(range(4),3))
        mixed=[key for key in keys if key[0]!=key[-1]]
        def tensor(bits):
            values=dict(zip(mixed,bits))
            return from_orbits(4,[values.get(key,0)for key in keys])
        cases=(
            (tensor((0,)*16),'ambiguous',0),
            (tensor((1,)*16),'unique',4),
            (tensor(tuple(map(int,format(5,'016b')))),'inconsistent',None),
        )
        for h,status,rank in cases:
            cert=solve(h);self.assertEqual(cert['status'],status)
            if rank is not None:self.assertEqual(cert['rank'],rank)
            self.assertTrue(verify({'n':4,'orbits':to_orbits(h)},cert))
    def test_worked_matrices(self):
        t=cauchy_family([Q(0),Q(1),Q(1,2)],[Q(1)]*3,Q(0));h=off_diagonal(t)
        rows,g,_=normal_system(h)
        self.assertEqual(g,[[5,-1,-4],[-1,5,-4],[-4,-4,8]])
        expected=[Q(85,24),Q(85,24),Q(41,12)]
        for j in range(3):
            m=[[g[i][k]+Q(i==k==j)for k in range(3)]for i in range(3)]
            b=inverse(m);self.assertIsNotNone(b)
            self.assertEqual(sum(b[i][i]for i in range(3)),expected[j])
            self.assertLessEqual(Q(1,4)*expected[j],1)
        # The asserted nonzero eigenvalues, without a floating eigensolver.
        for v,value in [([1,-1,0],6),([1,1,-2],12),([1,1,1],0)]:
            self.assertEqual([sum(g[i][j]*v[j]for j in range(3))for i in range(3)],[value*a for a in v])
        self.assertLess(Q(3,11),Q(24,85))
    def test_arbitrary_amplitude_exact_bound(self):
        t=cauchy_family([Q(0),Q(1),Q(1,2)],[Q(1)]*3,Q(1));h=off_diagonal(t)
        x=[t[i][i][i]for i in range(3)]
        for a in (Q(0),Q(-2),Q(10**30)):
            d=x.copy();d[0]+=a;c=certify(h,d,x,[0],1,0,0)
            self.assertTrue(verify_stability(c));self.assertEqual(Q(c['diagonal_radius']),0)
    def test_rank_one_slice(self):
        for a,b in ((Q(2),Q(3)),(Q(-1),Q(4))):
            t=from_orbits(2,[str(a*a/b),str(a),str(b),str(b*b/a)])
            self.assertEqual(_rank([[t[i][j][k]for j in range(2)for k in range(2)]for i in range(2)]),1)
    def test_nonclique_free_graph_is_not_consistency(self):
        # Balanced path 1--2--3; associativity forbids an unclosed path.
        h=from_orbits(3,[0,1,0,-1,0,0,0,1,-1,0])
        self.assertEqual(len(kernel_graph(h)['null_basis']),1)
        self.assertEqual(solve(h)['status'],'inconsistent')
    def test_pin_orientation(self):
        h=from_orbits(2,[0,1,0,0]);g=kernel_graph(h)
        self.assertEqual(g['null_basis'],[['1','0']])
    def test_mask_not_preserved_by_general_rotation(self):
        t=from_orbits(2,[1,0,0,0]);r=rational_orthogonal([1,2])
        rotated=rotate_tensor(t,r)
        self.assertNotEqual(rotated[0][0][1],0)
        perm=[[Q(0),Q(-1)],[Q(1),Q(0)]]
        mixed=off_diagonal(rotate_tensor(t,perm))
        self.assertFalse(any(v for sl in mixed for row in sl for v in row))
    def test_limits_and_invalid_encodings(self):
        for n in (0,17,True):
            with self.assertRaises(ValueError):from_orbits(n,[])
        with self.assertRaises(ValueError):from_orbits(1,[0.1])
        with self.assertRaises(ValueError):supports(16,8)
        self.assertFalse(verify({'n':True,'orbits':['0']},{}))
    def test_duplicate_and_missing_witnesses(self):
        t=cauchy_family([Q(0),Q(1),Q(2)],[Q(1)]*3,Q(1));h=off_diagonal(t)
        x=[t[i][i][i]for i in range(3)];c=certify(h,x,x,[0],1,0,0)
        bad=copy.deepcopy(c);bad['inverse_witnesses'][1]=bad['inverse_witnesses'][0]
        self.assertFalse(verify_stability(bad))
        bad=copy.deepcopy(c);bad['inverse_witnesses'].pop()
        self.assertFalse(verify_stability(bad))
    def test_support_free_query(self):
        h=off_diagonal(cauchy_family([Q(0),Q(1),Q(2)],[Q(1),Q(2),Q(3)],Q(1)))
        rows=[a for _,a,_ in equations(h)]
        self.assertEqual(_rank(rows),2)
        for i in range(3):
            self.assertEqual(_rank(rows+[[Q(j==i)for j in range(3)]]),3)


    def test_generic_rigidity_exact_witnesses(self):
        for n in range(3,13):
            t=synthesize(rational_orthogonal(list(range(1,n+1))),[Q(1<<i)for i in range(n)])
            h=off_diagonal(t);c=solve(h)
            self.assertEqual(c['status'],'unique')
            self.assertEqual(c['rank'],n)
            self.assertTrue(all(t[i][j][k] for i,j,k in combinations(range(n),3)))
            self.assertEqual(list(map(Q,c['particular'])),[t[i][i][i]for i in range(n)])

    def test_query_cost_and_error_correction_by_bruteforce(self):
        # One free triple and one free singleton.  Unique completion must hit
        # both; correction of one error needs only the singleton queried.
        block=cauchy_family([Q(0),Q(1),Q(3)],[Q(1),Q(-2),Q(4)],Q(1))
        t=from_orbits(4,[0]*20)
        for i in range(3):
            for j in range(3):
                for k in range(3):t[i][j][k]=block[i][j][k]
        t[3][3][3]=Q(5)
        h=off_diagonal(t);rows=[a for _,a,_ in equations(h)]
        costs=[Q(7),Q(2),Q(9),Q(5)]
        best_unique=None;best_one_error=None
        for mask in range(1<<4):
            queried=[i for i in range(4)if mask>>i&1]
            augmented=rows+[[Q(j==i)for j in range(4)]for i in queried]
            cost=sum((costs[i]for i in queried),Q(0))
            nullity=4-_rank(augmented)
            if nullity==0:best_unique=cost if best_unique is None else min(best_unique,cost)
            # Remaining free directions must all have support at least three.
            remaining=[]
            for z in kernel_graph(h)['null_basis']:
                zz=list(map(Q,z))
                if all(zz[i]==0 for i in queried):remaining.append(zz)
            distance=min((sum(v!=0 for v in z)for z in remaining),default=10**9)
            if distance>2:best_one_error=cost if best_one_error is None else min(best_one_error,cost)
        self.assertEqual(best_unique,Q(7))  # coordinate 1 plus singleton 3
        self.assertEqual(best_one_error,Q(5))

    def test_repeated_query_bound(self):
        z=[Q(2),Q(-3),Q(5),Q(7)]
        alpha=Q(11,4);q=(0,2,3);eps=[Q(1,5),Q(-1,7),Q(2,9)]
        y=[alpha*z[i]+e for i,e in zip(q,eps)]
        denom=sum((z[i]*z[i]for i in q),Q(0))
        ahat=sum((z[i]*yi for i,yi in zip(q,y)),Q(0))/denom
        left=sum((((ahat-alpha)*v)**2 for v in z),Q(0))*sum((z[i]*z[i]for i in q),Q(0))
        right=sum((e*e for e in eps),Q(0))*sum((v*v for v in z),Q(0))
        self.assertLessEqual(left,right)

if __name__=='__main__':unittest.main()
