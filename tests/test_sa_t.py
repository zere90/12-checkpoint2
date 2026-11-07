import unittest
from src.meta.nm import nelder_mead

class TestNelderMead(unittest.TestCase):
    def test_quadratic(self):
        f=lambda x:(x[0]-2)**2+3*(x[1]+1)**2
        r=nelder_mead(f,(0,0),step=(1,1),max_evals=150)
        self.assertLess(r['fun'],1e-7)
        self.assertLessEqual(r['nfev'],150)
    def test_budget(self):
        r=nelder_mead(lambda x:sum(v*v for v in x),(1,2),max_evals=7)
        self.assertLessEqual(r['nfev'],7)
        self.assertEqual(len(r['history']),r['nfev'])
        self.assertTrue(all(a>=b for a,b in zip(r['history'],r['history'][1:])))
    def test_initial_simplex(self):
        seen=[]
        def f(x): seen.append(x); return sum(v*v for v in x)
        nelder_mead(f,(1,2),step=(.2,.4),max_evals=3)
        self.assertEqual(seen,[(1.,2.),(1.2,2.),(1.,2.4)])
if __name__=='__main__': unittest.main()
