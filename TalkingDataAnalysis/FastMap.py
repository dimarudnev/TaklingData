import math
import random

# need scipy as usual
import scipy
DISTANCE_ITERATIONS=10



class FastMap: 

    def __init__(self, dist,verbose=False): 
        if dist.max()>1:
            dist/=dist.max()

        self.dist=dist
        self.verbose=verbose

    def _furthest(self, o): 
        mx=-1000000
        idx=-1
        for i in range(len(self.dist)): 
            d=self._dist(i,o, self.col)
            if d>mx: 
                mx=d
                idx=i

        return idx

    def _pickPivot(self):
        """Find the two most distant points"""
        o1=random.randint(0, len(self.dist)-1)
        o2=-1

        i=DISTANCE_ITERATIONS

        while i>0: 
            o=self._furthest(o1)
            if o==o2: break
            o2=o
            o=self._furthest(o2)
            if o==o1: break
            o1=o
            i-=1

        self.pivots[self.col]=(o1,o2)
        return (o1,o2)


    def _map(self, K): 
        if K==0: return 
    
        px,py=self._pickPivot()

        if self._dist(px,py,self.col)==0: 
            return 
        for i in range(len(self.dist)):
            self.res[i][self.col]=self._x(i, px,py)

        self.col+=1
        self._map(K-1)

    def _x(self,i,x,y):
        """Project the i'th point onto the line defined by x and y"""
        dix=self._dist(i,x,self.col)
        diy=self._dist(i,y,self.col)
        dxy=self._dist(x,y,self.col)
        return (dix + dxy - diy) / 2*math.sqrt(dxy)

    def _dist(self, x,y,k): 
        """Recursively compute the distance based on previous projections"""
        if k==0: return self.dist[x,y]**2
    
        rec=self._dist(x,y, k-1)
        resd=(self.res[x][k] - self.res[y][k])**2
        return rec-resd


    def map(self, K): 
        self.col=0
        self.res=scipy.zeros((len(self.dist),K))
        self.pivots=scipy.zeros((K,2),"i")
        self._map(K)
        return self.res
