import numpy as np

class MinMaxTransform1D:
    '''
    linear scaling into [lb,ub]
    '''
    def __init__(self, lb = -1.0, ub = 1.0):
        assert(lb < ub)
        self.lb = lb
        self.ub = ub
        self.xmin = None
        self.xmax = None
        self.a = None
        self.b = None
            
    def fit(self, x: np.array):
        assert(len(x.shape)==1)
        self.xmin = x.min()
        self.xmax = x.max()
        self.a = self.ub-self.lb
        self.b = self.lb
            
    def transform(self, x: np.array):
        assert(len(x.shape)==1)   
        z = self.a * (x - self.xmin)/(self.xmax - self.xmin) + self.b
        return z

    def forward_derivative(self, x: np.array):
        assert(len(x.shape)==1)   
        return self.a * np.ones_like(x) /(self.xmax - self.xmin) 
            
    def inverse_transform(self, z: np.array):
        assert(len(z.shape)==1)
        x = (self.xmax - self.xmin) * (z-self.b)/self.a + self.xmin
        return x

    def inverse_derivative(self, z: np.array):
        assert(len(z.shape)==1)   
        return (self.xmax - self.xmin) * np.ones_like(z)/self.a
            
            

class LogTransform1D:
    '''
    apply log transform to 1d array with positive entries, followed by linear scaling into [lb,ub]
    '''
    def __init__(self, lb = 0.0, ub = 1.0, eps = 0.0):
        assert(lb < ub)
        self.lb = lb
        self.ub = ub
        self.logxmin = None
        self.logxmax = None
        self.a = None
        self.b = None
        self.eps = eps
            
    def fit(self, x: np.array):
        assert(len(x.shape)==1)
        self.logxmin = np.log10(x.min()+self.eps)
        self.logxmax = np.log10(x.max()+self.eps)
        self.a = self.ub-self.lb
        self.b = self.lb
            
    def transform(self, x: np.array):
        assert(len(x.shape)==1)   
        z = self.a * (np.log10(x+self.eps) - self.logxmin)/(self.logxmax - self.logxmin) + self.b
        return z

    def forward_derivative(self, x: np.array):
        return self.a / (self.logxmax - self.logxmin)  * ( 1/( (x+self.eps) * np.log(10)) )
        
    def inverse_transform(self, z: np.array):
        assert(len(z.shape)==1)
        x = 10**( (self.logxmax - self.logxmin) * (z-self.b)/self.a + self.logxmin ) - self.eps
        return x
        
    def inverse_derivative(self, z: np.array):
        u = 10**( (self.logxmax - self.logxmin) * (z-self.b)/self.a + self.logxmin )         
        return u * np.log(10) *  (self.logxmax - self.logxmin) / self.a
        
class SymLogTransform1D:
    '''
    apply signed log transform to 1d array:
    
    log(1+x) if x >= 0
    -log(1-x) if x<0
    
    (this is equivalent to np.sign(x)*np.log(1+ np.abs(x)) )
    
    followed by linear scaling into [-a,a]
    '''
    def __init__(self, a=1.0):
        self.zmin = None
        self.zmax = None
        self.zbar = None
        self.a = a
            
    def fit(self, x: np.array):
        assert(len(x.shape)==1)
        z = np.sign(x)*np.log10(1.0 + np.abs(x))
        self.zmin = z.min()
        self.zmax = z.max()
        self.zbar = 0.5*(self.zmin + self.zmax)
            
    def transform(self, x: np.array):
        assert(len(x.shape)==1)
        z = np.sign(x)*np.log10(1.0 + np.abs(x))
        z = self.a * 2 * (z - self.zbar)/(self.zmax - self.zmin)
        return z

    def forward_derivative(self, x: np.array):
        dz = 1 / ((1.0+np.abs(x))*np.log(10))
        return self.a * 2 / (self.zmax - self.zmin) * dz 
            
    def inverse_transform(self, z: np.array):
        assert(len(z.shape)==1)
        x = (self.zmax - self.zmin) * z/(2*self.a) + self.zbar
        x = np.sign(x) * (10**( np.abs(x) ) - 1.0) 
        return x
     
    def inverse_derivative(self, z: np.array):
        x = (self.zmax - self.zmin) * z/(2*self.a) + self.zbar
        u = 10**( np.abs(x) )         
        return u * np.log(10) * (self.zmax - self.zmin)/(2*self.a)

class DataTransform:
    '''
    contains 1D data transforms, assumes array dimensions are (samples, features)
    '''
    def __init__(self, transforms: list):
        self.transforms = transforms
        self.length = len(transforms)
    
    def fit(self, x: np.array):
        assert(self.length == x.shape[-1])
        for i,f in enumerate(self.transforms):
            f.fit(x[:,i])
        
    def transform(self, x: np.array):
        '''
        Applies transform i to the i-th column (variable / feature) of x
        x is (B,nt)
        z is (B,nt)
        '''
        assert(self.length == x.shape[-1])
        z = x.copy()
        for i,f in enumerate(self.transforms):
            z[:,i] = f.transform(z[:,i])
        return z

    def forward_derivative(self, x: np.array):
        '''
        Computes the point-wise jacobian of the transform. 
        Note that by construction, the transforms are applied to each feature (dimension of x) independently.
        Therefore, the Jacobian is diagonal, and we return a vector.
        
        x is (1,nt)
        J is (nt,)
        '''
        assert(self.length == x.shape[-1])
        J = np.zeros((self.length,))
        for i,f in enumerate(self.transforms):
            J[i] = f.forward_derivative(x[0,i][None]) 
        return J
        
    
    def inverse_transform(self, z: np.array):
        assert(self.length == z.shape[-1])
        x = z.copy()
        for i,f in enumerate(self.transforms):
            x[:,i] = f.inverse_transform(x[:,i])
        return x

    def inverse_derivative(self, z: np.array):
        '''
        Computes the point-wise jacobian of the inverse transform. 
        Note that by construction, the transforms are applied to each feature (dimension of x) independently.
        Therefore, the Jacobian is diagonal, and we return a vector.
        
        z is (1,nt)
        J is (nt,)
        '''
        assert(self.length == z.shape[-1])
        J = np.zeros((self.length,))
        for i,f in enumerate(self.transforms):
            J[i] = f.inverse_derivative(z[0,i][None]) 
        return J
