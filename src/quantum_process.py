import numpy as np
import scipy.special as special

def partial_trace(rho, Hdim1, Hdim2, axis1=1, axis2=3):
    # calculate partial trace over a single particle from composite two particle
    # Hilbert space with  dimension
    # Hdim1 is dimension of a vector in First Hilbert space
    # Hdim2 is dimension of a vector in Second Hilbert space
    # axis1 = 1 axis2 =3 for  trace over Second Hilbert space
    # axis1 = 0 axis2 =2 for  trace over First Hilbert space

    rho_tensor=rho.reshape([Hdim1,Hdim2, Hdim1,Hdim2])
    
    return np.trace(rho_tensor, axis1=axis1, axis2=axis2)






def asymp_factorial(n):
    return np.math.factorial(n)
#     if n < 5:
#         return np.double(np.math.factorial(n))
#     else:
#         return np.sqrt(2*np.pi*n)*(1+1/(12*n))*(n/np.e)**n


class QProcess:
    
    Hdims: int
    pr_tensor: np.array
    cp_map: np.array
    pr_tensor: np.array    
    
    def __init__(self, Hdims):
        self.Hdims = Hdims
        self.pr_tensor = np.zeros((Hdims**2,Hdims**2), np.complex128)
        self.cp_map = np.zeros((Hdims**2,Hdims**2), np.complex128)
        self.pr_tensor = self.pr_tensor.reshape((Hdims,Hdims,Hdims,Hdims))
    
    def get_tensor(self):
        return self.pr_tensor
    
    def get_diag_tensor(self):
        diag_tensor = np.zeros((self.Hdims,self.Hdims),np.complex)
        for n in range(self.Hdims):
            for m in range(self.Hdims):
                diag_tensor[m,n] = self.pr_tensor[m,m,n,n]
                
        return diag_tensor
        
    
    def construct_cp_map(self):
        for k in np.arange(0,self.Hdims):
            for j in np.arange(0,self.Hdims):
                for m in np.arange(0,self.Hdims):
                    for n in np.arange(0,self.Hdims):
                        MN = np.zeros((self.Hdims,self.Hdims))
                        MN[m,n] = 1
                        JK = np.zeros((self.Hdims,self.Hdims))
                        JK[j,k] = 1
                        self.cp_map+= self.pr_tensor[m,n,j,k]*np.kron(MN,JK)
                        
                        
    def get_cp_map(self):
        return self.cp_map
    
    def operation(self, rho):
    
        Hdims = rho.shape[0]
        if not (Hdims == self.Hdims):
            raise ValueError(f'Wrong dimension of the density matrix: {Hdims}, the tensor is constructed for {self.Hdims}')
        # Composite state of "out" and "in" density matrices
        state = np.kron(rho,np.diag(np.ones(Hdims)))
        # convolution of the state with quantum operator
        rho_total = np.dot(self.cp_map,np.transpose(state))
        # partial trace over "in"  density matrix
        rho_out = partial_trace(rho=rho_total,Hdim1=Hdims,Hdim2=Hdims,axis1=0, axis2=2)
    
        return rho_out
    
    
    
class QIdentity(QProcess):
    pass
    def __init__(self,Hdims):
        super().__init__(Hdims)
        self.pr_tensor = np.diag(np.ones(Hdims**2))
        self.pr_tensor= self.pr_tensor.reshape((Hdims,Hdims,Hdims,Hdims))
        self.construct_cp_map()
    


class QPhaseShift(QProcess):
    pass
    def __init__(self,Hdims,phi):
        super().__init__(Hdims)
        self.phi=phi
        for j in np.arange(0,self.Hdims):
                for n in np.arange(0,self.Hdims):
                     for k in np.arange(0,self.Hdims):
                        for m in np.arange(0,self.Hdims):
                            self.pr_tensor[m,n,j,k]= self.t_comp(self.phi,m,n,j,k)
        self.construct_cp_map()
        
        
    def t_comp(self,phi,m,n,j,k):
        if m == j and n==k :
            return np.exp(-1.j*(j-k)*phi)
        else:
            return 0
        

        
    
class QAttenuation(QProcess):
    pass
    def __init__(self,Hdims,eta):
        super().__init__(Hdims)
        self.eta = eta
        for j in np.arange(0,self.Hdims):
                for n in np.arange(0,self.Hdims):
                     for k in np.arange(0,self.Hdims):
                        for m in np.arange(0,self.Hdims):
                            self.pr_tensor[m,n,j,k] = self.t_comp(self.eta,j,k,m,n)
        self.construct_cp_map()

        
        
        
    def t_comp(self,eta,j,k,m,n):
        if (m - j) == (n-k) and m >= j and n >= k:
            return np.sqrt((asymp_factorial(m)*asymp_factorial(n))/(asymp_factorial(j)*asymp_factorial(k)))*( (1-eta)**(m-j)*np.sqrt(eta)**(j+k) )/asymp_factorial(m-j)
        else:
            return 0
        
        

class QDisplacement(QProcess):
    pass
    def __init__(self,Hdims,beta,phi):
        super().__init__(Hdims)
        self.beta = beta
        self.phi = phi
        for j in np.arange(0,self.Hdims):
                for n in np.arange(0,self.Hdims):
                     for k in np.arange(0,self.Hdims):
                        for m in np.arange(0,self.Hdims):
                            self.pr_tensor[m,n,j,k] = self.t_comp(j,k,m,n)
        self.construct_cp_map()

        
        
        
    def t_comp(self,j,k,m,n):
        if m+n-k-j >=0 :
            return np.exp(-np.abs(self.beta)**2)*(np.abs(self.beta)**(m+n-j-k))*((-1)**(m+n-k-j)*special.binom(m+n-j-k,n-k)*np.exp(1.j*np.radians(self.phi)*(n-k-m+j))*np.sqrt(special.factorial(m)*special.factorial(n)) )/(special.factorial(m+n-k-j)*np.sqrt(special.factorial(j)*special.factorial(k)))
        else:
            return 0        
