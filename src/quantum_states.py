import scipy.linalg as linalg
import numpy as np
from scipy import special






def asymp_factorial(n):
    if n < 5:
        return np.double(np.math.factorial(n))
    else:
        return np.sqrt(2*np.pi*n)*(1+1/(12*n))*(n/np.e)**n


    
def x_pdf(rho, x, theta):
    N = rho.shape[0]
    HPolys = np.array([])
    # Create an array of Quadrature projectors polynomials 
    for k in np.arange(0,N):
        Hk = special.hermite(k, monic=False)
        XO = np.exp(k*theta*1.j)*(2/np.pi)**(1/4)*Hk(np.sqrt(2)*x)*np.exp(-x**2)/(np.sqrt(asymp_factorial(k)*2**k))
        HPolys = np.append(HPolys,XO)
    
    POVM = np.outer(np.conj(HPolys),HPolys)
    total = np.sum(POVM*rho)
    return total
    
    
class State:
    def __init__(self):
        pass
    
    def get_rho(self):
        return self.rho/np.trace(self.rho)
    
    def save_rho(self, name):
        np.save(name, self.rho)
    
        
    

class CoherentState(State):
    def __init__(self, alpha,phi,N):
        self.alpha = alpha
        self.phi = phi
        self.psi = np.zeros(N,dtype=np.complex_)
        self.psi[0] = 1
        for i in np.arange(1,N):
            self.psi[i] = np.exp(i*phi*1.j)*(alpha**i)/(np.sqrt(asymp_factorial(i)))
        self.psi*=np.exp(-alpha**(2)/2)    
        #print(self.psi)
        self.rho=np.outer(np.conj(self.psi),self.psi)
        #print(self.rho)
    
    def get_alpha(self):
        return self.alpha
    
    def get_phi(self):
        return self.phi
    
    def get_phi_deg(self):
        return self.phi*180/np.pi
    
        
    def get_psi(self):
        return self.psi
    

    
class ThermalState:
    def __init__(self,T,N):
        # Plank constant over Boltzman constant
        self.T = T
        hk = 4.8*10**-11
        # Frequency is 6 GHz
        omega = 6 * 10**9
        n = np.arange(0,N)
        self.rho = (1-np.exp(-2*hk*omega/T))*np.diag(np.around(np.exp(-hk*n*omega/T),decimals=5))
            
    def get_T(self):
        return self.T
    
    

class FockState:
    def __init__(self, n,N):
        self.n = n/np.linalg.norm(n)
        self.psi = np.zeros(N,dtype=np.complex_)
        for k in np.arange(0,np.size(self.n)):
            self.psi[k] = self.n[k]
            
        #print(self.psi)
        self.rho=np.outer(self.psi,np.conj(self.psi))
        #print(self.rho)
    
    def get_rho(self):
        return self.rho/np.sum(np.diag(self.rho))
    
    def get_n(self):
        return self.n