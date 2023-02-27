import numpy as np
import multiprocessing as mp
import os.path
import quantum_states as qs




def generate_quadratures(phases: np.array, path: str, ns: int, xmin: float, xmax: float, rho: np.array):
    QThetaPDF = np.vectorize(qs.x_pdf,excluded=['rho'])
    for theta in phases:
        x_data = random_variate(pdf=QThetaPDF, ns=ns,xmin=xmin, xmax=xmax, theta=theta, rho=rho);
        np.save(f'{path}/X-theta-{theta:.2f}', x_data[0])
        
        
def random_variate(pdf, ns:int, xmin=-1, xmax=1, **par):  
  # Calculates the minimal and maximum values of the PDF in the desired  
  # interval. The rejection method needs these values in order to work  
  # properly.  
    x = np.linspace(xmin, xmax, ns)  
    theta = par['theta']
    rho = par['rho']
    y = np.real(pdf(rho=rho, x=x, theta=theta))  
    pmin= 0.  
    pmax = y.max()  

    # Counters  
    naccept = 0  
    ntrial = 0  

    # Keeps generating numbers until we achieve the desired n  
    ran=[] # output list of random numbers  
    while naccept < ns:  
        x = np.random.uniform(xmin, xmax) # x'  
        y = np.random.uniform(pmin, pmax) # y'  

        if y < np.real(pdf(rho=rho, x=x, theta=theta)):  
            ran.append(x)  
            naccept=naccept+1  
        ntrial+=1  

    ran=np.asarray(ran)  

    return ran, ntrial          
        
        

def run_parrallel(phases,path,ns,rho,xmin,xmax):
    
    processes = []
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(f'{path}/phases', phases)
    for quad in phases:    
        pr=mp.Process(target=generate_quadratures, args=(quad,path,ns,xmin,xmax,rho))
        processes.append(pr)
        pr.start()

    for p in processes:
        p.join()
    
    for p in processes:
        p.close()
        
        
        
def parrallel_gen_quad(rho: np.array, quads: np.array, size_cpu: int, path: str, ns: int, xmin: float, xmax:float):
    """ 
    Function generates quadrature for given  density matrix
    rho is a density matrix
    quads is an array of phase representing a quadrature to generate
    size_cpu is a number of processor cores for parallel execution
    path, where the quadratures are saved
    ns is a number of samples for each quadrature to generate
    xmin is minimal value of quadrature
    xmax is maximum value of quadrature
    """
    x,y = np.shape(rho) 
    if not x ==y:
        raise ValueError('The density matrix is not square matrix')
    
    even_part = np.size(quads) // size_cpu # number of catchments for each process to analyze
    remainder = np.size(quads) % size_cpu # extra catchments if n is not a multiple of size
    
    if remainder > 0:
            bulk = quads[:-remainder]    
            rem  = quads[-remainder:]
            bulk_data = bulk.reshape(size_cpu, even_part)
            rem_data = rem.reshape(np.size(rem),1)
    else:
            bulk = quads
            bulk_data = bulk.reshape(size_cpu, even_part)
            
    run_parrallel(phases=bulk_data, path=path, ns=ns, rho=rho, xmin=xmin, xmax=xmax)
    
    if remainder > 0:
        run_parrallel(phases=rem_data,path=path,ns=ns,rho=rho,xmin=xmin,xmax=xmax)