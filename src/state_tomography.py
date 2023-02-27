import multiprocessing as mp
import numpy as np
import csqpt as qpt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import quantum_states as qs
import argparse
import pickle as pkl
import os


def renorm_trace(rho):
    return rho/np.trace(rho)



def Roper(rho, q_bins, Freq, Thetas):
    # Construct R operator 
    # Thetas is N dimensional array of used quadratures
    # Freq is an 2D array contaning Number of times of acquired quadratures for given Theta
    # X is an 2D array contaning set of acquired quadratures for given Theta
    Xstep = np.abs(q_bins[1]-q_bins[0])
    N = np.size(Freq[0])
    Hdim = rho.shape[0]
    R = np.zeros((Hdim,Hdim),dtype = np.complex)
    
    for t in np.arange(0,np.size(Thetas)):
        theta = Thetas[t]
        for i in np.arange(0,N):
            if Freq[t,i]>0:
                # Find probability for acquring given quadrature value
                prob = np.real(qs.x_pdf(rho = rho, x = q_bins[i], theta = theta))
                if prob > 0:
                    proj = qpt.q_proj(q_bins[i], theta, h_dim=Hdim)
                    R += Freq[t,i]*proj/prob
            # Construct a projector |X_j,ϴ_j><X_j,ϴ_j| for given value of quadrature:
    return R        







def state_tomography(rho, q_bins, freq_data, Thetas, message_queue):
    op = Roper(rho = rho, q_bins = q_bins, Freq = freq_data, Thetas = Thetas)
    message_queue.put(op)
    
def run_parrallel(path, bulk_thetas, n_bins,rho, xmin, xmax, message_queue,ST=True):
    
    processes = []
    q_bins = np.linspace(xmin, xmax, n_bins+1)
    for thetas in bulk_thetas:
        i=0 
        freq_data = np.zeros((np.size(thetas),n_bins))
        for theta in thetas:
            #X=np.genfromtxt(f'{folderName}/X-theta-{theta:.2f}.out', delimiter=",")
            q_data = np.load(f"{path}/X-theta-{theta:.2f}.npy")
            Freq = np.histogram(q_data, bins = q_bins)
            if ST:
                freq_data[i] = (Freq[0]/np.size(q_data))
            else:
                freq_data[i]=Freq[0]
            i+=1
            
        pr = mp.Process(target = state_tomography, args = (rho, q_bins, freq_data, thetas, message_queue,))
        processes.append(pr)
        pr.start()

    for p in processes:
        p.join()
    
    for p in processes:
        p.close()
    
    
    
def main(args):
    
    cpu_size = args.cpu
    
    path = args.path
    
    
    output_file_name = args.out
    
    
    xmin = args.xmin
    xmax = args.xmax
    h_dim = args.h_dim
    it = args.it
    n_bins = args.n_bins

    
    
    Thetas = np.load(f"{path}/phases.npy").flatten()
    
    
    n = np.size(Thetas)
    
    part_even = np.size(Thetas) // cpu_size # number of catchments for each process to analyze
    remainder = np.size(Thetas) % cpu_size # extra catchments if n is not a multiple of size
    

    if remainder > 0:

        bulk = Thetas[:-remainder]    
        rem = Thetas[-remainder:]


        bulk_data = bulk.reshape(cpu_size,part_even)
        rem_data = rem.reshape(np.size(rem),1)

        
    else:
        
        bulk = Thetas
        bulk_data = bulk.reshape(cpu_size, part_even)

    
    
    vacuum_state =  qs.FockState(n = [1,0], N = h_dim)

    rho = vacuum_state.get_rho()

    message_queue  = mp.Queue()

    start = time.time()

    
    
    for k in range(it):
        run_parrallel(path=path, bulk_thetas = bulk_data, n_bins = n_bins, rho = rho, xmin =xmin,xmax = xmax, message_queue = message_queue, ST=True)
        if remainder > 0:
            run_parrallel(path=path, bulk_thetas = rem_data, n_bins = n_bins, rho = rho, xmin = xmin,xmax = xmax, message_queue = message_queue, ST=True)
        r_operator = 0
        i = 0
        while not message_queue.empty():
            i+=1
            r_operator += message_queue.get()          

        rho = renorm_trace(np.dot(np.dot(r_operator, rho), r_operator))
    
    
    
    print("rho=",rho)
    
    
    
    np.save(f"{output_file_name}", rho)
    print(f"Reconstructed density matrix is saved as {output_file_name}.npy")
    
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cpu", dest="cpu", type=int, default=8, help="Number of cores for computing")
    parser.add_argument(
        "-path", dest="path", type=str, help="Directory containing Quadratures")
    parser.add_argument('-xmin', dest='xmin', type=float, default=-3.0, help='negative Xcutoff')
    parser.add_argument('-xmax', dest='xmax', type=float, default=3.0, help='positive Xcutoff')
    parser.add_argument('-bins', dest='n_bins', type=int, default=200, help='number of bins')
    parser.add_argument('-h_dim', dest='h_dim', type=int, default=4, help='Hilbert space dimensions in MaxLik')
    parser.add_argument('-it', type=int, default=10, help='Number of iterations in MaxLik')
    parser.add_argument(
        "-out", dest="out", type=str, default="reconstructed-density-matrix", help="output file name for reconstructed tensor"
    )
    
    
    args = parser.parse_args()
    main(args) 