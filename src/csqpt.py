import multiprocessing as mp
import numpy as np
from scipy import special
import quantum_states as qs
import os
import quantum_process as qp
import scipy.linalg
import argparse
import time
import functools


def normalization_factor(state: np.array, cp_map: np.array):
    # Calculate normalization factor for a state after an action of cp_map
    AfterProcess = np.dot(cp_map, state)
    return np.real(np.trace(AfterProcess))

#@functools.lru_cache(maxsize = None)
def q_proj(x: float, theta: float, h_dim: int) -> np.array:
    # Construct projector for given quadrature value.
    # E.g. |m><m|x,theta><x,theta|n><n|.
    # x is value of quadrature.
    # theta is a phase of quadrature.
    # h_dim is Hilbert dimension cut-off.
    h_pol = np.array([])

    for k in np.arange(0, h_dim):
        h_k = special.hermite(k, monic=False)
        x_o = (
            np.exp(k * theta * 1j)
            * (2 / np.pi) ** (1 / 4)
            * h_k(np.sqrt(2) * x)
            * np.exp(-(x**2))
            / (np.sqrt(qs.asymp_factorial(k) * 2**k))
        )
        h_pol = np.append(h_pol, x_o)

    proj = np.outer(h_pol, np.conj(h_pol))
    return proj

def proj(m:int, Hdims:int) -> np.array :
    X = np.zeros((Hdims,Hdims))
    X[m,m] = 1 
    return X



def mask_in(Eop: np.array, Hdims: int) -> np.array:
    indices = np.indices((Hdims, Hdims, Hdims, Hdims))
    m, n, j, k = indices[0], indices[1], indices[2], indices[3]
    mask = (m - n) == (j - k)
    kronecker1 = np.eye(Hdims)[:, :, np.newaxis, np.newaxis] * np.eye(Hdims)[np.newaxis, np.newaxis, :, :]
    kronecker2 = np.eye(Hdims)[:, :, np.newaxis, np.newaxis] * np.eye(Hdims)[np.newaxis, np.newaxis, :, :]
    res = np.sum(mask[:, :, :, :, np.newaxis, np.newaxis] * np.dot(kronecker1[m, j], np.dot(Eop, kronecker2[n, k])), axis=(0, 1, 2, 3))
    return res


def mask_in(Eop: np.array, Hdims: int) -> np.array:
    res = 0
    for k in np.arange(Hdims):
        for j in np.arange(Hdims):
            for m in np.arange(Hdims):
                for n in np.arange(Hdims):
                    if (m - n) == (j - k):
                        res += np.dot(
                            np.kron(proj(m, Hdims), proj(j, Hdims)),
                            np.dot(Eop, np.kron(proj(n, Hdims), proj(k, Hdims))),
                        )

    return res


def r_oper(
    rho: np.array,
    q_bins: np.array,
    freq: np.array,
    thetas: np.array,
    cp_map: np.array,
    PI: bool,
) -> np.array:
    # Construct R operator for  single thread
    # rho used density matrix
    # thetas is N dimensional array of used quadratures
    # freq is an 2D array contaning Number of times of acquired quadratures for given quadrature theta
    # X is an 2D array contaning set of acquired quadratures for given quadrature theta
    # cp_map is completly positive map representing a quantum process
    # PI is bool variable for phase invariant process
    N = np.size(freq[0])
    Hdims = rho.shape[0]
    tr_rho = np.transpose(rho)
    R = np.zeros((Hdims**2, Hdims**2), dtype=np.complex128)
    for t in np.arange(np.size(thetas)):
        theta = thetas[t]
        for i in np.arange(0, N):
            if freq[t, i] > 0:
                # Find probability for acquring given quadrature value
                rp = np.kron(tr_rho, q_proj(x=q_bins[i], theta=theta, h_dim=Hdims))
                if PI:
                    rp = mask_in(Eop=rp, Hdims=Hdims)
                # Normalization after an action of cp map
                prob = normalization_factor(state=rp, cp_map=cp_map)
                if prob > 0:
                    R += freq[t, i] * rp / prob
            # Construct a projector rho |X_j,ϴ_j><X_j,ϴ_j| for given value of quadrature:
    return R


def process_tomography(rho, q_bins, freq_data, thetas, cp_map, PI, messageQueue):
    op = r_oper(
        rho=rho, q_bins=q_bins, freq=freq_data, thetas=thetas, cp_map=cp_map, PI=PI
    )
    messageQueue.put(op)


def parallel_process_tomograpy(
    bulk_phases: np.array,
    dir_name: list,
    n_bins: int,
    rho: np.array,
    xmin: float,
    xmax: float,
    message_queue: mp.Queue,
    cp_map: np.array,
    PI: bool,
):
    processes = []
    q_bins = np.linspace(xmin, xmax, n_bins + 1)

    for thetas in bulk_phases:
        i = 0
        freq_data = np.zeros((np.size(thetas), n_bins))
        for i, theta in np.ndenumerate(thetas):
            # load a statistics for given quadrature
            q_data = np.load(f"{dir_name}/X-theta-{theta:.2f}.npy")
            # calculate the probability distribution
            freq = np.histogram(q_data, bins=q_bins)
            freq_data[i] = freq[0]

        # run parralel tomography
        pr = mp.Process(
            target=process_tomography,
            args=(
                rho,
                q_bins,
                freq_data,
                thetas,
                cp_map,
                PI,
                message_queue),
        )
        processes.append(pr)
        pr.start()

    for p in processes:
        p.join()

    for p in processes:
        p.close()


def parallel_r_operator(
    dir_list: list,
    Eop: np.array,
    xmin: float,
    xmax: float,
    n_bins: int,
    size_cpu: int,
    PI: bool,
):
    # parallely calculates the R super operator.
    # dir_list is a list of files containg input density matrices
    # and quadratures
    Hdims = int(np.sqrt(Eop.shape[0]))
    r_processor = np.kron(
        np.zeros((Hdims, Hdims), dtype=np.complex128),
        np.zeros((Hdims, Hdims), dtype=np.complex128),
    )

    for dir_name in dir_list:
        rho = np.load(f"{dir_name}/rho.npy")
        phases = np.load(f"{dir_name}/phases.npy").flatten()

        if not (rho.shape[0] == Hdims):
            raise ValueError(
                f"Choosen dimension {Hdims} is not aligned with a dimension of density matrix {rho.shape[0]}"
            )


        # introduce parallel execuation of the maxlink algorithm

        part_even = (
            np.size(phases) // size_cpu
        )  # number of catchments for each process to analyze
        remainder = (
            np.size(phases) % size_cpu
        )  # extra catchments if n is not a multiple of size

        if remainder > 0:
            bulk_phases = phases[:-remainder]
            rem_phases = phases[-remainder:]

            bulk_phases = bulk_phases.reshape(size_cpu, part_even)
            rem_phases = rem_phases.reshape(np.size(rem), 1)

        else:
            bulk = phases
            bulk_phases = bulk.reshape(size_cpu, part_even)

        message_queue = mp.Queue()
        parallel_process_tomograpy(
            bulk_phases=bulk_phases,
            dir_name=dir_name,
            n_bins=n_bins,
            rho=rho,
            cp_map=Eop,
            xmin=xmin,
            xmax=xmax,
            PI=PI,
            message_queue=message_queue,
        )

        if remainder > 0:
            parallel_process_tomograpy(
                bulk_phases=rem_phases,
                dir_name=dir_name,
                n_bins=n_bins,
                rho=rho,
                cp_map=Eop,
                xmin=xmin,
                xmax=xmax,
                PI=PI,
                message_queue=message_queue,
            )

        R = np.zeros((Hdims**2, Hdims**2), dtype=np.complex128)

        while not message_queue.empty():
            R += message_queue.get()

        message_queue.close()
        r_processor += R

    return r_processor


def str_2_bool(v):
    # convert string argument from string 
    str1 = v.lower()
    return v.lower() in ('true')

def main(args):
    start = time.perf_counter()

    path = args.path
    h_dim = args.hdim
    xmin = args.xmin
    xmax = args.xmax
    number_of_iterations = args.it
    Nbins = args.bins
    size_cpu = args.cpu
    PI = str_2_bool(args.pi)
    output_file_name = args.out

    dir_list = []
    for dir_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir_name)) and  not (dir_name[0]=='.'):
            dir_list.append(f"{path}/{dir_name}")

    print("Directories to check:\n", dir_list)
    print("Phase Invariant:", PI)

    # Initial seed operator

    q_proc = qp.QAttenuation(Hdims=h_dim, eta=0.95)

    process_tensor = q_proc.get_cp_map()

    new_process_tensor = parallel_r_operator(
        dir_list=dir_list,
        Eop=process_tensor,
        xmin=xmin,
        xmax=xmax,
        n_bins=Nbins,
        size_cpu=size_cpu,
        PI=PI,
    )


    for i in range(number_of_iterations):
        l1 = qp.partial_trace(
            rho=np.dot(new_process_tensor, np.dot(process_tensor, new_process_tensor)),
            Hdim1=h_dim,
            Hdim2=h_dim,
        )

        ls = scipy.linalg.sqrtm(l1)

        lambda_matr = np.kron(ls, np.diag(np.ones(h_dim)))

        inv_lambda_matr = np.linalg.inv(lambda_matr)

        process_tensor_intr = np.dot(
            process_tensor, np.dot(new_process_tensor, inv_lambda_matr)
        )

        process_tensor = np.dot(
            np.dot(inv_lambda_matr, new_process_tensor), process_tensor_intr
        )

        new_process_tensor = parallel_r_operator(
            dir_list=dir_list,
            Eop=process_tensor,
            xmin=xmin,
            xmax=xmax,
            n_bins=Nbins,
            size_cpu=size_cpu,
            PI=PI,
        )

        print(f"Iteration #{i+1} is finished")

    np.save(f"{output_file_name}", process_tensor)

    print(f"Reconstructed process tensor is saved as {output_file_name}.npy")

    end = time.perf_counter()
    print(f"Execution time is {end - start:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cpu", dest="cpu", type=int, default=8, help="Number of cores for computing")
    parser.add_argument(
        "-path", dest="path", type=str, help="Directory containing Quadratures")
    parser.add_argument(
        "-pi",
        dest="pi",
        choices=["true", "false"],
        default="false",
        help="Is process phase invariant?",
    )
    parser.add_argument(
        "-xmin", dest="xmin", type=float, default=-3.0, help="Negative Xcutoff"
    )
    parser.add_argument(
        "-xmax", dest="xmax", type=float, default=3.0, help="Positive Xcutoff"
    )
    parser.add_argument(
        "-bins",
        dest="bins",
        type=int,
        default=200,
        help="Number of discretization bins",
    )
    parser.add_argument(
        "-hdim", dest="hdim", type=int, default=4, help="Hilbert space dimensions"
    )
    parser.add_argument(
        "-it", dest="it", type=int, default=10, help="Number of iterations in MaxLik"
    )
    parser.add_argument(
        "-out", dest="out", type=str, default="reconstructed-process-tensor", help="output file name for reconstructed tensor"
    )

    args = parser.parse_args()
    main(args)
