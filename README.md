# CSQPT

# Coherent State Quantum Process Tomography (csQPT)
- The code in the project implements algorithms of [maximum likelihood coherent state  quantum process tomography](https://doi.org/10.1088/1367-2630/14/10/105021) and [maximum likelihood quantum state tomography (QST)](https://doi.org/10.1088/1464-4266/6/6/014)  in Python. 
- The path to the folders containing the recorded quadratures and the used input density matrix is provided as an input to the algorithm.
- The algorithm outputs a process tensor in Fock space that represents the maximum likelihood transformation of input coherent state being converted to the observed quadratures for csQPT and density matrix for QST.
- The algorithms are parallelized by performing the likelihood calculation for a set of individual quadratures on each CPU core.
# Structure of the project
The src folder contains the main scripts and libraries for CSQPT  and QST.


- csqpt.py is a script that reconstructs quantum process tensor from provided input quantum states and output quadratures.
- state_tomography.py is a script that reconstructs density matrix of the quantum state from provided quadratures.
- quantum_process.py is a small library for describing properties of single-mode optical bosonic  processes: identity, phase shift, attenuation, displacement.  
- quantum_states.py is a small library for describing properties of single-mode optical bosonic states: Fock, Coherent, Thermal states.  
- gen_quad_sample.py is a tunable script that generates a sample of quadratures for given quantum state.
- barplot.py is a script for generating bar plots that visualise density matrices or process tensors.

To see a demonstration, see

- CSQPT-example.ipynb for quantum process tomography
- QST-example.ipynb for quantum state tomography
- Folders csqpt-data and qst-data contains quadrature structure and data that are used  in the examples.
