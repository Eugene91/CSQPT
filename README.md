# CSQPT

# Coherent State Quantum Process Tomography (csQPT)
- The code in the project implements algorithms of [maximum likelihood coherent state  quantum process tomography](https://doi.org/10.1088/1367-2630/14/10/105021) and [maximum likelihood quantum state tomography (QST)](https://doi.org/10.1088/1464-4266/6/6/014)  in Python. 
- At the input for algorithm the path to folders contating set of the recorded quadratures and used input density matrix is provided. 
- The algorithm outputs a process tensor in Fock space that represents the maximum likelihood transformation of input coherent state being converted to the observed quadratures for csQPT and density matrix for QST.
- The algorithms are parallized by running calculation of likelihhood data of a set of individual quadrature  on individual CPU core.    
# Structure of the project
Folder src contains major scripts and libraries for quantum process tomography

- csqpt.py is script that reconstructs quantum process tensor from provided input quantum states and output quadratures.
- state_tomography.py is script that reconstructs density matrix of the quantum state from provided  quadratures.
- quantum_process.py is small library for describing properties of single-mode optical bosonic  processes: identity, phase shift, attenuation, displacement.  
- quantum_states.py is a small library for describing properties of single-mode optical bosonic  states: Fock, Coherent, Thermal states.  
- gen_quad_sample.py is a tunable script that generates a sample of quadratures for given quantum state.
- barplot.py is script to generate bar plots for visualizing density matrix or process tensors

To see a demonstration, see

- CSQPT-example.ipynb for quantum process tomography
- QST-example.ipynb for quantum state tomography
- Folders csqpt-data and qst-data contains quadrature structure and data that are used  in the examples.
