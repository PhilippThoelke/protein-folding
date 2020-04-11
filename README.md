# Protein folding

It is possible to predict the tertiary/quaternary structure of proteins using so called molecular dynamics simulations. Unfortunately, these simulations are computationally very expensive and take a lot of time to reach meaningful results, especially for larger proteins. The goal of this project is to reduce the computational power needed to run protein structure predictions, building upon the idea of a molecular dynamics simulation.

### The project is currently at a very early stage!

Implemented so far:
- Basic molecular dynamics simulation using OpenMM, theoretically capable of folding a protein, i.e. finding the native state, given the protein's primary structure.
- Non gradient based genetic algorithm approach, which optimizes the angles between neighbouring amino acids inside a molecular dynamics simulation with OpenMM. Evaluation is based on the protein's potential energy in the thermodynamic system. It currently does not work properly because the molecular dynamics simulation runs into numerical problems due to large forces applied to the system.
- Gradient descent optimization, minimizing the spatial error of covalent atomic bonds and non-covalent atomic forces. Bonds and forces are parsed from an AMBER forcefield. The gradient is computed on the locations of individual atoms in three-dimensional space using TensorFlow. The minimization term is still missing some essential forces defined in the forcefield, causing amino acids to collapse into each other.
