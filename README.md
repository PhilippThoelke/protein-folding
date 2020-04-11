# protein-folding
## The project is currently at a very early stage!

Implemented so far:
- Non gradient based genetic algorithm approach, which optimizes the angles between neighbouring amino acids inside a molecular dynamics simulation with OpenMM. Evaluation is based on the protein's potential energy in the thermodynamic system. It currently does not work properly because the molecular dynamics simulation runs into numerical problems due to large forces applied to the system.
- Gradient descent optimization, minimizing the spatial error of covalent atomic bonds and non-covalent atomic forces. Bonds and forces are parsed from an AMBER forcefield. The gradient is computed on the locations of individual atoms in three-dimensional space using TensorFlow. The minimization term is still missing some essential forces defined in the forcefield, causing amino acids to collapse into each other.
