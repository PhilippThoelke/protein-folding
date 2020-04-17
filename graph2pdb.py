import pickle
import sys
import os

if len(sys.argv) < 2:
	print('Please specify a file to convert.')
	exit()

path = sys.argv[1]
with open(path, 'rb') as file:
	graph, _, _ = pickle.load(file)

with open(path + '.pdb', 'w') as file:
	for i, v in enumerate(graph.vs):
		x, y, z = v['pos'] * 10 # nanometer to Angstrom
		line = f'{"ATOM":6}{i+1:5} {v["name"]:^4}{v["residue"]:>4} A{v["residue_index"]:4}    {x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{0:6.2f}          {v["element"]:2}  \n'
		file.write(line)
