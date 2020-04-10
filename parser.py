import xml.etree.ElementTree as ElementTree
import tensorflow as tf

class HarmonicBondForce:
	def __init__(self, atom1, atom2, length, k):
		self.atom1 = atom1
		self.atom2 = atom2
		self.length = length
		self.k = k

	@tf.function
	def __call__(self):
		return self.k * (tf.norm(self.atom2.pos - self.atom1.pos) - self.length) ** 2

	def __repr__(self):
		return f'HarmonicBondForce(between {self.atom1.name} and {self.atom2.name}, length is {self.length})'

class HarmonicAngleForce:
	def __init__(self, atom1, atom2, atom3, angle, k):
		self.atom1 = atom1
		self.atom2 = atom2
		self.atom3 = atom3
		self.angle = angle
		self.k = k

	@tf.function
	def __call__(self):
		side1 = self.atom1.pos - self.atom2.pos
		side2 = self.atom3.pos - self.atom2.pos
		cosine_angle = tf.tensordot(side1, side2, 1) / (tf.norm(side1) * tf.norm(side2))
		ang = tf.math.acos(cosine_angle)
		return self.k * (ang - self.angle) ** 2

	def __repr__(self):
		return f'HarmonicAngleForce(between {self.atom1.name}, {self.atom2.name} and {self.atom3.name}, angle is {self.angle})'

class Atom:
	def __init__(self, name, element, type_id, mass, pos=None):
		self.name = name
		self.element = element
		self.type_id = type_id
		self.mass = mass
		if pos is None:
			self.pos = tf.Variable(tf.random.uniform(shape=(3,)))
		elif type(pos) == float or type(pos) == int:
			self.pos = tf.Variable(tf.random.uniform(minval=0 + pos, maxval=1 + pos, shape=(3,)))
		else:
			self.pos = tf.Variable(pos)

	def __repr__(self):
		return f'Atom({self.name}: element {self.element} with mass {self.mass})'

class Residue:
	def __init__(self, name, forcefield, add_hydrogens=False, add_oxygen=False, his_replacement='HID'):
		# parse a mapping from single letter amino acid codes to three letter abbreviations
		mappings = ('ala:A|arg:R|asn:N|asp:D|cys:C|gln:Q|glu:E|gly:G|his:H|ile:I|'
				   + 'leu:L|lys:K|met:M|phe:F|pro:P|ser:S|thr:T|trp:W|tyr:Y|val:V').upper().split('|')
		letter2aa = dict([m.split(':')[::-1] for m in mappings])

		# figure out the 3 letter amino acid abbreviation from the name parameter
		if len(name) == 1:
			self.name = letter2aa[name]
		else:
			self.name = name

		if add_hydrogens and add_oxygen:
			# theoretically it's possible (I think) but the AMBER forcefield doesn't list this directly
			raise ValueError('Can\'t add hydrogens and oxygen to the same residue')

		# Histidine (HIS, H) is either one of HID, HIE or HIP in AMBER
		if self.name == 'HIS':
			self.name = his_replacement

		if add_hydrogens:
			self.name = 'N' + self.name
		if add_oxygen:
			self.name = 'C' + self.name

		# load the forcefield xml and store the root element
		if type(forcefield) == str:
			self.forcefield = ElementTree.parse('forcefields/amber99sb.xml').getroot()
		elif type(forcefield) == ElementTree.ElementTree:
			self.forcefield = forcefield.getroot()
		elif type(forcefield) == ElementTree.Element:
			self.forcefield = forcefield
		else:
			raise ValueError(f'Forcefield type {type(forcefield)} not supported')

		self.atoms = []
		self.bonds = []
		self.external_bond_indices = []
		# load all atomic attributes for this residue from the forecefield and store atomic bonds
		for obj in self.forcefield.find(f'Residues/Residue[@name=\'{self.name}\']'):
			if obj.tag == 'Atom':
				self.atoms.append(self._get_atom(obj))
			elif obj.tag == 'Bond':
				self.bonds.append(self._get_bond(obj))
			elif obj.tag == 'ExternalBond':
				self.external_bond_indices.append(self._get_external_bond(obj))
			else:
				print(f'Unsupported type {obj.type}')

		self.harmonic_bond_forces = []
		# get the harmonic bond forces between atoms
		for bond in self.bonds:
			a1 = bond[0]
			a2 = bond[1]
			search_options = [(a1.name, a2.name), (a2.name, a1.name),
							  (a1.name, a2.element), (a1.element, a2.name),
							  (a1.element, a2.element), (a2.element, a1.element)]
			for option in search_options:
				force = self._get_harmonic_bond_force(*option)
				if force is not None:
					break
			if force is not None:
				self.harmonic_bond_forces.append(HarmonicBondForce(a1, a2, float(force.get('length')), float(force.get('k'))))

		self.harmonic_angle_forces = []
		for i, a1 in enumerate(self.atoms):
			for j, a2 in enumerate(self.atoms[i+1:]):
				for k, a3 in enumerate(self.atoms[i+j+2:]):
					search_options = [(a1.name, a2.name, a3.name), (a3.name, a2.name, a1.name),
									  (a1.name, a2.name, a3.element), (a3.element, a2.name, a1.name),
									  (a1.name, a2.element, a3.name), (a3.name, a2.element, a1.name),
									  (a1.element, a2.name, a3.name), (a3.name, a2.name, a1.element),
									  (a1.element, a2.element, a3.name), (a3.name, a2.element, a1.element),
									  (a1.element, a2.name, a3.element), (a3.element, a2.name, a1.element),
									  (a1.name, a2.element, a3.element), (a3.element, a2.element, a1.name),
									  (a1.element, a2.element, a3.element), (a3.element, a2.element, a1.element)]
					for option in search_options:
						force = self._get_harmonic_angle_force(*option)
						if force is not None:
							break
					if force is not None:
						self.harmonic_angle_forces.append(HarmonicAngleForce(a1, a2, a3, float(force.get('angle')), float(force.get('k'))))

	def _get_atom(self, xml_element):
		# extract the attributes of an atom from the forcefield
		name = xml_element.get('name')
		type_id = int(xml_element.get('type'))
		atom_traits = self.forcefield[0][type_id].attrib
		element = atom_traits['element']
		mass = float(atom_traits['mass'])
		return Atom(name, element, type_id, mass, pos=len(self.atoms))

	def _get_bond(self, xml_element):
		# extract the indices of two bonded atoms from the forcefield
		attribs = xml_element.attrib
		return [self.atoms[int(attribs['from'])], self.atoms[int(attribs['to'])]]

	def _get_external_bond(self, xml_element):
		# extract the index of an atom with an external bond from the forcefield
		return int(xml_element.attrib['from'])

	def _get_harmonic_bond_force(self, name1, name2):
		return self.forcefield.find(f'HarmonicBondForce/Bond[@class1=\'{name1}\'][@class2=\'{name2}\']')

	def _get_harmonic_angle_force(self, name1, name2, name3):
		return self.forcefield.find(f'HarmonicAngleForce/Angle[@class1=\'{name1}\'][@class2=\'{name2}\'][@class3=\'{name3}\']')

	def get_atom_count(self):
		return len(self.atoms)

	def get_bond_count(self):
		return len(self.bonds)

	def get_forces(self):
		return self.harmonic_bond_forces + self.harmonic_angle_forces

	def get_variables(self):
		return [atom.pos for atom in self.atoms]

	def __repr__(self):
		return f'Residue({self.name}: {self.get_atom_count()} atoms, {self.get_bond_count()} bonds)'

class Chain:
	def __init__(self, sequence, forcefield='forcefields/amber99sb.xml'):
		if len(sequence) == 1:
			raise ValueError('Must have at least two amino acids to form a chain')

		self.sequence = sequence

		self.residues = []
		# generate residues from the amino acid sequence
		for i, aa in enumerate(self.sequence):
			self.residues.append(Residue(aa, forcefield, add_hydrogens=(i == 0), add_oxygen=(i == len(self.sequence) - 1)))

		self.external_bonds = []
		# store the atoms which have external bonds, reaching from one residue to another
		for i in range(1, len(self.residues)):
			idx1 = self.residues[i-1].external_bond_indices[-1]
			idx2 = self.residues[i].external_bond_indices[0]
			self.external_bonds.append([self.residues[i-1].atoms[idx1], self.residues[i].atoms[idx2]])

		self.external_harmonic_bond_forces = []
		# get the harmonic bond forces between atoms with external bonds
		for bond in self.external_bonds:
			a1 = bond[0]
			a2 = bond[1]
			search_options = [(a1.name, a2.name), (a2.name, a1.name),
							  (a1.name, a2.element), (a1.element, a2.name),
							  (a1.element, a2.element), (a2.element, a1.element)]
			for option in search_options:
				force = self.residues[0]._get_harmonic_bond_force(*option)
				if force is not None:
					break
			if force is not None:
				self.external_harmonic_bond_forces.append(HarmonicBondForce(a1, a2, float(force.get('length')), float(force.get('k'))))

	def get_atom_count(self):
		return sum([res.get_atom_count() for res in self.residues])

	def get_bond_count(self):
		return sum([res.get_bond_count() for res in self.residues]) + len(self.external_bonds)

	def get_atoms(self):
		return sum([res.atoms for res in self.residues], [])

	def get_bonds(self):
		return sum([res.bonds for res in self.residues], []) + self.external_bonds

	def get_forces(self):
		return sum([res.get_forces() for res in self.residues], []) + self.external_harmonic_bond_forces

	def get_energy(self, normalize=False):
		if normalize:
			ks = sum([force.k for force in self.get_forces()])
		else:
			ks = 1
		return sum([force() for force in self.get_forces()]) / ks

	def get_variables(self):
		return sum([res.get_variables() for res in self.residues], [])

	def __repr__(self):
		return f'Chain({len(self.residues)} residues, {self.get_atom_count()} atoms, {self.get_bond_count()} bonds)'

if __name__ == '__main__':
	chain = Chain('QED', 'forcefields/amber99sb.xml')
	print(chain)
	print(chain.get_energy())
