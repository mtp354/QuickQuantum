###########################            QuickQuantum              ###################################
"""
A handy library for basic quantum computation. Constructed by Matthew Prest and Nigel Shen as part of NYU Shanghai's
quantum information research lab.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy


def bitflip(bit, index):  # A simple function that flips the bit at a particular index
    bit = list(bit)
    if bit[index] == '0':
        bit[index] = '1'
    elif bit[index] == '1':
        bit[index] = '0'
    bit = ''.join(bit)
    return bit


def get_bin(x, m):  # gets the binary representation of x with m digits filled.
    return format(x, 'b').zfill(m)


class Ket:  # Defining a class for a Ket state
    def __init__(self, qubit_number, precision = 1):
        self.kets = [self]
        self.weights = [1]
        self.time = sy.symbols('t')
        self.precision = precision
        self.size = qubit_number
        self.states = []  # list of basis states that this state is in a superposition of
        self.amplitudes = []  # lists the probability amplitudes of each respective state
        self.well_order = False  # whether or not the state is formatted in a standard order
        self.amplitude_instance = []
        self.evolutions = 0
        self.circuit = Circuit(self.size)

    def __repr__(self):
        return "<number of qubits:%s, precision:%s>" % (self.size, self.precision)

    def __str__(self):  # prints the state using ket notation e.g. |0010>
        composition = []
        if self.evolutions == 0:
            for k in range(len(self.states)):
                part = str(np.round(self.amplitudes[k], 2)) + '|' + self.states[k] + '〉'
                composition.append(part)

        if self.evolutions > 0:
            for k in range(len(self.states)):
                part = str(np.round(self.amplitude_instance[k], 2)) + '|' + self.states[k] + '〉'
                composition.append(part)

        composition = ' + '.join(composition)
        return composition

    def add_term(self, qubit_string, amplitude):  # Add a new term to the superposition state
        self.states.append(qubit_string)
        self.amplitudes.append(amplitude)
        self.well_order = False

    def time_evolve(self, stop_time):  # Evolve the state to a particular time point
        self.amplitude_instance = self.amplitudes.copy()
        self.evolutions = self.evolutions + 1
        for i in range(len(self.amplitude_instance)):
                self.amplitude_instance[i] = self.amplitude_instance[i].subs('t', stop_time)

    def recombine(self):  # reorders and recombines the list of terms
        if not self.well_order:
            vector = vectorize(self)
            self.states = []
            self.amplitudes = []
            for i in range(2 ** self.size):
                if vector[i][0] != 0:
                    self.states.append(get_bin(i, self.size))
                    self.amplitudes.append(vector[i][0])
        self.well_order=True

    def normalize(self):  # Renormalize the probability amplitudes
        self.recombine()
        current_sum = 0
        for item in self.amplitudes:
            current_sum = current_sum + (np.abs(item)) ** 2
        for i in range(len(self.amplitudes)):
            self.amplitudes[i] = self.amplitudes[i] / np.sqrt(current_sum)


def random_state_generator(n, precision = 1):  # makes a completely random state vector for n qubits
    x = Ket(n, precision)
    random_directions = np.random.normal(size = 2 ** (n + 1))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    for i in range(2 ** n):
        amp = complex(random_directions[i], random_directions[i + 2 ** n])
        x.add_term(str(get_bin(i, n)), amp)
    x.well_order = True
    return x


def inner_product(state1, state2):  # Outputs the inner product of two states
    vector1 = vectorize(state1)
    vector2 = np.conjugate(vectorize(state2))
    output = np.inner(vector1, vector2)
    return output


def tensor_product(state1, state2):  # combining two states via tensor product
    new_n = state1.size + state2.size
    if isinstance(state1, Ket) and isinstance(state2, Ket):
        output = Ket(new_n, state1.precision)
        # now we have to populate the new state using the input states
        for i in range(len(state1.states)):
            for j in range(len(state2.states)):
                output.add_term(state1.states[i] + state2.states[j], state1.amplitudes[i] * state2.amplitudes[j])
    else:
        output = Mixed_State(new_n)
        for i in range(len(state1.kets)):
            for j in range(len(state2.kets)):
                output.add_state(tensor_product(state1.kets[i], state2.kets[j]), state1.weights[i] * state2.weights[j])
    return output


def full_readout(in_state, bits = ''):  # measuring in the logical basis a full readout
    luck = np.random.choice(in_state.kets, 1, False, in_state.weights)[0]
    prob_list = [np.abs(x) ** 2 for x in luck.amplitudes]
    luck = np.random.choice(luck.states, 1, False, prob_list)[0]
    output = Ket(in_state.size)
    output.add_term(luck, 1)
    output.well_order = True
    return output


def measure(in_state, t_index, c_index = '', bits = ''):  # Measurement of a single qubit
    luck = np.random.choice(in_state.kets, 1, False, in_state.weights)[0]
    prob_list = [np.abs(x) ** 2 for x in luck.amplitudes]
    luck = np.random.choice(luck.states, 1, False, prob_list)[0][t_index]
    if c_index != '' and bits != '':
        bits.bit_change(c_index, luck)
    for item in in_state.kets:
        removal_list = []
        for i in range(len(item.states)):
            if item.states[i][t_index] != luck:
                removal_list.append(i)
        for index in sorted(removal_list, reverse = True):
            item.amplitudes.pop(index)
            item.states.pop(index)
    in_state.normalize()
    return in_state


def vectorize(in_state):  # Transforms an input ket state and outputs the column vector
    vector = np.zeros((2 ** in_state.size, 1), dtype=complex)
    for i in range(len(in_state.states)):
        index = int(in_state.states[i], 2)
        vector[index][0] = vector[index][0] + in_state.amplitudes[i]
    return vector


def get_substate(in_state, start, end):  # Returns the substate of a Ket
    if isinstance(in_state, Ket):
        output = Ket(end - start, in_state.precision)
        for i in range(len(in_state.states)):
            output.states.append(in_state.states[i][start:end])
            output.amplitudes.append(in_state.amplitudes[i])
        output.recombine()
    elif isinstance(in_state, Mixed_State):
        output = Mixed_State(end - start)
        output.weights = in_state.weights.copy()
        for item in in_state.kets:
            output.kets.append(get_substate(item, start, end))
    return output


def bell_state_generator(n, precision):  # Generates a maximally entangled state
    bit_string = ''
    for i in range(2 * n):
        bit_string = bit_string + '0'
    bell_state = Ket(2 * n, precision)
    bell_state.add_term(bit_string, 1.0)
    for k in range(n):
        bell_state = hadamard(bell_state, k)
        bell_state = cnot(bell_state, k, k + n)
    return bell_state


def get_fidelity(state_1, state_2):  # returns the fidelity of two states, a measure of similarity
    if isinstance(state_1, Ket) and isinstance(state_2, Ket):
        vector_1 = vectorize(state_1)
        vector_2 = vectorize(state_2)
        fidelity = np.abs(np.vdot(vector_1, vector_2)) ** 2
    else:
        rho_1 = get_density_matrix(state_1)
        rho_2 = get_density_matrix(state_2)
        part_1 = np.trace(np.dot(rho_1, rho_2))
        part_2 = np.sqrt((1-np.trace(np.dot(rho_1, rho_1)))*(1-np.trace(np.dot(rho_2, rho_2))))
        fidelity = part_1 + part_2
    fidelity = np.real(fidelity)
    return fidelity


def get_Ket(vector, precision=1):  # this function reverses the vectorize process
    qubit_number = int(np.log2(len(vector)))
    state = Ket(qubit_number, precision)
    for i in range(len(vector)):
        if vector[i][0] != 0:
            state.add_term(get_bin(i, qubit_number), vector[i][0])
    return state


def decomposer(uni_operator):  # decomposes a unitary operator into rotations
    a = sy.symbol('a') #alpha
    b = sy.symbol('b') #beta
    d = sy.symbol('d') #delta
    g = sy.symbol('g') #gamma
    soln = sy.solve([sy.exp(1j*(a-b/2-d/2))*sy.cos(g/2)-uni_operator[0][0], sy.exp(1j*(a-b/2+d/2))*sy.sin(g/2)-uni_operator[0][1], sy.exp(1j*(a+b/2-d/2))*sy.sin(g/2)-uni_operator[1][0], sy.exp(1j*(a+b/2+d/2))*sy.cos(g/2)-uni_operator[1][1]], [a, b, d, g])
    return soln


class Mixed_State:  # Defining a class for a mixed state
    def __init__(self, qubit_number):
        self.time = sy.symbols('t')
        self.size = qubit_number
        self.kets = []  # list of kets
        self.weights = []  # lists the probability weights of each ket
        self.evolutions = 0

    # Now we can add either a Ket or another mixed state to the mixed state
    def add_state(self, state, weight = 1):
        for i in range(len(state.kets)):
            self.kets.append(state.kets[i])
            self.weights.append(state.weights[i] * weight)
    # It is not neccesary to create states[] and amplitudes[] lists,
    # we can instead get them by using mixed_state.kets[n].states or mixed_state.kets[n].amplitudes

    def __repr__(self):
        return "<number of kets:%s, number of kets:%s, precision:%s>" % (self.size, len(self.kets), self.precision)

    def __str__(self):
        string_list = []
        for i in range(len(self.kets)):
            string_list.append(str(np.round(self.weights[i],4)) + ' * (' + str(self.kets[i]) + ')')
        return '[' + ',\n'.join(string_list) + ']'

    def recombine(self):
        for item in self.kets:
            item.recombine()

    def normalize(self):
        for item in self.kets:
            item.normalize()


def get_density_matrix(in_state):  # returns the density matrix of a ket or mixed state
    rho = np.zeros((2 ** in_state.size, 2 ** in_state.size), dtype=complex)
    for i in range(len(in_state.kets)):
        vector = np.asmatrix(vectorize(in_state.kets[i]), dtype=complex)
        vector_dagger = vector.H
        rho = rho + in_state.weights[i] * np.dot(vector, vector_dagger)
    return rho


def visualize(state):  # visualizes the amplitudes of a state
    rho = get_density_matrix(state)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8), sharey=True)
    real_imag_list = [rho.real, rho.imag]
    k=0
    for ax in axes.flat:
        im = ax.imshow(real_imag_list[1], vmin=0, vmax=1)
        k = k + 1
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


def pauli_x(in_state, t_index, bits = ''):  # Implements a Pauli X gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            in_state.kets[i].states[j] = bitflip(in_state.kets[i].states[j], t_index)
        in_state.kets[i].well_order = False
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def pauli_y(in_state, t_index, bits = ''): # Implements a Pauli Y gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            in_state.kets[i].states[j] = bitflip(in_state.kets[i].states[j], t_index)
            if in_state.kets[i].states[j][t_index] == "0":
                in_state.kets[i].amplitudes[j] *= (-1j)
            else:
                in_state.kets[i].amplitudes[j] *= (1j)
        in_state.kets[i].well_order = False
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def pauli_z(in_state, t_index, bits = ''): # Implements a Pauli Z gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            if in_state.kets[i].states[j][t_index] == "1":
                in_state.kets[i].amplitudes[j] *= (-1)
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def phase(in_state, t_index, bits = '', ctransposed = False):  # Implements a Phase gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            if in_state.kets[i].states[j][t_index] == "1":
                in_state.amplitudes[i] *= (1j)
                if ctransposed:
                    in_state.amplitudes[i] *= -1
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def phase_change(in_state, t_index, alpha):  # Implements a Phase change gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for item in in_state.kets[i].amplitudes:
            item *= np.exp(1j * alpha)
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def cnot(in_state, c_index, t_index, bits = ''):  # Implements a CNOT gate on a particular pair of qubits
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            if in_state.kets[i].states[j][c_index] == '1':
                in_state.kets[i].states[j] = bitflip(in_state.kets[i].states[j], t_index)
        in_state.kets[i].well_order = False
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def hadamard(in_state, t_index, bits = ''):  # Implements a Hadamard gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            in_state.kets[i].states.append(bitflip(in_state.kets[i].states[j], t_index))
            if in_state.kets[i].states[j][t_index] == '0':
                in_state.kets[i].amplitudes[j] *= 1 / np.sqrt(2)
                in_state.kets[i].amplitudes.append(in_state.kets[i].amplitudes[j])
            else:
                in_state.kets[i].amplitudes[j] *= -1 / np.sqrt(2)
                in_state.kets[i].amplitudes.append(-1 * in_state.kets[i].amplitudes[j])
        in_state.kets[i].well_order = False
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def t(in_state, t_index, bits = '', ctransposed = False):  # Implements a T gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            if in_state.kets[i].states[j][t_index] == '1':
                if ctransposed:
                    in_state.kets[i].amplitudes[j] *= np.exp(np.pi*(-1j)/4)
                else:
                    in_state.kets[i].amplitudes[j] *= np.exp(np.pi*(1j)/4)
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def rotation_x(in_state, t_index, theta, bits = ''):  # Implements a rotation X gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            in_state.kets[i].states.append(bitflip(in_state.kets[i].states[j], t_index))
            in_state.kets[i].amplitudes.append(in_state.kets[i].amplitudes[j] * (-1j) * np.sin(theta/2))
            in_state.kets[i].amplitudes[j] *= np.cos(theta/2)
        in_state.kets[i].well_order = False
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def rotation_y(in_state, t_index, theta, bits = ''):  # Implements a rotation Y gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            in_state.kets[i].states.append(bitflip(in_state.kets[i].states[j], t_index))
            if in_state.kets[i].states[j][t_index] == '0':
                in_state.kets[i].amplitudes.append(-1 * in_state.kets[i].amplitudes[j] * np.sin(theta/2))
            else:
                in_state.kets[i].amplitudes.append(in_state.kets[i].amplitudes[j] * np.sin(theta/2))
            in_state.kets[i].amplitudes[j] *= np.cos(theta/2)
        in_state.kets[i].well_order = False
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def rotation_z(in_state, t_index, theta, bits = ''):  # Implements a rotation Z gate on a particular qubit
    output = Mixed_State(in_state.size)
    for i in range(len(in_state.kets)):
        for j in range(len(in_state.kets[i].states)):
            if in_state.kets[i].states[j][t_index] == '0':
                in_state.kets[i].amplitudes[j] *= np.exp(theta*(-1j)/2)
            else:
                in_state.kets[i].amplitudes[j] *= np.exp(theta*(1j)/2)
        output.add_state(in_state.kets[i], in_state.kets[i].precision * in_state.weights[i])
        if in_state.kets[i].precision != 1:
            output.add_state(error_generator(in_state.kets[i], t_index), (1 - in_state.kets[i].precision) * in_state.weights[i])
    return output


def error_generator(in_state, t_index):  # A function that generates a small error to alter the fidelity
    error = Ket(in_state.size)
    error.states = in_state.states.copy()
    error.amplitudes = in_state.amplitudes.copy()
    angles = np.random.normal(scale = np.pi/4, size = 4) #alpha, beta, delta, gamma
    error = rotation_x(error, t_index, angles[0])
    error = rotation_y(error, t_index, angles[1])
    error = rotation_z(error, t_index, angles[2])
    error = phase_change(error, 0, angles[3])
    error.recombine()
    return error


def uni_operation(in_state, uni_operator, t_index):  # Implements a decomposed unitary operator
    soln = decomposer(uni_operator)
    in_state = rotation_z(in_state, t_index, soln[2])
    in_state = rotation_y(in_state, t_index, soln[3])
    in_state = rotation_z(in_state, t_index, soln[1])
    in_state = phase_change(in_state, t_index, soln[0])
    in_state.recombine()


def get_order(gate):  # Returns the order of the gate
    return gate[1]


def get_gate_fid(precision):  # Returns the fidelity of a particular gate implementation
    if isinstance(precision, float):
        fid = []
        Alice = Ket(1,precision)
        Alice.add_term('0', 1)
        for i in range(10000):
            Bob = Mixed_State(1)
            Bob.add_state(Alice, precision)
            Bob.add_state(error_generator(Alice, 0), 1 - precision)
            fid.append(get_fidelity(Alice, Bob))
        average = np.mean(fid)
        return average
    elif isinstance(precision, list):
        fidelities = []
        for item in precision:
            fidelities.append(get_gate_fid(item))
        return fidelities

def representation_switch(gate):  # Charaterizes the gate to be added to the circuit
    if gate == 'pauli_x':
        return '-|X|-'
    elif gate == 'pauli_y':
        return '-|Y|-'
    elif gate == 'pauli_z':
        return '-|Z|-'
    elif gate == 'phase':
        return '-|P|-'
    elif gate == 'hadamard':
        return '-|H|-'
    elif gate == 't':
        return '-|T|-'
    elif gate == 'rotation_x':
        return '|R_x|'
    elif gate == 'rotation_y':
        return '|R_y|'
    elif gate == 'rotation_z':
        return '|R_z|'

class Circuit:  # Defining a class for the circuit diagram of an algorithm
    def __init__(self, qubit_number, bit_number = 0):
        self.size = qubit_number
        self.classical_size = bit_number
        self.tracks = []  # a list of strings visualizing a quantum circuit
        self.gate_list = []
        self.acting_list = []

        for i in range(self.size):
            self.tracks.append(['|Q%s|' % i])

        for j in range(self.classical_size):
            self.tracks.append(['|C%s|' % j])

    def __repr__(self):
        return "<number of qubits:%s, classical bits:%s>" % (self.size, self.classical_size)

    def __str__(self):
        return "to visualize the circuit use the .visualize() method"

    def single_gate(self, gate, t_index, theta = 0):
        for i in range(self.size + self.classical_size):
            if i >= self.size:
                self.tracks[i].append('=====')
            elif i != t_index:
                self.tracks[i].append('-----')
            else:
                self.tracks[i].append(representation_switch(gate))
        self.gate_list.append(gate)
        if theta == 0:
            self.acting_list.append([t_index])
        else:
            self.acting_list.append([t_index, theta])

    def clsc_ctrl_gate(self, gate, q_index, c_index):
        for i in range(self.size + self.classical_size):
            if i == q_index:
                self.tracks[i].append(representation_switch(gate))
            elif i == c_index + self.size:
                self.tracks[i].append('==•==')
            elif i > q_index and i < c_index + self.size:
                if i >= self.size:
                    self.tracks[i].append('==|==')
                else:
                    self.tracks[i].append('--|--')
            elif i >= self.size:
                self.tracks[i].append('=====')
            else:
                self.tracks[i].append('-----')
        self.gate_list.append('classical_control')
        self.acting_list.append([gate, q_index, c_index])

    def add_cnot(self, qubit_control, qubit_target):
        for i in range(self.size + self.classical_size):
            if i == qubit_target:
                self.tracks[i].append('--⊕--')
            elif i == qubit_control:
                self.tracks[i].append('--•--')
            elif (qubit_target > i) == (i > qubit_control):
                self.tracks[i].append('--|--')
            elif i >= self.size:
                self.tracks[i].append('=====')
            else:
                self.tracks[i].append('-----')
        self.gate_list.append('cnot')
        self.acting_list.append([qubit_control, qubit_target])

    def add_measure(self, qubit_number, bit_number = 0):
        for i in range(self.size + self.classical_size):
            if i == qubit_number:
                self.tracks[i].append('-◜/◝-')
            elif i == bit_number + self.size:
                self.tracks[i].append('==□==')
            elif i > qubit_number and i < bit_number + self.size:
                if i >= self.size:
                    self.tracks[i].append('==‖==')
                else:
                    self.tracks[i].append('--‖--')
            elif i >= self.size:
                self.tracks[i].append('=====')
            else:
                self.tracks[i].append('-----')
        self.gate_list.append('measure')
        self.acting_list.append([qubit_number, bit_number])

    def add_fr(self): ##full readout
        for i in range(self.size):
            self.tracks[i].append('-◜/◝-')
        self.gate_list.append('full_readout')
        self.acting_list.append('')

    def visualize(self, compact = False):
        s = [[str(e) for e in row] for row in self.tracks]
        lens = [max(map(len, col)) for col in zip(*s)]
        if compact:
            fmt = ''.join('{{:{}}}'.format(x) for x in lens)
        else:
            fmt = ' '.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))


def run_circuit(qubits, circuit, bits = ''):
    for i in range(len(circuit.gate_list)):
        if len(circuit.acting_list[i]) == 1:
            qubits = eval(circuit.gate_list[i])(qubits, circuit.acting_list[i][0], bits)
        elif len(circuit.acting_list[i]) == 2:
            qubits = eval(circuit.gate_list[i])(qubits, circuit.acting_list[i][0], circuit.acting_list[i][1], bits)
        elif len(circuit.acting_list[i]) == 3:
            qubits = eval(circuit.gate_list[i])(qubits, circuit.acting_list[i][0], circuit.acting_list[i][1], circuit.acting_list[i][2], bits)
        else:
            qubits = eval(circuit.gate_list[i])(qubits, bits)
    return qubits

class Clsc_Bit:
    def __init__(self, bits):
        self.digits = bits
        self.size = len(bits)

    def __repr__(self):
        return "<number of classical bits:%s>" % (self.size)

    def __str__(self):
        return self.digits

    def bit_change(self, index, new_bit):
        self.digits = list(self.digits)
        self.digits[index] = new_bit
        self.digits = ''.join(self.digits)

def not_gate(clsc_bits, t_index):
    clsc_bits.digits = bitflip(clsc_bits.digits, t_index)
    return clsc_bits

def classical_control(qubits, gate, q_index, c_index, bits):
    if bits.digits[c_index] == '1':
        qubits = eval(gate)(qubits, q_index)
    return qubits




