import json
from builtins import list
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import Planck, c
from scipy.special import erfcinv

class Lightpath(object):
    def __init__(self, path, channel, power=None, Rs=32e9, df=50e9, transceiver= 'shanon'):
        self._signal_power = power
        self._path = path
        self._channel = channel
        self._Rs = Rs
        self._df = df
        self._noise_power = 0
        self._latency = 0
        self._snr = None
        self._optimized_powers = {}
        self._transceiver = transceiver
        self._bitrate = None

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, power):
        self._signal_power = power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def channel(self):
        return self._channel

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @property
    def rs(self):
        return self._Rs

    @property
    def df(self):
        return self._df

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):
        self.noise_power += noise

    def add_latency(self, latency):
        self.latency += latency

    def next(self):
        self.path = self.path[1:]

    @property
    def optimized_powers(self):
        return self._optimized_powers

    @optimized_powers.setter
    def optimized_powers(self, optimized_powers):
        self._optimized_powers = optimized_powers

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    def update_snr(self, snr):
        if self.snr is None:
            self.snr = snr
        else:
            self.snr = 1 / (1 / self.snr + 1 / snr)

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate


class Connection(object):
    def __init__(self, input_node, output_node, signal_power=None, rate_request=0):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = signal_power
        self._latency = 0
        self._snr = []
        self._bitrate = None
        self._rate_request = float(rate_request)
        self._residual_rate_request = float(rate_request)
        self._lightpaths = []
        self._full_path = ''

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._output_node

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, power):
        self._signal_power = power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    @property
    def rate_request(self):
        return self._rate_request

    @property
    def residual_rate_request(self):
        return self._residual_rate_request

    @property
    def lightpaths(self):
        return self._lightpaths

    @lightpaths.setter
    def lightpaths(self, lightpath):
        self._lightpaths.append(lightpath)

    @snr.setter
    def snr(self, snr):
        self._snr.append(snr)

    def clear_lightpaths(self):
        self._lightpaths = []

    def calculate_capacity(self):
        self.bitrate = sum([lightpath.bitrate for lightpath in self.lightpaths])
        return self.bitrate

    def set_connection(self, lightpath):
        self.signal_power = lightpath.signal_power
        self.latency = max(self.latency, lightpath.latency)
        self.snr = 10 * np.log10(lightpath.snr)
        self.lightpaths = lightpath

        self._residual_rate_request = self._residual_rate_request - lightpath.bitrate
        return self

    def block_connection(self):
        self.latency = None
        self._snr = [0]
        self.bitrate = 0
        self.clear_lightpaths()
        return self

    @property
    def full_path(self):
        return self._full_path

    @full_path.setter
    def full_path(self, path):
        self._full_path = path

    def crossed_channels(self):
        crossed = self.full_path
        for lightpath in self.lightpaths:
            crossed += str(lightpath.channel)
        return crossed


class Line(object):
    def __init__(self, line_dict, fiber_type='SMF'):
        self._label = line_dict['label']
        self._length = line_dict['length']*1e3
        self._Nch = line_dict['Nch']
        self._amplifiers = int(np.ceil(self._length / 80e3))
        self._span_length = self._length / self._amplifiers
        self._state = ['free'] * 10
        self._successive = {}
        self._gain = 20
        self._noise_figure = 5

        # Physical parameters of the fiber
        self._alpha = 4.6e-5
        self._gamma = 1.27e-3
        # beta --> dispersion: the spreading out of a light pulse in time as it propagates down the fiber
        if fiber_type == 'LEAF':
            self._beta = 6.58e-27
        else:
            self._beta = 21.27e-27

        # Set Gain to transparency
        self._gain = self.transparency()

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        state = [s.lower().strip() for s in state]

        if set(state).issubset({'free', 'occupied'}):
            self._state = state
        else:
            print('ERROR: line state  not recognized.Value: ', set(state) - {'free', 'occupied'})

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def amplifiers(self):
        return self._amplifiers

    @property
    def span_length(self):
        return self._span_length

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def noise_figure(self):
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self, noise_figure):
        self._noise_figure = noise_figure

    @property
    def Nch(self):
        return self._Nch

    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency

    def noise_generation(self, lightpath):
        noise = self.ase_generation() + self.nli_generation(lightpath.signal_power, lightpath.rs, lightpath.df)
        return noise

    def propagate(self, lightpath, occupation=False):

        # Update latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)

        # Update noise
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)

        # Update SNR
        snr = lightpath.signal_power / noise
        lightpath.update_snr(snr)

        # Update line state
        if occupation:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = 'occupied'
            self.state = new_state

        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)

        return lightpath

    def ase_generation(self):
        """
        Amplified spontaneous emission --> added after every amplifier: ILA(inline amplifier), Booster, Preamp
        amplified spontaneous emission calculation for a line
        :return: ase noise resulting from amplifying
        """
        gain_lin = 10 ** (self._gain / 10)
        noise_figure_lin = 10 ** (self._noise_figure / 10)
        n_amplifiers = self._amplifiers
        f = 193.4e12
        h = Planck
        Bn = 12.5e9
        ase_noise = n_amplifiers * h * f * Bn * noise_figure_lin * (gain_lin - 1)

        return ase_noise

    def nli_generation(self, signal_power, Rs, df):
        """
        non linear interference calculation: a Gaussian noise
        generated along the fiber span
        Kerr effect
        :param signal_power:
        :param Rs:
        :param df:
        :return:
        """

        Pch = signal_power
        Bn = 12.5e9
        loss = np.exp(- self.alpha * self.span_length)
        # num of spans = num of amps
        N_spans = self.amplifiers
        eta = 16 / (27 * np.pi) * np.log(
            np.pi ** 2 * self.beta * Rs ** 2 * self.Nch ** (2 * Rs / df) / (2 * self.alpha)) * self.gamma ** 2 / (
                          4 * self.alpha * self.beta * Rs ** 3)

        nli_noise = N_spans * (Pch ** 3 * loss * self.gain * eta * Bn)
        return nli_noise

    def transparency(self):
        gain = 10 * np.log10(np.exp(self.alpha * self.span_length))
        return gain


class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath.signal_power = lightpath.optimized_powers[line_label]
            lightpath = line.propagate(lightpath, occupation)

        return lightpath

    def optimize(self, lightpath):
        """
        a recursive func used to calculate GSNR along a lightpath
        performs the role of the OLS controller
        :param lightpath: ligthpath along which GSNR is being calculated
        :return: optimized ligthpath
        """
        path = lightpath.path

        # not the last node
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            # calculate line noise
            ase = line.ase_generation()
            # eta = P_NLI/P_CH^3
            eta = line.nli_generation(1, lightpath.rs, lightpath.df)
            # calculate optimum power
            p_opt = (ase / (2 * eta)) ** (1 / 3)
            # optimum power update for line
            lightpath.optimized_powers.update({line_label: p_opt})
            lightpath.next()
            node = line.successive[lightpath.path[0]]
            # recursively move to next nodes
            lightpath = node.optimize(lightpath)

        return lightpath


class Path(object):
    def __init__(self, start_node, end_node):
        self._start_node = start_node
        self._end_node = end_node
        self._path_string = start_node


    @property
    def start_node(self):
        return self._start_node

    @property
    def end_node(self):
        return self._end_node

    @property
    def path_string(self):
        return self._path_string

    @path_string.setter
    def path_string(self, path):
        self._path_string = path


class Realization:
    def __init__(self, network, streamed_connections):
        self._network = network
        self._streamed_connections = streamed_connections

        rbl = []
        snrs = []
        rbc = []

        for connection in streamed_connections:
            # snr
            snrs.extend(connection.snr)
            # rbl
            for lightpath in connection.lightpaths:
                rbl.append(lightpath.bitrate)
            # rbc
            rbc.append(connection.calculate_capacity())

        self._rbl = rbl
        self._snrs = snrs
        self._rbc = rbc

    @property
    def rbl(self):
        return self._rbl

    @property
    def rbc(self):
        return self._rbc

    @property
    def snrs(self):
        return self._snrs

    @property
    def network(self):
        return self._network

    @property
    def lines(self):
        return self.network.lines

    @property
    def lines_states(self):
        return self.network

    @property
    def streamed_connections(self):
        return self._streamed_connections

    def plot_SNR_dist(self):
        plt.hist(self.snrs, bins=10)
        plt.title('SNR Distribution')
        plt.show()

    def plot_bitrate_dist(self):
        plt.hist(self.rbc, bins=10)
        plt.title('Bitrate Distribution [Gbps]')
        plt.show()

    def plot_lightpath_capacity_dist(self):
        plt.hist(self.rbl, bins=10)
        plt.title('Lightpaths Capacity Distribution [Gbps]')
        plt.show()

    def plot_connection_capacity_dist(self):
        plt.hist(self.rbc, bins=10)
        plt.title('Connection Capacity Distribution [Gbps]')
        plt.show()

    def print_stats(self):
        # print('Total Capacity Connections: {:.2f} Tbps'.format(np.sum(self.rbc) * 1e-3))
        # print('Total Capacity Lightpaths: {:.2f} Tbps'.format(np.sum(self.rbl) * 1e-3))
        print('Total Capacity: {:.2f} Tbps '.format(np.sum(self.rbc) * 1e-3))
        print('Avg Capacity: {:.2f} Gbps '.format(np.mean(self.rbc)))
        print('Avg SNR: {:.2f} dB'.format(np.mean(list(filter(lambda x: x != 0, self.snrs)))))

    def bit_rate_matrix(self):
        node_labels = list(self.network.nodes.keys())
        s = pd.Series(data=[0.0] * len(node_labels), index=node_labels)
        df = pd.DataFrame(0.0, index=s.index, columns=s.index, dtype=s.dtype)

        for connection in self.streamed_connections:
            df.loc[connection.input_node, connection.output_node] = connection.bitrate

        return df

    def plot_bit_rate_matrix(self):
        matrix = self.bit_rate_matrix()
        plot_3d_bars(matrix.values)

    def blocking_ratio(self):
        return len([c for c in self.streamed_connections if c.bitrate == 0.0])/len(self.streamed_connections)


def calculate_bitrate(lightpath, bert=1e-3, bn=12.5e9):
    """
    calculate bitrate along a lightpath depending on the used transceiver
    saves the calculated bitrate to lightpath
    :param lightpath:
    :param bert:
    :param bn:
    :return:
    """
    snr = lightpath.snr
    rs = lightpath.rs
    rb = None

    if lightpath.transceiver.lower() == 'fixed-rate':
        # fixed-rate transceiver --> PM-QPSK modulation
        snrt = 2 * erfcinv(2 * bert) * (rs / bn)
        rb = np.piecewise(snr, [snr < snrt, snr >= snrt], [0, 100])

    elif lightpath.transceiver.lower() == 'flex-rate':
        snrt1 = 2 * erfcinv(2 * bert) ** 2 * (rs / bn)
        snrt2 = (14 / 3) * erfcinv(3 / 2 * bert) ** 2 * (rs / bn)
        snrt3 = (10) * erfcinv(8 / 3 * bert) ** 2 * (rs / bn)

        cond1 = (snr < snrt1)
        cond2 = (snrt1 <= snr < snrt2)
        cond3 = (snrt2 <= snr < snrt3)
        cond4 = (snr >= snrt3)

        rb = np.piecewise(snr, [cond1, cond2, cond3, cond4], [0, 100, 200, 400])

    elif lightpath.transceiver.lower() == 'shannon':
        rb = 2 * rs * np.log2(1 + snr * (rs / bn)) * 1e-9

    lightpath.bitrate = float(rb)
    return float(rb)


class Network(object):
    def __init__(self, json_path, nch=10, upgrade_line='', fiber_type='SMF'):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._connected = False
        self._weighted_paths = None
        self._route_space = None
        self._nch = nch
        self._all_paths = None
        self._json_path = json_path
        self._upgrade_line = upgrade_line
        self._fiber_type = fiber_type

        # loop through all nodes
        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))
                line_dict['Nch'] = self.nch

                self._lines[line_label] = Line(line_dict, fiber_type= fiber_type)

        # upgrade a line by decreasing by 3 dB the noise figure of the amplifiers along it
        if not upgrade_line == '':
            self.lines[upgrade_line].noise_figure = self.lines[upgrade_line].noise_figure - 3


    @property
    def nodes(self):
        return self._nodes

    @property
    def nch(self):
        return self._nch

    @property
    def lines(self):
        return self._lines

    def draw(self):
        if not self.connected:
            self.connect()

        nodes = self.nodes

        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0,'go', markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)

            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')

        plt.title('Network')
        plt.show()

    def find_paths_wrong(self, label1, label2, available_only=False):

        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {'0': label1}

        # generate all possible combinations of paths
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node for cross_node in cross_nodes if (
                        (inner_path[-1] + cross_node in cross_lines) & (cross_node not in inner_paths))]

        # filtered based on existing/available
        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)

        return paths

    def find_paths(self, node1_label, node2_label):

        path = Path(node1_label, node2_label)
        paths = []

        self.find_paths_from(node1_label, path, paths)

        return paths

    def find_paths_from(self, current_node_label, path, paths):
        """
        :param current_node_label: node to start from
        :param path: current path
        :param paths: all paths found so far
        :return: updated paths list
        """
        current_node = self.nodes[current_node_label]

        for connected_node in current_node.connected_nodes:
            # avoid loops
            if connected_node == path.start_node or connected_node in path.path_string:
                continue

            line = current_node_label + connected_node
            if line in self.lines:

                if connected_node != path.end_node:
                    # continue along the path
                    npath = Path(path.start_node, path.end_node)
                    npath.path_string = path.path_string + connected_node
                    self.find_paths_from(connected_node, npath, paths)
                else:
                    # add path to list
                    paths.append(path.path_string + connected_node)

        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        self._connected = True

        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]

    def propagate(self, lightpath, occupation=False):
        start_node = self.nodes[lightpath.path[0]]

        propagated_lightpath = start_node.propagate(lightpath, occupation)

        return propagated_lightpath

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space

    @property
    def connected(self):
        return self._connected

    def set_weighted_paths(self):
        """
        :return:
        """
        if not self.connected:
            self.connect()

        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []

        for pair in self.node_pairs():
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                # Propagation
                lightpath = Lightpath(path=path, channel=0)
                self.optimization(lightpath)
                self.propagate(lightpath, occupation=False)

                latencies.append(lightpath.latency)
                noises.append(lightpath.noise_power)
                snrs.append(10 * np.log10(lightpath.snr))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs

        self._weighted_paths = df

        route_space = pd.DataFrame()
        route_space['path'] = paths

        # wavelength availability matrix
        for i in range(self.nch):
            route_space[str(i)] = ['free'] * len(paths)
        self._route_space = route_space

    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        return best_path

    def find_best_latency(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)

        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0]
        else:
            best_path = None

        return best_path

    def stream(self, connections, best='latency', transceiver='shannon'):
        streamed_connections = []

        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node

            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('ERROR: best input not recognized.Value:', best)
                continue

            if path:
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                channel = [i for i in range(len(path_occupancy)) if path_occupancy[i] == 'free'][0]
                path = path.replace('->', '')

                connection.full_path = path

                # calculate GSNR
                in_lightpath = Lightpath(path, channel, transceiver=transceiver)
                in_lightpath = self.optimization(in_lightpath)
                out_lightpath = self.propagate(in_lightpath, True)

                # bitrate depending on transceiver technology
                calculate_bitrate(out_lightpath)

                if out_lightpath.bitrate == 0.0:
                    # [self.update_route_space(path, channel, 'free') for lp in connection.lightpaths]
                    self.update_route_space(path, channel, 'free')
                    connection.block_connection()

                else:
                    connection.set_connection(out_lightpath)
                    self.update_route_space(path, channel, 'occupied')

                    if connection.residual_rate_request > 0:
                        self.stream([connection], best, transceiver)
            else:
                # [self.update_route_space(path, channel, 'free') for lp in connection.lightpaths]
                connection.block_connection()

            streamed_connections.append(connection)

        return streamed_connections

    def available_paths(self, input_node, output_node):
        if self.weighted_paths is None:
            self.set_weighted_paths()

        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == input_node) and (path[-1] == output_node))]
        available_paths = []

        for path in all_paths:
            path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
            if 'free' in path_occupancy:
                available_paths.append(path)

        return available_paths

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    @property
    def all_paths(self):
        if self._all_paths is None:
            self._all_paths = [self.path_to_line_set(p) for p in self.route_space.path.values]
        return self._all_paths

    def update_route_space(self, path, channel, state):
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)

        for i in range(len(self.all_paths)):
            line_set = self.all_paths[i]
            if lines.intersection(line_set):
                states[i] = state

        self.route_space[str(channel)] = states

    def optimization(self, lightpath):
        """
        OLS controller
        find optimal channel power (for ---> ASE + NLI)
        :param lightpath: lightpath to calculate the GSNR of
        :return: optimized lightpath
        """
        path = lightpath.path

        start_node = self.nodes[path[0]]
        optimized_lightpath = start_node.optimize(lightpath)

        # path changes with recursion, redefine it
        optimized_lightpath.path = path

        return optimized_lightpath

    def node_pairs(self):
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)

        return pairs

    def make_new_network(self):
        return Network(self._json_path, self.nch, upgrade_line=self._upgrade_line, fiber_type=self._fiber_type)


def create_traffic_matrix(nodes, rate, multiplier=1):
    """
    :param nodes: nodes of the network
    :param rate: rate at which requests are made (more rate --> more congestion)
    :param multiplier: is an integer number multiplying the values of the traffic matrix in order to
    increase the number of lightpaths that need to be allocated for each connection request between a node pair
    :return: a matrix in which every cell indicates amount of traffic to be conveyed between two nodes
    """

    # increasing the multiplier value will make the network more congested and the lines more occupied
    s = pd.Series(data=[0.0] * len(nodes), index=nodes)
    df = pd.DataFrame(float(multiplier * rate), index=s.index, columns=s.index, dtype=s.dtype)
    np.fill_diagonal(df.values, 0.0)

    return df


def plot_3d_bars(t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data = np.meshgrid(np.arange(t.shape[1]), np.arange(t.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = t.flatten()
    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)
    plt.show()


class MonteCarloAnalysis(object):
    def __init__(self, mc):
        self._mc = mc

    def plot_lines_occupation(self):
        plt.bar(range(len(self._mc.lines_congestion())), height=self._mc.lines_congestion().values)
        plt.xticks(range(len(self._mc.lines_congestion())), self._mc.lines_congestion().keys())
        plt.title('lines occupation')
        plt.show()

    def find_line_to_upgrade(self):
        return self._mc.lines_congestion().sort_values(ascending=False).keys()[0]

    def print_stats(self):
        print('Total Capacity: {:.2f} Tbps '.format(self._mc.total_capacity()))
        print('Avg Capacity: {:.2f} Gbps '.format(self._mc.avg_rbl()))
        print('Avg SNR: {:.2f} dB'.format(self._mc.avg_snr()))

    def bit_rate_matrix(self):
        node_labels = list(self._mc.network.nodes.keys())
        s = pd.Series(data=[0.0] * len(node_labels), index=node_labels)
        df = pd.DataFrame(0.0, index=s.index, columns=s.index, dtype=s.dtype)

        connections = self._mc.connections[0]

        # averaging bitrates for connections
        bitrates = [[connection.bitrate for connection in connection_list] for connection_list in self._mc.connections]
        bitrates = np.mean(bitrates, axis=0)

        for i in range(len(connections)):
            df.loc[connections[i].input_node, connections[i].output_node] = bitrates[i]

        return df

    def plot_bit_rate_matrix(self):
        matrix = self.bit_rate_matrix()
        plot_3d_bars(matrix.values)

    def draw_network_occupation(self):
        nodes = self._mc.network.nodes
        fig, (ax1, ax2) = plt.subplots(1, 2)
        drawn = []
        fig.suptitle('Network')

        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            ax1.plot(x0, y0, 'go', markersize=10)
            ax1.text(x0 + 20, y0 + 20, node_label)
            ax2.plot(x0, y0, 'go', markersize=10)
            ax2.text(x0 + 20, y0 + 20, node_label)

            for connected_node_label in n0.connected_nodes:
                line_label = node_label + connected_node_label
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                start = [x0, x1]
                end = [y0, y1]

                if line_label in drawn:
                    ax = ax2
                else:
                    ax = ax1

                if self._mc.lines_congestion()[str(node_label + connected_node_label)] == 1.0:
                    ax.plot(start, end, 'r')
                elif self._mc.lines_congestion()[str(node_label + connected_node_label)] > 0.9:
                    ax.plot(start, end, 'y')
                else:
                    ax.plot(start, end, 'b')

                ax.text(np.mean(start), np.mean(end), line_label)
                drawn.append(line_label[::-1])

        plt.show()

    def blocking_ratio(self):
        ratios = []
        for connections_list in self._mc.connections:
            ratios.append(len([c for c in connections_list if c.bitrate == 0.0])/len(connections_list))
        return np.mean(ratios)


class MonteCarloSim(object):
    def __init__(self, num_realizations, network):
        self._rbcs = []
        self._rbls = []
        self._snrs = []
        self._connections = []
        self._lines_states = []
        self._lines = []
        self._num_realizations = num_realizations
        self._realizations = []
        self._network = network

    def simulate(self, rate, multiplier):
        """
        :param rate: rate at which requests are made (uniform for all nodes)
        :param multiplier: for data rate
        :return: None
        """
        realizations = []
        for i in range(self.num_realizations):

            # create a matching network
            network = self.network.make_new_network()

            # change order of connections by shuffling pairs
            pairs = network.node_pairs().copy()
            shuffle(pairs)

            # create a connection for every random pair
            traffic_matrix = create_traffic_matrix(list(network.nodes.keys()), rate, multiplier=multiplier)
            connections = []
            for node_pair in pairs:
                connection = Connection(node_pair[0], node_pair[1],
                                        rate_request=float(traffic_matrix.loc[node_pair[0], node_pair[1]]))
                connections.append(connection)

            # stream created connections
            streamed_connections = network.stream(connections, best='snr')

            # create realization
            realization = Realization(network, streamed_connections)
            realizations.append(realization)

        self._realizations = realizations

    @property
    def network(self):
        return self._network

    @property
    def num_realizations(self):
        return self._num_realizations

    @property
    def realizations(self):
        return self._realizations

    @property
    def rbls(self):
        return [realization.rbl for realization in self.realizations]

    @property
    def rbcs(self):
        """
        :return: a list of lists each containing the rbcs of every connection of a single MonteCarlo realization
        """
        return [realization.rbc for realization in self.realizations]

    @property
    def snrs(self):
        return [realization.snrs for realization in self.realizations]

    @property
    def connections(self):
        return [realization.streamed_connections for realization in self.realizations]

    def avg_snr(self):
        return np.mean([np.mean(snr) for snr in self.snrs])

    def avg_rbl(self):
        """
        :return: avg lightpath bitrate in Gbps
        """
        return np.mean([np.mean(rbl) for rbl in self.rbls])

    def avg_rbc(self):
        return np.mean(self.rbcs)

    def total_capacity(self):
        """
        :return: total capacity of the network in Tbps
        """
        return np.mean([np.sum(rbl) for rbl in self.rbls]) * 1e-3

    def lines(self):
        return [realization.lines for realization in self.realizations]

    def lines_congestion(self):
        df = pd.DataFrame(self.lines())
        df = df.applymap(lambda x: x.state.count('occupied')/len(x.state))

        return df.mean()


network_small = Network('nodes_2.json', nch=10, fiber_type='LEAF')
# network_big = Network('nodes_big.json', nch=10)
# network_mesh = Network('nodes_mesh.json', nch=10)

# specify network
network = network_small

# draw network
network.draw()

# # simulate network
mc = MonteCarloSim(num_realizations=5, network=network)
mc.simulate(300, 7)
# at 11-14 the network_mesh gets congested
#
#

# analyze the small network
mc_analysis = MonteCarloAnalysis(mc)

mc_analysis.plot_bit_rate_matrix()
mc_analysis.plot_lines_occupation()
mc_analysis.draw_network_occupation()

# print results
mc_analysis.print_stats()
print('line to be upgraded: ' + mc_analysis.find_line_to_upgrade())