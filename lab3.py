import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c


class SignalInformation(object):
    def __init__(self, power, path):
        self._signal_power = power
        self._noise_power = 0
        self._latency = 0
        self._path = path

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):
        self.noise_power += noise

    def add_latency(self, latency):
        self.latency += latency

    def next(self):
        self.path = self.path[1:]

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

    def propagate(self, signal_information):
        path = signal_information.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.next()
            signal_information = line.propagate(signal_information)

        return signal_information

class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def latency_generation(self):
        latency = self.length / (c * 2/3)
        return latency

    def noise_generation(self, signal_power):
        noise = signal_power / (2 * self.length)
        return noise

    def propagate(self, signal_information):
        #update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        #update noise
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        node = self._successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information)
        return signal_information

class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        for node_label in node_json:
            #creating node instance
            node_dict = node_json[node_label]
            node_dict['label']  =node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            #creating line instance
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))
                line = Line(line_dict)
                self._lines[line_label] = line

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go', markersize = 10)
            plt.text(x0 + 20, y0 + 20, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys()
                        if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {}
        paths = []
        inner_paths['0'] = label1
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i+1)] = []
            #print(inner_paths[str(i)])
            for inner_path in inner_paths[str(i)]:
                print(inner_path)
                inner_paths[str(i+1)] += [inner_path + cross_node
                                          for cross_node in cross_nodes
                                          if((inner_path[-1] + cross_node in cross_lines) & (cross_node not in inner_path))]
                #if((inner_path[-1] + label2 in cross_lines)&((inner_path[-1] + label2) not in paths)):
                 #   paths.append(inner_path+label2)

        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)

        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node._connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]

    def propagate(self, signal_information):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(signal_information)
        return propagated_signal_information


network = Network('nodes.json')
network.connect()
node_labels = network.nodes.keys()
pairs = []
for label1 in node_labels:
    for label2 in node_labels:
     if label1 != label2:
        pairs.append(label1+label2)

columns = ['path', 'latency', 'noise', 'snr']
df = pd.DataFrame()
paths = []
latencies = []
noises = []
snrs = []

for pair in pairs:
    for path in network.find_paths(pair[0], pair[1]):
        path_string = ''
        for node in path:
            path_string += node + '->'
        paths.append(path_string[:-2])

        #Propagation
        signal_information = SignalInformation(1, path)
        signal_information = network.propagate(signal_information)
        latencies.append(signal_information.latency)
        noises.append(signal_information.noise_power)
        snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))

df['path'] = paths
df['latency'] = latencies
df['noise'] = noises
df['snr'] = snrs

#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df)
network.draw()