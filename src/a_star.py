import numpy as np

from src.main import config


class Graph:
    def __init__(self, graph_dict=None):
        self.graph_dict = graph_dict or {}

    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        self.graph_dict.setdefault(B, {})[A] = distance

    def get_neighbors(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        return links.get(b) if b else links

    def get_nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


class Node:
    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent
        self.distance_to_start_node = 0
        self.distance_to_end_node = 0
        self.total_distance = 0

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.total_distance < other.total_distance

    def __repr__(self):
        return '({0}, {1})'.format(self.name, self.total_distance)


def add_to_open(opened_nodes, neighbor):
    for node in opened_nodes:
        if neighbor == node and neighbor.total_distance > node.total_distance:
            return False
    return True


def update_neighbor_distances(neighbor, node, graph, heuristics):
    neighbor.distance_to_start_node = node.distance_to_start_node + graph.get_neighbors(node.name,
                                                                                                neighbor.name)
    neighbor.distance_to_end_node = heuristics.get(neighbor.name)
    neighbor.total_distance = neighbor.distance_to_start_node + neighbor.distance_to_end_node
    return neighbor


def update_neighbors_distances(neighbors, node, opened, closed, graph, heuristics):
    for key, value in neighbors.items():
        neighbor = Node(key, node)

        if neighbor in closed:
            continue

        neighbor = update_neighbor_distances(neighbor, node, graph, heuristics)

        if add_to_open(opened, neighbor):
            opened.append(neighbor)

    return opened, closed


def construct_path(current_node, start_node):
    path = []

    while current_node != start_node:
        path.append([current_node.name, str(current_node.distance_to_start_node)])
        current_node = current_node.parent

    path.append([start_node.name, str(start_node.distance_to_start_node)])
    return path[::-1]


def a_star_search(graph, heuristics, start, end):
    opened = []
    closed = []

    start_node = Node(start)
    goal_node = Node(end)

    opened.append(start_node)

    while len(opened) > 0:
        opened.sort()
        current_node = opened.pop(0)
        closed.append(current_node)

        if current_node == goal_node:
            return construct_path(current_node, start_node)

        neighbors = graph.get_neighbors(current_node.name)
        opened, closed = update_neighbors_distances(neighbors, current_node, opened, closed, graph, heuristics)

    return None


def find_a_star_path(distance_matrix, mapping, start, end):
    graph = Graph()
    heuristics = {}

    rows, cols = np.where(distance_matrix < config['DEFAULT_EDGE_COST'])

    for row, col in zip(rows, cols):
        point1 = mapping[row]
        point2 = mapping[col]
        graph.connect(str(point1), str(point2), distance_matrix[row][col])

    for row in rows:
        point = mapping[row]
        heuristics[str(point)] = 0

    path = a_star_search(graph, heuristics, str(start), str(end))
    return path
