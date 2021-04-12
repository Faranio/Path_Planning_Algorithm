import numpy as np

from src.main import default_edge_cost


class Graph:
    def __init__(self, graph_dict=None):
        self.graph_dict = graph_dict or {}

    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        self.graph_dict.setdefault(B, {})[A] = distance

    # Get neighbors or a neighbor of the node
    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})

        if b is None:
            return links
        else:
            return links.get(b)

    # Return a list of nodes of the graph
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


class Node:
    def __init__(self, name: str, parent: str):
        self.name = name
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to end node
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return '({0}, {1})'.format(self.name, self.f)


def add_to_open(open, neighbor):
    for node in open:
        if neighbor == node and neighbor.f > node.f:
            return False
    return True


def a_star_search(graph, heuristics, start, end):
    open = []
    closed = []

    start_node = Node(start, None)
    goal_node = Node(end, None)

    open.append(start_node)

    while len(open) > 0:
        open.sort()
        current_node = open.pop(0)
        closed.append(current_node)

        if current_node == goal_node:
            path = []

            while current_node != start_node:
                path.append([current_node.name, str(current_node.g)])
                current_node = current_node.parent

            path.append([start_node.name, str(start_node.g)])
            return path[::-1]

        neighbors = graph.get(current_node.name)

        for key, value in neighbors.items():
            neighbor = Node(key, current_node)

            if neighbor in closed:
                continue

            neighbor.g = current_node.g + graph.get(current_node.name, neighbor.name)
            neighbor.h = heuristics.get(neighbor.name)
            neighbor.f = neighbor.g + neighbor.h

            if add_to_open(open, neighbor):
                open.append(neighbor)

    return None


def find_a_star_path(distance_matrix, mapping, start, end):
    graph = Graph()
    heuristics = {}

    i, j = np.where(distance_matrix < default_edge_cost)

    for row, col in zip(i, j):
        p1 = mapping[row]
        p2 = mapping[col]
        graph.connect(str(p1), str(p2), distance_matrix[row][col])

    # Setting heuristic values
    for row in i:
        p = mapping[row]
        heuristics[str(p)] = 0

    path = a_star_search(graph, heuristics, str(start), str(end))
    return path
