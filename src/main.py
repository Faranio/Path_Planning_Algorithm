import collections
import itertools
import matplotlib.pyplot as plt
import more_itertools as mi
import shapely.geometry as shg

from lgblkb_tools import logger
from lgblkb_tools.common.utils import ParallelTasker
from lgblkb_tools.geometry import FieldPoly
from tsp_solver.greedy import solve_tsp

from src.a_star import *

config = {
    'DEFAULT_EDGE_COST': 1e6,
    'DFS_THRESHOLD': 45,
    'EXISTING_EDGE_COST': 1,
    'EXTRAPOLATION_OFFSET': 2,
    'MINIMUM_DISTANCE': 1e-6,
    'PATH_WIDTH': 40,
    'REQUIRED_EDGE_COST': -1e5,
    'SEARCH_LOOP_LIMIT': 3e5,
    'WORKERS_COUNT': 4
}


def get_field_lines(field_poly: FieldPoly, show=False):
    field_poly = FieldPoly.as_valid(field_poly)
    field_lines = field_poly.get_field_lines_old(config['PATH_WIDTH'],
                                                 field_poly.get_min_cost(config['PATH_WIDTH']).base_line, show=show)
    return field_lines


def get_paths(poly: FieldPoly, start, goal, loop_limit=5e6):
    graph = poly.adjacency_info

    if len(poly.G.decision_points['all']) < config['DFS_THRESHOLD']:
        pop_index = -1
        loop_limit = 5e6
    else:
        pop_index = 0

    hole_points = [hole.geometry.representative_point() for hole in poly.holes]
    stack = [(start, [start])]
    seen = set()
    loop_counter = itertools.count()

    while stack:
        (vertex, path) = stack.pop(pop_index)
        curr_loop_count = next(loop_counter)

        if curr_loop_count % int(loop_limit / 50) == 0:
            logger.debug(f"curr_loop_count: {curr_loop_count}")

        for _next in graph[vertex] - set(path):
            if _next == goal:
                some_path = path + [_next]

                if len(some_path) < 3:
                    continue

                path_hash = sum(hash(x) for x in some_path)

                if path_hash in seen:
                    continue

                seen.add(path_hash)
                some_field = FieldPoly(some_path)

                if some_field.geometry.area < config['MINIMUM_DISTANCE']:
                    continue

                if not some_field.geometry.is_valid:
                    continue

                some_field = FieldPoly.as_valid(some_field.geometry)
                contains_hole = False

                for point_in_hole in hole_points:
                    if point_in_hole.within(some_field.geometry):
                        contains_hole = True
                        break

                if contains_hole:
                    continue

                yield some_field
            else:
                stack.append((_next, path + [_next]))

        if curr_loop_count > loop_limit:
            logger.info(f"Final_loop_count: {curr_loop_count}")
            break


def get_other_polys(parent_poly, child_poly):
    try:
        diff_result = parent_poly.geometry.difference(child_poly.geometry)
    except Exception as exc:
        logger.warning(str(exc))
        return []

    if isinstance(diff_result, shg.Polygon):
        polygons = [diff_result]
    elif isinstance(diff_result, shg.MultiPolygon):
        polygons = [x for x in diff_result]
    elif isinstance(diff_result, shg.GeometryCollection):
        diff_results = diff_result
        polygons = list()

        for diff_result in diff_results:
            if isinstance(diff_result, shg.Polygon):
                polygons.append(diff_result)
            elif isinstance(diff_result, shg.MultiPolygon):
                polygons.extend([x for x in diff_result])
    else:
        logger.error(f"diff_result: \n{diff_result}")
        raise NotImplementedError(str(type(diff_result)))

    polygons = [FieldPoly.as_valid(x) for x in polygons if x.area > config['MINIMUM_DISTANCE']]
    return polygons


Candidate = collections.namedtuple('Candidate', ['acr', 'field'])


def decompose_from_point(field_poly, some_point, show=False):
    counter = itertools.count()
    curr_count = 0
    start = some_point
    goal = field_poly.G.nodes[some_point]['lines'][0][-1]
    best = Candidate(field_poly.area_cost_ratio, field_poly)

    if show:
        field_poly.plot()
        logger.debug(f"Best: {best}")
        best.field.plot(text=f'Accuracy = {best.acr}')
        start.plot('Start')
        goal.plot('Goal')
        plt.tight_layout()
        plt.show()

    decomposed_polygons = [best.field]

    for path_field in get_paths(field_poly, start, goal, loop_limit=config['SEARCH_LOOP_LIMIT']):
        curr_count = next(counter)
        polygons = get_other_polys(field_poly, path_field)
        costs = [p.get_min_cost(config['PATH_WIDTH']).cost for p in [path_field] + polygons]
        area_cost_ratio = field_poly.geometry.area / sum(costs)

        if area_cost_ratio > best.acr:
            best = Candidate(area_cost_ratio, path_field)
            decomposed_polygons = [best.field, *polygons]

    logger.debug(f"Final_count: {curr_count}")
    logger.info(f"Final best: {best}")

    if show:
        field_poly.plot()
        best.field.plot(text=f'Accuracy = {best.acr}')

    return decomposed_polygons


def decompose_from_points(field_poly: FieldPoly, points=None, use_mp=False, show=False):
    resultant_polys = [field_poly]
    max_acr = field_poly.area_cost_ratio
    logger.debug(f"Starting area_cost_ratio: {max_acr}")
    points = field_poly.get_outer_points() if points is None else points

    if use_mp:
        chunks_of_polygons = ParallelTasker(decompose_from_point, field_poly, show=show)\
            .set_run_params(some_point=points).run(workers_count=config['WORKERS_COUNT'])

        for res_polygons in chunks_of_polygons:
            decomp_cost = res_polygons[0].area_cost_ratio

            if decomp_cost > max_acr:
                logger.info(f"Better decomposition accuracy found: {decomp_cost}")
                max_acr = decomp_cost
                resultant_polys = res_polygons
    else:
        for outer_point in points:
            res_polygons = decompose_from_point(field_poly, outer_point, show=show)
            decomp_cost = res_polygons[0].area_cost_ratio

            if decomp_cost > max_acr:
                logger.info(f"Better decomposition accuracy found: {decomp_cost}")
                max_acr = decomp_cost
                resultant_polys = res_polygons

    return resultant_polys


@logger.trace()
def perform_optimization(field_poly, use_mp=False):
    polygons = decompose_from_points(field_poly, field_poly.get_outer_points(), use_mp=use_mp, show=False)
    optimum_polygons = [polygons[0]]

    if use_mp:
        if not len(polygons[1:]) == 0:
            chunks_of_optimum_sub_polygons = mi.flatten(ParallelTasker(perform_optimization)
                                                        .set_run_params(field_poly=polygons[1:]).run(len(polygons[1:])))
            optimum_polygons.extend(chunks_of_optimum_sub_polygons)
    else:
        for other_polygon in polygons[1:]:
            optimum_sub_polygons = perform_optimization(other_polygon)
            optimum_polygons.extend(optimum_sub_polygons)

    return optimum_polygons


@logger.trace()
def plot_optimum_polygons(optimum_polygons, field_poly):
    plt.figure(figsize=(20, 20))
    costs = []
    total_field_count = 0
    all_lines = []

    for optimum_polygon in optimum_polygons:
        lines_count = len(get_field_lines(optimum_polygon, show=False))
        total_field_count += lines_count
        optimum_polygon.plot(lw=5)
        temp_x, temp_y = optimum_polygon.polygon.centroid.xy
        temp_x, temp_y = float(temp_x[0]), float(temp_y[0])
        plt.text(temp_x, temp_y, f"Cost: {lines_count}", ha='center', va='center', fontsize=40)
        costs.append(optimum_polygon.get_min_cost(config['PATH_WIDTH']).cost)
        field_lines = get_field_lines(optimum_polygon)

        for line in field_lines:
            ls = shg.LineString(line.line)
            all_lines.append(ls)
            ls_x, ls_y = ls.xy
            plt.plot(ls_x, ls_y, c='r', lw=5)

    final_acr = field_poly.geometry.area / sum(costs)
    logger.info(f"Final Area Cost Ratio: {final_acr}")
    logger.info(f"Total Field Count: {total_field_count}")
    field_poly.plot(lw=5)
    plt.gca().set_aspect('equal', 'box')
    plt.grid(axis='both')
    plt.title("Optimum Polygons", fontsize=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.show()


def get_lines(field_poly):
    all_lines = []

    if isinstance(field_poly, list):
        for optimum_polygon in field_poly:
            field_lines = get_field_lines(optimum_polygon)

            for line in field_lines:
                ls = shg.LineString(line.line)
                all_lines.append(ls)
    else:
        field_lines = get_field_lines(field_poly)

        for line in field_lines:
            ls = shg.LineString(line.line)
            all_lines.append(ls)

    return all_lines


def get_extrapolated_line(p1, p2):
    divisor = p2[0] - p1[0]

    if divisor != 0:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p2[1] - m * p2[0]
        sign = 1 if p1[0] - p2[0] > 0 else -1
        a = (p1[0] + sign * config['EXTRAPOLATION_OFFSET'], m * (p1[0] + sign * config['EXTRAPOLATION_OFFSET']) + c)
        b = (p2[0] + (-1) * sign * config['EXTRAPOLATION_OFFSET'], m * (p2[0] + (-1) * sign * config['EXTRAPOLATION_OFFSET']) + c)
    else:
        sign = 1 if p1[1] - p2[1] > 0 else -1
        a = (p1[0], p1[1] + sign * config['EXTRAPOLATION_OFFSET'])
        b = (p2[0], p2[1] + (-1) * sign * config['EXTRAPOLATION_OFFSET'])

    return shg.LineString([a, b])


def get_intersections(ls, E):
    intersection_points = set()
    unique_points = set()

    for ext_line in E:
        ext_ls = shg.LineString(ext_line)
        temp = ls.intersection(ext_ls)

        if not temp.is_empty:
            res = (round(temp.coords[0][0]), round(temp.coords[0][1]))

            if res not in unique_points:
                intersection_points.add(temp.coords[0])
                unique_points.add(res)

    return intersection_points


def segments(curve):
    return list(map(shg.LineString, zip(curve.coords[:-1], curve.coords[1:])))


def polygons_to_graph(optimum_polygons, distance_threshold=1e-4, show=False):
    V, E, R = set(), set(), set()
    exterior_lines = set()
    optimum_polygon_lines = get_lines(optimum_polygons)

    for polygon in optimum_polygons:
        for line in polygon.exterior_lines:
            ls = shg.LineString(line)
            p1 = ls.coords[0]
            p2 = ls.coords[1]
            exterior_lines.add((p1, p2))
            V.add(p1)
            V.add(p2)

    for line in optimum_polygon_lines:
        ls = shg.LineString(line)
        p1 = ls.coords[0]
        p2 = ls.coords[1]
        ls = get_extrapolated_line(p1, p2)
        intersections = get_intersections(ls, exterior_lines)
        points = []

        for i in intersections:
            points.append(i)

        if len(points) > 0:
            V.add(points[0])
            V.add(points[1])
            R.add((points[0], points[1]))
            R.add((points[1], points[0]))
            E.add((points[0], points[1]))
            E.add((points[1], points[0]))

    for ext_edge in exterior_lines:
        ext_ls = shg.LineString(ext_edge)
        points = []
        for point in V:
            p1 = shg.Point(point)

            if ext_ls.distance(p1) < distance_threshold:
                points.append(point)

        points = sorted(points)
        temp = shg.LineString(points)
        temp = segments(temp)

        for line in temp:
            p1, p2 = line.coords[0], line.coords[1]
            E.add((p1, p2))
            E.add((p2, p1))

    logger.debug(f"Length of V: {len(V)}")
    logger.debug(f"Length of E: {len(E)}")
    logger.debug(f"Length of R: {len(R)}")

    if show:
        plt.figure(figsize=(20, 20))

        for edge in E:
            plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c='b', lw=5)

        for point in V:
            plt.plot(point[0], point[1], marker='o', color='red', markersize=15)

        plt.gca().set_aspect('equal', 'box')
        plt.grid(axis='both')
        plt.title("Graph Representation", fontsize=50)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
        plt.show()

    V, E, R = list(V), list(E), list(R)

    return V, E, R


@logger.trace()
def get_distance_matrix(V, E, R):
    num_nodes = len(V)
    mapping = {}
    reverse_mapping = {}
    distance_matrix = np.ones([num_nodes, num_nodes])

    for i in range(len(V)):
        mapping[i] = V[i]
        reverse_mapping[V[i]] = i

    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                p1 = mapping[i]
                p2 = mapping[j]

                if (p1, p2) in R:
                    distance_matrix[i][j] = config['REQUIRED_EDGE_COST']
                elif (p1, p2) in E:
                    distance_matrix[i][j] = np.linalg.norm(np.asarray(p2) - np.asarray(p1)) * config['EXISTING_EDGE_COST']
                else:
                    distance_matrix[i][j] = np.linalg.norm(np.asarray(p2) - np.asarray(p1)) * config['DEFAULT_EDGE_COST']

    logger.debug(f"Distance Matrix Shape: {distance_matrix.shape}")
    return distance_matrix, mapping


def add_arrow(line, direction='right', size=30, color=None):
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    start_ind = len(xdata) // 2

    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="fancy", color=color),
        size=size
    )


def choose_region(idx, show=True):
    if idx == 1:
        field_poly = FieldPoly(shg.Polygon([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], holes=[[[200, 200],
                                                                                                 [200, 800],
                                                                                                 [800, 800],
                                                                                                 [800, 200]]]))
    elif idx == 2:
        field_poly = FieldPoly(shg.Polygon([[0, 0], [400, 0], [400, 400], [600, 400], [600, 0], [1000, 0], [1000, 600],
                                            [800, 700], [750, 800], [1000, 750], [1000, 1000], [0, 1000], [0, 550],
                                            [500, 650], [500, 550], [0, 450]]))
    elif idx == 3:
        field_poly = FieldPoly(
            shg.Polygon([[0, 0], [575, 0], [575, 500], [425, 500], [425, 300], [500, 300], [500, 200],
                         [300, 200], [300, 650], [700, 650], [700, 0], [1000, 0], [1000, 1000],
                         [0, 1000]], holes=[[[350, 850], [400, 750], [600, 850], [450, 925]]]))
    elif idx == 4:
        field_poly = FieldPoly(shg.Polygon([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], holes=[[[100, 400], [200, 300],
                                                                                                 [300, 300], [350, 450],
                                                                                                 [300, 500],
                                                                                                 [150, 450]],
                                                                                                [[750, 300], [750, 100],
                                                                                                 [850, 100],
                                                                                                 [850, 300]],
                                                                                                [[400, 700], [850, 500],
                                                                                                 [950, 600], [900, 650],
                                                                                                 [850, 600], [600, 700],
                                                                                                 [700, 800],
                                                                                                 [600, 900]]]))
    elif idx == 5:
        field_poly = FieldPoly(shg.Polygon([[0, 0], [400, 0], [400, 500], [200, 500], [200, 600], [500, 600], [500, 0],
                                            [1000, 0], [1000, 450], [700, 450], [700, 550], [1000, 550], [1000, 1000],
                                            [0, 1000]], holes=[[[200, 800], [300, 700], [600, 800], [300, 900]],
                                                               [[700, 900], [700, 700], [800, 700],
                                                                [800, 900]]]))
    elif idx == 6:
        field_poly = FieldPoly(shg.Polygon([[0, 0], [1000, 0], [1000, 1000], [0, 1000]]))
    else:
        field_poly = FieldPoly.synthesize(cities_count=5, hole_count=1, hole_cities_count=5, poly_extent=1000)

    if show:
        plt.figure(figsize=(20, 20))
        initial_lines_count = len(get_field_lines(field_poly, show=False))
        lines = get_field_lines(field_poly)

        for line in lines:
            line = shg.LineString(line)
            p1, p2 = line.coords.xy
            plt.plot(p1, p2, lw=5)

        field_poly.plot(lw=5)
        temp_x, temp_y = field_poly.polygon.centroid.xy
        temp_x, temp_y = float(temp_x[0]), float(temp_y[0])
        plt.text(temp_x, temp_y, f"Cost: {initial_lines_count}", ha='center', va='center', fontsize=40)
        plt.gca().set_aspect('equal', 'box')
        plt.grid(axis='both')
        plt.title("Original Polygon", fontsize=50)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
        plt.show()

    return field_poly


def swap_indices(distance_matrix, idx):
    new_distance_matrix = np.array(distance_matrix)
    new_distance_matrix[[0, idx]] = new_distance_matrix[[idx, 0]]
    new_distance_matrix[:, [0, idx]] = new_distance_matrix[:, [idx, 0]]
    new_distance_matrix[:, 0] = 0
    return new_distance_matrix


def best_LKH_path(E, R, distance_matrix, mapping):
    lowest_cost = 1e10
    lowest_path = None

    for i in range(len(distance_matrix) - 1):
        for j in range(i + 1, len(distance_matrix)):
            path = solve_tsp(distance_matrix, endpoints=(i, j))
            unique_edges = set()
            cost = 0

            for k in range(len(path) - 1):
                p1 = mapping[path[k]]
                p2 = mapping[path[k + 1]]

                if (p1, p2) in R and (p1, p2) not in unique_edges:
                    unique_edges.add((p1, p2))

                cost += distance_matrix[path[k], path[k + 1]]

            if cost < lowest_cost:
                lowest_cost = cost
                lowest_path = path

    # for i in range(len(distance_matrix)):
    #     new_distance_matrix = swap_indices(distance_matrix, i)
    #     path = solve_tsp_simulated_annealing(new_distance_matrix)[0]
    #     # path = elkai.solve_float_matrix(new_distance_matrix)
    #     unique_edges = set()
    #     cost = 0
    #
    #     for i in range(len(path) - 1):
    #         p1 = mapping[path[i]]
    #         p2 = mapping[path[i + 1]]
    #
    #         if (p1, p2) in R and (p1, p2) not in unique_edges:
    #             unique_edges.add((p1, p2))
    #
    #         cost += new_distance_matrix[path[i], path[i+1]]
    #
    #     if cost > lowest_cost:
    #         lowest_cost = cost
    #         lowest_path = path

    unique_edges = set()
    covered_required = 0

    for edge in E:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c='skyblue')

    for i in range(len(lowest_path) - 1):
        p1 = mapping[lowest_path[i]]
        p2 = mapping[lowest_path[i + 1]]
        xs = np.linspace(p1[0], p2[0], 100)
        ys = np.linspace(p1[1], p2[1], 100)
        line = plt.plot(xs, ys, c='orange', linewidth=2)[0]
        add_arrow(line, color='green')

        if i == 0:
            plt.plot(p1[0], p1[1], marker='o', color='blue', markersize=5)
        if i == len(lowest_path) - 2:
            plt.plot(p2[0], p2[1], marker='o', color='red', markersize=5)

        if (p1, p2) in R and (p1, p2) not in unique_edges:
            unique_edges.add((p1, p2))
            covered_required += 1

    logger.info(f"\n-------------LKH Solver Results-------------")
    logger.info(f"Covered required edges: {covered_required} out of {len(R) // 2}")
    logger.info(f"Path cost: {lowest_cost}")
    plt.gca().set_aspect('equal', 'box')
    plt.grid(axis='both')
    plt.title("LKH Solver")
    plt.tight_layout()
    plt.show()


def best_a_star_path(V, E, R, distance_matrix, mapping):
    min_cost = config['DEFAULT_EDGE_COST']
    max_covered = -1
    min_path = None

    plt.figure(figsize=(20, 20))

    for i in range(len(V) - 1):
        for j in range(i + 1, len(V)):
            unique_edges = set()
            covered_required = 0
            path = find_a_star_path(distance_matrix, mapping, start=V[i], end=V[j])

            # for k in range(len(path) - 1):
            #     p1 = tuple(map(float, path[k][0][1:-1].split(',')))
            #     p2 = tuple(map(float, path[k + 1][0][1:-1].split(',')))
            #
            #     if (p1, p2) in R and (p1, p2) not in unique_edges:
            #         unique_edges.add((p1, p2))
            #         covered_required += 1
            #
            # if covered_required >= max_covered:
            #     max_covered = covered_required
            #     min_path = path

            cost = float(path[-1][1])

            if cost < min_cost:
                min_cost = cost
                min_path = path

    path = min_path

    for edge in E:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c='skyblue', lw=5)

    unique_edges = set()
    covered_required = 0

    for i in range(len(path) - 1):
        p1 = tuple(map(float, path[i][0][1:-1].split(',')))
        p2 = tuple(map(float, path[i + 1][0][1:-1].split(',')))
        xs = np.linspace(p1[0], p2[0], 100)
        ys = np.linspace(p1[1], p2[1], 100)
        line = plt.plot(xs, ys, c='orange', linewidth=5)[0]
        add_arrow(line, color='green')

        if i == 0:
            plt.plot(p1[0], p1[1], marker='o', color='blue', markersize=15)
        if i == len(path) - 2:
            plt.plot(p2[0], p2[1], marker='o', color='red', markersize=15)

        if (p1, p2) in R and (p1, p2) not in unique_edges:
            unique_edges.add((p1, p2))
            covered_required += 1

    logger.info(f"\n-------------A* Search Results-------------")
    logger.info(f"Covered required edges: {covered_required} out of {len(R) // 2}")
    logger.info(f"Path cost: {min_cost}")
    plt.text(500, 500, f"Cost: {covered_required}", ha='center', va='center', fontsize=40)
    plt.gca().set_aspect('equal', 'box')
    plt.grid(axis='both')
    plt.title("A* Search", fontsize=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.show()


def main():
    field_poly = choose_region(idx=1, show=True)
    optimum_polygons = perform_optimization(field_poly, use_mp=False)
    plot_optimum_polygons(optimum_polygons, field_poly)
    V, E, R = polygons_to_graph(optimum_polygons, show=True)
    distance_matrix, mapping = get_distance_matrix(V, E, R)

    # Finding a path using LKH TSP Solver
    # best_LKH_path(E, R, distance_matrix, mapping)

    # Finding a path using A* Search
    best_a_star_path(V, E, R, distance_matrix, mapping)


if __name__ == "__main__":
    main()
