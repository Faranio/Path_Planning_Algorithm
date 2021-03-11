import collections
import itertools
import matplotlib.pyplot as plt
import more_itertools as mi
import numpy as np
import shapely.geometry as shg

from lgblkb_tools import logger
from lgblkb_tools.common.utils import ParallelTasker
from lgblkb_tools.geometry import FieldPoly

cut_len = 0
dfs_threshold = 45
min_dist = 1e-6
path_width = 150  # 20
search_loop_limit = 3e5
workers_count = 4


def get_field_lines(field_poly: FieldPoly):
	field_poly = FieldPoly.as_valid(field_poly)
	field_lines = field_poly.get_field_lines_old(path_width, field_poly.get_min_cost(path_width).base_line, show=False)
	return field_lines


def get_field_lines_count(field_poly: FieldPoly, show=False):
	field_poly = FieldPoly.as_valid(field_poly)
	field_lines = field_poly.get_field_lines_old(path_width, field_poly.get_min_cost(path_width).base_line, show=show)
	return len(field_lines)


def get_paths(poly: FieldPoly, start, goal, loop_limit=5e6):
	graph = poly.adjacency_info
	
	if len(poly.G.decision_points['all']) < dfs_threshold:
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
				
				if some_field.geometry.area < min_dist:
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
	
	polygons = [FieldPoly.as_valid(x) for x in polygons if x.area > min_dist]
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
		plt.show()
	
	decomposed_polygons = [best.field]
	
	for path_field in get_paths(field_poly, start, goal, loop_limit=search_loop_limit):
		curr_count = next(counter)
		polygons = get_other_polys(field_poly, path_field)
		costs = [p.get_min_cost(path_width).cost for p in [path_field] + polygons]
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
			.set_run_params(some_point=points).run(workers_count=workers_count)
		
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
	polygons = decompose_from_points(field_poly, field_poly.get_outer_points(), use_mp=False, show=False)
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
	costs = []
	field_poly.plot(lw=1)
	total_field_count = 0
	all_lines = []
	
	for optimum_polygon in optimum_polygons:
		lines_count = get_field_lines_count(optimum_polygon, show=False)
		total_field_count += lines_count
		optimum_polygon.plot(f"Cost: {lines_count}", lw=1)
		costs.append(optimum_polygon.get_min_cost(path_width).cost)
		field_lines = get_field_lines(optimum_polygon)
		
		for line in field_lines:
			ls = shg.LineString(line.line)
			p1 = ls.interpolate(cut_len).coords[0]
			p2 = ls.interpolate(int(line.line.length - cut_len)).coords[0]
			ls = shg.LineString([shg.Point(p1), shg.Point(p2)])
			all_lines.append(ls)
			ls_x, ls_y = ls.xy
			plt.plot(ls_x, ls_y, c='r')
	
	final_acr = field_poly.geometry.area / sum(costs)
	logger.info(f"Final Area Cost Ratio: {final_acr}")
	logger.info(f"Total Field Count: {total_field_count}")
	plt.gca().set_aspect('equal', 'box')
	plt.show()
	
	
def get_lines(field_poly):
	all_lines = []
	
	if isinstance(field_poly, list):
		for optimum_polygon in field_poly:
			field_lines = get_field_lines(optimum_polygon)
			
			for line in field_lines:
				ls = shg.LineString(line.line)
				p1 = ls.interpolate(cut_len).coords[0]
				p2 = ls.interpolate(int(line.line.length - cut_len)).coords[0]
				ls = shg.LineString([shg.Point(p1), shg.Point(p2)])
				all_lines.append(ls)
	else:
		field_lines = get_field_lines(field_poly)
		
		for line in field_lines:
			ls = shg.LineString(line.line)
			p1 = ls.interpolate(cut_len).coords[0]
			p2 = ls.interpolate(int(line.line.length - cut_len)).coords[0]
			ls = shg.LineString([shg.Point(p1), shg.Point(p2)])
			all_lines.append(ls)
			
	return all_lines


def plot_directions(field_poly, permutation, optimum_polygons):
	field_poly.plot(lw=1)
	all_lines = get_lines(optimum_polygons)
	num_of_lines = len(all_lines)
	
	for i in range(len(permutation) - 1):
		idx_from, idx_to = permutation[i], permutation[i + 1]
		
		if idx_from < num_of_lines:
			x1, y1 = list(all_lines[idx_from].coords)[0]
		else:
			x1, y1 = list(all_lines[idx_from - num_of_lines].coords)[-1]
			
		if idx_to < num_of_lines:
			x2, y2 = list(all_lines[idx_to].coords)[0]
		else:
			x2, y2 = list(all_lines[idx_to - num_of_lines].coords)[-1]
			
		if i == 0:
			plt.plot((x1, x2), (y1, y2), lw=3, c='g')
		else:
			plt.plot((x1, x2), (y1, y2), lw=1, c='r')
	
	plt.gca().set_aspect('equal', 'box')
	plt.show()
	
	
def polygons_to_graph(optimum_polygons, show=False):
	V = set()
	E = set()
	R = set()
	optimum_polygon_lines = get_lines(optimum_polygons)
		
	for polygon in optimum_polygons:
		for line in polygon.exterior_lines:
			ls = shg.LineString(line)
			p1 = ls.interpolate(cut_len).coords[0]
			p2 = ls.interpolate(int(ls.length - cut_len)).coords[0]
			ls = shg.LineString([shg.Point(p1), shg.Point(p2)])
			ls_x, ls_y = ls.xy
			ls_x, ls_y = tuple([ls_x[0], ls_y[0]]), tuple([ls_x[1], ls_y[1]])
			E.add((ls_x, ls_y))
	
	for line in optimum_polygon_lines:
		ls = shg.LineString(line)
		p1 = ls.interpolate(cut_len).coords[0]
		p2 = ls.interpolate(int(line.length - cut_len)).coords[0]
		ls = shg.LineString([shg.Point(p1), shg.Point(p2)])
		ls_x, ls_y = ls.xy
		ls_x, ls_y = tuple([ls_x[0], ls_y[0]]), tuple([ls_x[1], ls_y[1]])
		E.add((ls_x, ls_y))
		R.add((ls_x, ls_y))
		R.add((ls_y, ls_x))

	real_edges = set()
	
	for edge in E:
		V.add(edge[0])
		V.add(edge[1])
		real_edges.add(edge)
		plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c='b')

	for point1 in V:
		for point2 in V:
			for edge in E:
				if point1 != point2:
					ls1 = shg.LineString(edge)
					p1 = shg.Point(point1)
					p2 = shg.Point(point2)

					if ls1.distance(p1) < 1e-8 and ls1.distance(p2) < 1e-8:
						ls = (point1, point2)
						real_edges.add(ls)

	for edge1 in real_edges.copy():
		ls1 = shg.LineString(edge1)

		for edge2 in real_edges.copy():
			if edge1 != edge2:
				p1 = shg.Point(edge2[0])
				p2 = shg.Point(edge2[1])
				ls2 = shg.LineString([p1, p2])
				if ls1.distance(p1) < 1e-8 and ls1.distance(p2) < 1e-8:
					if ls1.length > ls2.length:
						real_edges.remove(edge1)
						break
						
	if show:
		for point in V:
			plt.plot(point[0], point[1], marker='o', color='red', markersize=5)
	
	if show:
		plt.gca().set_aspect('equal', 'box')
		plt.show()

	E = real_edges
		
	V = sorted(V)
	E = sorted(E)
	R = sorted(R)
		
	return V, E, R


@logger.trace()
def get_distance_matrix(V, E, R):
	num_nodes = len(V)
	mapping = {}
	reverse_mapping = {}
	distance_matrix = np.ones([num_nodes, num_nodes]) * np.inf
	
	for i in range(len(V)):
		mapping[i] = V[i]
		reverse_mapping[V[i]] = i

	for edge in E:
		node_from = reverse_mapping[edge[0]]
		node_to = reverse_mapping[edge[1]]
		
		if edge in R:
			distance_matrix[node_from][node_to] = 0
		else:
			distance_matrix[node_from][node_to] = np.linalg.norm(np.asarray(edge[1]) - np.asarray(edge[0]))

	logger.debug(f"Distance Matrix Shape: {distance_matrix.shape}")
	return distance_matrix, mapping


def main():
	field_poly = FieldPoly(shg.Polygon([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], holes=[[[200, 200],
	                                                                                         [200, 800],
	                                                                                         [800, 800],
	                                                                                         [800, 200]]])).plot()
	# field_poly = FieldPoly.synthesize(cities_count=9, hole_count=1, hole_cities_count=5, poly_extent=1000)
	
	initial_lines_count = get_field_lines_count(field_poly, show=True)

	field_poly.plot(f"Cost: {initial_lines_count}", lw=1)
	plt.gca().set_aspect('equal', 'box')
	plt.show()

	optimum_polygons = perform_optimization(field_poly, use_mp=False)
	plot_optimum_polygons(optimum_polygons, field_poly)
	V, E, R = polygons_to_graph(optimum_polygons, show=True)
	distance_matrix, mapping = get_distance_matrix(V, E, R)

	i, j = np.where(distance_matrix != np.inf)

	for row, col in zip(i, j):
		p1 = mapping[row]
		p2 = mapping[col]
		xs = [p1[0], p2[0]]
		ys = [p1[1], p2[1]]
		plt.plot(xs, ys)

	plt.gca().set_aspect('equal', 'box')
	plt.show()


if __name__ == "__main__":
	main()
