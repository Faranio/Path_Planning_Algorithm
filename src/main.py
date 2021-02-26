import collections
import itertools
import matplotlib.pyplot as plt
import more_itertools as mi
import numpy as np
import shapely.geometry as shg

from lgblkb_tools import logger
from lgblkb_tools.common.utils import ParallelTasker
from lgblkb_tools.geometry import FieldPoly

cut_len = 25
dfs_threshold = 45
min_dist = 1e-6
path_width = 15
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
	field_poly.plot()
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
			plt.plot(ls_x, ls_y)
	
	final_acr = field_poly.geometry.area / sum(costs)
	logger.info(f"Final Area Cost Ratio: {final_acr}")
	logger.info(f"Total Field Count: {total_field_count}")
	plt.gca().set_aspect('equal', 'box')
	plt.show()
	
	
@logger.trace()
def get_tsp_distance_matrix(field_poly):
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
	
	num_of_lines = len(all_lines)
	tsp_distance_matrix = np.zeros([num_of_lines * 2, num_of_lines * 2])
	
	for i in range(num_of_lines * 2):
		for j in range(num_of_lines * 2):
			if i == j or i == j + num_of_lines or j == i + num_of_lines:
				tsp_distance_matrix[i][j] = np.inf
				continue
			
			if i < num_of_lines:
				x1, y1 = list(all_lines[i].coords)[0]
			else:
				x1, y1 = list(all_lines[i - num_of_lines].coords)[0]
			
			if j < num_of_lines:
				x2, y2 = list(all_lines[j].coords)[-1]
			else:
				x2, y2 = list(all_lines[j - num_of_lines].coords)[-1]
			
			tsp_distance_matrix[i][j] = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))
	
	logger.debug(f"TSP Cost Matrix: {tsp_distance_matrix}")
	logger.debug(f"TSP Cost Matrix Shape: {tsp_distance_matrix.shape}")
	return tsp_distance_matrix


def main():
	field_poly = FieldPoly(shg.Polygon([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], holes=[[[200, 200],
	                                                                                         [200, 800],
	                                                                                         [800, 800],
	                                                                                         [800, 200]]])).plot()
	# field_poly = FieldPoly.synthesize(cities_count=9, hole_count=1, hole_cities_count=5, poly_extent=1000)
	
	initial_lines_count = get_field_lines_count(field_poly, show=True)
	logger.info(f"initial_lines_count: {initial_lines_count}")
	logger.info(f"Initial area_cost_ratio: {field_poly.area_cost_ratio}")
	field_poly.plot(f"Cost: {initial_lines_count}", lw=1)
	
	plt.gca().set_aspect('equal', 'box')
	plt.show()
	
	optimum_polygons = perform_optimization(field_poly, use_mp=False)
	logger.debug(f"Number of polygons: {len(optimum_polygons)}")
	
	plot_optimum_polygons(optimum_polygons, field_poly)
	tsp_distance_matrix = get_tsp_distance_matrix(optimum_polygons)
	
	
if __name__ == "__main__":
	main()
