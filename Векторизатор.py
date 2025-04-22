# built-in python libs
import math
import random

# python libs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from shapely import Point, LineString
from shapely.geometry import Polygon
from scipy.spatial import Delaunay


class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]

    def union(self, a, b):
        rootA = self.find(a)
        rootB = self.find(b)
        if rootA != rootB:
            if self.rank[rootA] > self.rank[rootB]:
                self.parent[rootB] = rootA
            elif self.rank[rootA] < self.rank[rootB]:
                self.parent[rootA] = rootB
            else:
                self.parent[rootB] = rootA
                self.rank[rootA] += 1


def coords(picture):
    result = []
    for i, row in enumerate(picture):
        for j, value in enumerate(row):
            if value == 1:
                result.append((i, j))
    return result


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def max_dist(points):
    max_distance = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance(points[i], points[j])
            if dist > max_distance:
                max_distance = dist
    return max_distance


def min_dist(points):
    min_distance = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance(points[i], points[j])
            if dist < min_distance:
                min_distance = dist
    return min_distance


def are_connected(picture, point1, point2):
    rows, columns = len(picture), len(picture[0])
    dsu = DSU(rows * columns)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for r in range(rows):
        for c in range(columns):
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < columns and picture[r][c] == picture[nr][nc]:
                    dsu.union(r * columns + c, nr * columns + nc)
    p1_row, p1_col = point1
    p2_row, p2_col = point2
    if picture[p1_row][p1_col] != picture[p1_row][p1_row]:
        return False
    return dsu.find(p1_row * columns + p1_col) == dsu.find(p2_row * columns + p2_col)


def white(rgb):
    degree = 100
    for i in range(3):
        if 255 - degree > rgb[i]:
            return False
        return True


def segment_picture(picture):
    rows, columns = len(picture), len(picture[0])
    dsu = DSU(rows * columns)
    for i in range(rows):
        for j in range(columns):
            if white(picture[i][j]):
                if i + 1 < rows:
                    if white(picture[i + 1][j]):
                        dsu.union(i * columns + j, (i + 1) * columns + j)
                if j + 1 < columns:
                    if white(picture[i][j + 1]):
                        dsu.union(i * columns + j, i * columns + j + 1)
    components = {}
    for i in range(rows):
        for j in range(columns):
            if picture[i][j][0] == 255:
                root = dsu.find(i * columns + j)
                base_root = dsu.find(0)
                if root != base_root:
                    if root not in components:
                        components[root] = []
    components[root].append((i, j))

    circles = []
    for i in components:
        x_coord = 0
        y_coord = 0
        for j in components[i]:
            x_coord += j[1]
            y_coord += j[0]
        x_coord /= len(components[i])
        y_coord /= len(components[i])
        circles.append([len(components[i]), math.sqrt(len(components[i]) / math.pi), [x_coord, y_coord]])
    rad = round(circles[-1][1], 3)
    center = circles[-1][2]


def color_dist(first, second):
    return sum(abs(int(f)-int(s)) for f,s in zip(first,second))


def calculate_covering_radius(image, approximating_colors, distance_function=color_dist):
    max_distance = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            min_dist = min(distance_function(pixel, color) for color in approximating_colors)
            max_distance = max(max_distance, min_dist)
    return max_distance


def monte_carlo_color_optimization(image, colors_palette, max_colors, iterations=10000, distance_function=color_dist):
    best_colors = None
    min_radius = float('inf')
    for _ in range(iterations):
        num_colors = random.randint(1, min(len(colors_palette), max_colors))
        random_colors = []
        current_palette = list(colors_palette)
        for _ in range(num_colors):
            current_color = random.choice(current_palette)
            current_palette.remove(current_color)
            random_colors.append(current_color)
            random_colors_np = np.array(random_colors)
            radius = calculate_covering_radius(image, random_colors_np, distance_function)
            if radius < min_radius:
                min_radius = radius
                best_colors = random_colors
    return best_colors, min_radius


def approximate_image(image, approximating_colors, distance_function=color_dist):
    new_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            #closest_color = min(approximating_colors, key=lambda color: distance_function(pixel, color))
            closest_color = approximating_colors[0]
            min_clr_dist = 255 * 4
            for color in approximating_colors:
                cur_dist = distance_function(pixel, color)
                if cur_dist < min_clr_dist:
                    min_clr_dist = cur_dist
                    closest_color = color
            new_image[i, j] = closest_color
    return new_image.astype(np.uint8)


def pixel_index(r, c, columns):
    return r * columns + c


def find_connected_components_dsu_color(image, color_tolerance=10):
    rows, columns, _ = image.shape
    dsu = DSU(rows * columns)

    for r in range(rows):
        for c in range(columns):
            current_pixel = image[r, c]
            if c + 1 < columns:
                neighbour_pixel = image[r, c + 1]
                if color_dist(current_pixel, neighbour_pixel) <= color_tolerance:
                    dsu.union(pixel_index(r, c, columns), pixel_index(r, c + 1, columns))
            if r + 1 < rows:
                neighbour_pixel = image[r + 1, c]
                if color_dist(current_pixel, neighbour_pixel) <= color_tolerance:
                    dsu.union(pixel_index(r, c, columns), pixel_index(r + 1, c, columns))
    return dsu


def get_component_pixels(dsu, image, component_root):
    rows, columns, _ = image.shape
    pixels = []
    for r in range(rows):
        for c in range(columns):
            if dsu.find(r * columns + c) == component_root:
                pixels.append((c, r))
    return pixels


def is_convex(points):
    n = len(points)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            num_steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
            for step in range(num_steps):
                x = int(x1 + (x2 - x1) * step / num_steps)
                y = int(y1 + (y2 - y2) * step / num_steps)
                if (x, y) not in points:
                    return False
    return True


def is_like_circle(points, tolerance=0.2):
    if not points:
        return False
    sum_x = sum(x for x, y in points)
    sum_y = sum(y for x, y in points)
    center_x = sum_x / len(points)
    center_y = sum_y / len(points)
    distances = [math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in points]
    avg_radius = sum(distances) / len(distances)
    deviations = [(abs(d - avg_radius) / avg_radius) for d in distances]
    num_outliers = sum(1 for dev in deviations if dev > tolerance)
    if num_outliers / len(points) > tolerance:
        return False
    return True


def find_circle_properties(points):
    if not points:
        return None, None
    sum_x = sum(x for x, y in points)
    sum_y = sum(y for x, y in points)
    center_x = sum_x / len(points)
    center_y = sum_y / len(points)
    distances = [math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in points]
    avg_radius = sum(distances) / len(distances)
    return (center_x, center_y), avg_radius


def approximate_boundary_with_polyline(points, tolerance):
    polygon = Polygon(points)
    simplified_polygon = polygon.simplify(tolerance, preserve_topology=False)
    return simplified_polygon


def angle_at_point(p1, p2, point):
    dx1 = p1[0] - point[0]
    dy1 = p1[1] - point[1]
    dx2 = p2[0] - point[0]
    dy2 = p2[1] - point[1]
    cross_product = dx1 * dy2 - dy1 * dx2
    dot_product = dx1 * dx2 + dy1 * dy2
    angle = np.arctan2(cross_product, dot_product)
    return angle


def calculate_winding_number(point, polygon):
    winding_number = 0
    n = len(polygon.exterior.coords)
    for i in range(n - 1):
        p1 = polygon.exterior.coords[i]
        p2 = polygon.exterior.coords[i + 1]
        angle = angle_at_point(p1, p2, point)
        winding_number += angle
    return round(winding_number / (2 * np.pi))


def is_left(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[0] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def calculate_symmetric_difference_area(polygon, circle):
    symmetric_difference = polygon.symmetric_difference(circle)
    return symmetric_difference.area


def create_circle(center, radius, num_segments=360):
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return Polygon(zip(x, y))


def calculate_convexity(polygon):
    if polygon.area == 0:
        return 0
    return polygon.area / polygon.convex_hull.area


def calculate_circularity(polygon):
    area = polygon.area
    perimeter = polygon.length
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter ** 2)


def calculate_radius_variation(polygon, center):
    distances = [Point(p).distance(Point(center)) for p in polygon.exterior.coords]
    return np.std(distances)


def calculate_area_ratio(polygon, circle):
    return polygon.area / circle.area


def is_self_intersecting(polygon):
    return not polygon.is_simple


def find_approximate_boundary_points(points, distance=2):
    boundary_points = []
    point_set = set(points)
    for x, y in points:
        nearby = False
        not_nearby = False
        for dx in [-distance, 0, distance]:
            for dy in [-distance, 0, distance]:
                if dx == 0 and dy == 0:
                    continue
                neighbour_x = x + dx
                neighbour_y = y + dy
                neighbour = (neighbour_x, neighbour_y)
                if neighbour in point_set:
                    nearby = True
                else:
                    not_nearby = True
                if nearby and not_nearby:
                    boundary_points.append((x, y))
                    break
            if nearby and not_nearby:
                break
    return list(set(boundary_points))


def generate_initial_non_self_intersecting_polyline_points(boundary_points, num_points, diameter, approximating_points_count):
    probability_const = approximating_points_count / diameter
    cur_points = list()
    boundary_processed_points = list()
    while len(cur_points) < num_points:
        if len(boundary_points) > 0:
            cur_point = random.choice(boundary_points)
        else:
            cur_point = random.choice(boundary_processed_points)
        minimal_dist = diameter * 2
        for generated_point in cur_points:
            if generated_point == cur_point:
                continue
            minimal_dist = min(minimal_dist, distance(cur_point, generated_point))
        adding_probability = 1 - 1 / (1 + probability_const * diameter)
        if random.random() > adding_probability:
            if len(boundary_points) != 0:
                boundary_processed_points.append(cur_point)
        else:
            cur_points.append(cur_point)
        boundary_points.remove(cur_point)
    return cur_points

def generate_initial_non_self_intersecting_polyline(boundary_points, num_points, diameter, approximating_points_count):
    if len(boundary_points) < num_points:
        return None
    # selected_points = random.sample(boundary_points, num_points)
    selected_points = generate_initial_non_self_intersecting_polyline_points(boundary_points, num_points, diameter, approximating_points_count)
    try:
        tri = Delaunay(selected_points)
    except Exception as e:
        return None
    current_point = random.choice(selected_points)
    polyline = [current_point]
    remaining_points = set(selected_points)
    remaining_points.remove(current_point)
    while remaining_points:
        nearest_point = None
        min_distance = float('inf')
        for p in remaining_points:
            distance = math.sqrt(((current_point[0] - p[0]) ** 2) + ((current_point[1] - p[1]) ** 2))
            if distance < min_distance:
                min_distance = distance
                nearest_point = p
        new_line = LineString([current_point, nearest_point])
        intersects = False
        intersects = is_self_intersecting(LineString(polyline + [nearest_point]))
        if intersects:
            remaining_points.remove(nearest_point)
            if not remaining_points:
                return None
            continue
        polyline.append(nearest_point)
        current_point = nearest_point
        remaining_points.remove(nearest_point)
    return polyline


def calculate_polyline_length(polyline):
    length = 0
    for i in range(len(polyline) - 1):
        length += ((polyline[i + 1][0] - polyline[i][0]) ** 2 + (polyline[i + 1][1] - polyline[i][1]) ** 2) ** 0.5
    return length


def optimize_polyline(initial_polyline, points, num_iterations=1000, step_size=0.1):
    current_polyline = initial_polyline
    best_polyline = initial_polyline
    best_length = calculate_polyline_length(initial_polyline)
    for _ in range(num_iterations):
        index_to_modify = random.randint(1, len(current_polyline) - 2)
        new_x = current_polyline[index_to_modify][0] + random.uniform(-step_size, step_size)
        new_y = current_polyline[index_to_modify][1] + random.uniform(-step_size, step_size)
        new_point = (new_x, new_y)
        new_polyline = current_polyline[:index_to_modify] + [new_point] + current_polyline[index_to_modify + 1:]
        line = LineString(new_polyline)
        if not line.is_simple:
            continue
        new_length = calculate_polyline_length(new_polyline)
        if new_length < best_length:
            best_length = new_length
            best_polyline = new_polyline
        current_polyline = best_polyline
    return best_polyline


def bernstein_poly(i, n, t):
    return math.comb(n, i) * (t ** i) * (1 - t) ** (n - i)


def bezier_curve(points, n_times=1000):
    n_points = len(points)
    x_points = np.array([p[0] for p in points])
    y_points = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, n_times)
    polynomial_array = np.array([bernstein_poly(i, n_points - 1, t) for i in range(0, n_points)])
    x_values = np.dot(x_points, polynomial_array)
    y_values = np.dot(y_points, polynomial_array)
    return list(zip(x_values, y_values))


def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def symmetric_difference_area(polygon_points, bezier_points):
    polygon = Polygon(polygon_points)
    bezier_polygon = Polygon(bezier_points)
    symmetric_difference = polygon.symmetric_difference(bezier_polygon)
    if symmetric_difference.geom_type == 'MultiPolygon':
        return sum([p.area for p in symmetric_difference.geoms])
    else:
        return symmetric_difference.area


def fit_bezier_curve(polyline, points, n_times=1000, optimization_iterations=100):
    control_points = polyline[:]
    best_control_points = control_points[:]
    min_symmetric_difference = float('inf')
    original_polygon = Polygon(points)
    for _ in range(optimization_iterations):
        perturbed_control_points = []
        for i, (x, y) in enumerate(control_points):
            if i == 0 or i == len(control_points) - 1:
                perturbed_control_points.append((x, y))
            else:
                step_size = 0.05
                new_x = x + random.uniform(-step_size, step_size)
                new_y = y + random.uniform(-step_size, step_size)
                perturbed_control_points.append((new_x, new_y))
        bezier_points = bezier_curve(perturbed_control_points, n_times)
        try:
            bezier_polygon = Polygon(bezier_points)
            symmetric_diff = original_polygon.symmetric_difference(bezier_polygon)
            if symmetric_diff.geom_type == 'MultiPolygon':
                symmetric_difference_area_val = sum([p.area for p in symmetric_diff.geoms])
            else:
                symmetric_difference_area_val = symmetric_diff.area
            if symmetric_difference_area_val < min_symmetric_difference:
                min_symmetric_difference = symmetric_difference_area_val
                best_control_points = perturbed_control_points[:]
        except Exception as e:
            continue
        control_points = best_control_points[:]
    final_bezier_points = bezier_curve(best_control_points, n_times)
    return final_bezier_points


def get_image_colors_palette(image):
    rows, columns, _ = image.shape
    colors_palette = set()
    for r in range(rows):
        for c in range(columns):
                colors_palette.add(tuple(image[r, c]))
    return colors_palette



def ask_user_for_int_data(text):
    res = 0
    while True:
        try:
            res = int(input(text))
            break
        except:
            print("Not correct value is given, try again")
    return res


def main():
    image = Image.open(input("Filename: "))
    pixel_array = np.array(image)
    max_colors_amount = ask_user_for_int_data("Max amount of colors: ")
    monte_carlo_color_optimization_iterations_amount = ask_user_for_int_data("Amount of monte carlo optimization iterations: ")
    connected_components_color_tolerance = ask_user_for_int_data("Color tolerance for different elements of connected components in dsu: ")
    initial_polyline_num_points = ask_user_for_int_data("Initial polyline points amount: ")
    attempts_to_generate_initial_polyline = ask_user_for_int_data("Attempt to generate initial polyline: ")
    colors_palette = get_image_colors_palette(pixel_array)
    best_colors = monte_carlo_color_optimization(pixel_array, colors_palette, max_colors_amount, monte_carlo_color_optimization_iterations_amount)
    approximated_image = approximate_image(pixel_array, best_colors[0])
    approximated_image_dsu = find_connected_components_dsu_color(approximated_image, connected_components_color_tolerance)
    approximated_image_components = dict()
    rows, columns, _ = approximated_image.shape
    output_image = Image.new('RGB', (columns, rows), (255, 255, 255))
    draw = ImageDraw.Draw(output_image)
    for r in range(rows):
        for c in range(columns):
            component_parent = approximated_image_dsu.find(pixel_index(r, c, columns))
            if component_parent in approximated_image_components:
                approximated_image_components[component_parent].append((r, c))
            else:
                approximated_image_components[component_parent] = [(r, c)]

    for component in approximated_image_components:
        if not is_like_circle(approximated_image_components[component]):
            continue
        boundary_points = find_approximate_boundary_points(approximated_image_components[component])
        initial_non_self_intersecting_polyline = None
        for attempt in range(attempts_to_generate_initial_polyline):
            initial_non_self_intersecting_polyline = generate_initial_non_self_intersecting_polyline(boundary_points, initial_polyline_num_points, find_circle_properties(approximated_image_components[component])[1], len(approximated_image_components[component]))
            if not(initial_non_self_intersecting_polyline is None) and not is_self_intersecting(LineString(initial_non_self_intersecting_polyline)):
                break
            print("Fail, one more attempt...")
        else:
            print("Failed to generate initial polyline for current component")
            continue
        optimized_polyline = optimize_polyline(initial_non_self_intersecting_polyline, approximated_image_components[component])
        final_bezier_curve = fit_bezier_curve(optimized_polyline, approximated_image_components[component])
        component_color = tuple(approximated_image[approximated_image_components[component][0][0]][approximated_image_components[component][0][1]])
        final_bezier_curve = [(x, y) for y, x in final_bezier_curve]
        draw.line(final_bezier_curve + [final_bezier_curve[0]], fill=component_color, width=2)

    output_image.save("res.png")
    output_image.show()


if __name__ == "__main__":
    main()
