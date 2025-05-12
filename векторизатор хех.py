# built-in python libs
import os
import math
import random
import shutil
import logging

# python libs
import numpy as np
from matplotlib import pyplot as plt
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


def white(rgb):
    degree = 100
    for i in range(3):
        if 255 - degree > rgb[i]:
            return False
        return True


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


#function is quite bad in current condition, but nevertheless works
def point_adding_probability(approximating_points_count, num_points, diameter, minimal_dist):
    probability_const = approximating_points_count / diameter
    adding_probability = 1 - min(0.95, (probability_const ** 2) / (math.e ** minimal_dist))
    return adding_probability

def generate_initial_non_self_intersecting_polyline_points(boundary_points, num_points, diameter, approximating_points_count):
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

        adding_probability = point_adding_probability(approximating_points_count, num_points, diameter, minimal_dist)
        if random.random() > adding_probability:
            if len(boundary_points) != 0:
                boundary_processed_points.append(cur_point)
        else:
            cur_points.append(cur_point)
        if len(boundary_points):
            boundary_points.remove(cur_point)
    return cur_points

def generate_initial_non_self_intersecting_polyline(boundary_points, num_points, diameter, approximating_points_count):
    if len(boundary_points) < num_points:
        return None

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
        length += math.sqrt((polyline[i + 1][0] - polyline[i][0]) ** 2 + (polyline[i + 1][1] - polyline[i][1]) ** 2)
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


def ask_user_for_data(text, type="int", default=""):
    user_data = None
    while True:
        try:
            query_text = text + (f" (default is \"{default}\")" if default else "") + ":"
            user_data = input(query_text)
            if user_data == "":
                user_data = default
            if type == "int":
                user_data = int(user_data)
            elif type == "string":
                break
            elif type == "image":
                user_data = Image.open(user_data)
            break
        except:
            if type == "int":
                print("Not correct value is given, try again")
            elif type == "image":
                print("File not found, try again")
    return user_data


def print_boundary_points(boundary_points, width, height, filename):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for point in boundary_points:
        draw.point((point[1], point[0]), fill="red")
    os.makedirs("vectorizator_debug_files", exist_ok=True)
    image.save("vectorizator_debug_files/" + filename + "_boundary_points" + ".png")


def main():
    #prepare and get data
    shutil.rmtree("vectorizator_debug_files", ignore_errors=True)
    os.makedirs("vectorizator_debug_files", exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                        handlers=[
                            logging.FileHandler('vectorizator_debug_files/debug.log'),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    image = ask_user_for_data("File", type="image", default="assets/test_1.png")
    pixel_array = np.array(image)
    scale_parameter = ask_user_for_data("Scale parameter", default="1")
    max_colors_amount = ask_user_for_data("Max amount of colors", default="5")
    monte_carlo_color_optimization_iterations_amount = ask_user_for_data("Amount of monte carlo optimization iterations", default="100")
    connected_components_color_tolerance = ask_user_for_data("Color tolerance for different elements of connected components in dsu", default="10")
    initial_polyline_num_points = ask_user_for_data("Initial polyline points amount", default="25")
    attempts_to_generate_initial_polyline = ask_user_for_data("Attempt to generate initial polyline", default="10000")
    optimization_of_polyline_iterations_amount = ask_user_for_data("Iterations of polyline optimization", default="1000")
    save_result_file_as = ask_user_for_data("Save result file as", type="string", default="res")
    colors_palette = get_image_colors_palette(pixel_array)
    logger.info("Monte carlo color optimization")
    best_colors = monte_carlo_color_optimization(pixel_array, colors_palette, max_colors_amount, monte_carlo_color_optimization_iterations_amount)
    logger.info("Approximating image")
    approximated_image = approximate_image(pixel_array, best_colors[0])
    logger.info("Dividing image into components")
    approximated_image_dsu = find_connected_components_dsu_color(approximated_image, connected_components_color_tolerance)
    approximated_image_components = dict()
    rows, columns, _ = approximated_image.shape
    for r in range(rows):
        for c in range(columns):
            component_parent = approximated_image_dsu.find(pixel_index(r, c, columns))
            if component_parent in approximated_image_components:
                approximated_image_components[component_parent].append((r, c))
            else:
                approximated_image_components[component_parent] = [(r, c)]

    output_image = Image.new('RGB', (columns * scale_parameter, rows * scale_parameter), (255, 255, 255))
    draw = ImageDraw.Draw(output_image)
    logger.info("Vectorizing components")
    for component in approximated_image_components:
        if not is_like_circle(approximated_image_components[component]):
            continue
        logger.info(f"Component {component}:")
        logger.info("Getting boundary points")
        boundary_points = find_approximate_boundary_points(approximated_image_components[component])
        print_boundary_points(boundary_points, columns, rows, str(component))
        initial_non_self_intersecting_polyline = None
        polygon_width = int(round(round(len(approximated_image_components[component]) / (math.pi * find_circle_properties(approximated_image_components[component])[1])) / 2))
        for attempt in range(attempts_to_generate_initial_polyline):
            logger.info("Creating initial polyline")
            initial_non_self_intersecting_polyline = generate_initial_non_self_intersecting_polyline(boundary_points, initial_polyline_num_points, find_circle_properties(approximated_image_components[component])[1], len(approximated_image_components[component]))
            if not(initial_non_self_intersecting_polyline is None) and not is_self_intersecting(LineString(initial_non_self_intersecting_polyline)):
                break
            print("Fail, one more attempt...")
        else:
            print("Failed to generate initial polyline for current component")
            continue

        initial_polyline_image = Image.new("RGB", (columns, rows), "white")
        initial_polyline_draw = ImageDraw.Draw(initial_polyline_image)
        initial_polyline_draw.line(initial_non_self_intersecting_polyline + [initial_non_self_intersecting_polyline[0]], fill="red", width=polygon_width*scale_parameter)
        initial_polyline_image.save("vectorizator_debug_files/" + str(component) + "_initial_polyline" + ".png")

        initial_polyline_image = Image.new("RGB", (columns, rows), "white")
        initial_polyline_draw = ImageDraw.Draw(initial_polyline_image)
        for point in initial_non_self_intersecting_polyline:
            initial_polyline_draw.point((point[0], point[1]), fill="red")
        initial_polyline_image.save("vectorizator_debug_files/" + str(component) + "_initial_polyline_dots" + ".png")

        logger.info("Optimization of initial polyline")
        optimized_polyline = optimize_polyline(initial_non_self_intersecting_polyline, approximated_image_components[component], optimization_of_polyline_iterations_amount)

        optimized_polyline_image = Image.new("RGB", (columns, rows), "white")
        optimized_polyline_draw = ImageDraw.Draw(optimized_polyline_image)
        optimized_polyline_draw.line(optimized_polyline + [optimized_polyline[0]], fill="red", width=polygon_width*scale_parameter)
        optimized_polyline_image.save("vectorizator_debug_files/" + str(component) + "_optimized_polyline" + ".png")

        logger.info("Creating bezier curve")
        final_bezier_curve = fit_bezier_curve(optimized_polyline, approximated_image_components[component])
        component_color = tuple(approximated_image[approximated_image_components[component][0][0]][approximated_image_components[component][0][1]])
        final_bezier_curve = [(x, y) for y, x in final_bezier_curve]
        draw_points = [(x * scale_parameter, y * scale_parameter) for x, y in final_bezier_curve + [final_bezier_curve[0]]]
        draw.line(draw_points, fill=component_color, width=polygon_width*scale_parameter)

    output_image.save(f"{save_result_file_as}.png")
    output_image.show()


if __name__ == "__main__":
    main()
