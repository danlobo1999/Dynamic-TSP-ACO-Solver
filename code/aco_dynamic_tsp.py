import os
import random
import math
import numpy as np

def readFile():
    # filename = "test.tsp"
    filename = "berlin52.tsp"
    current_directory_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_directory_path, "..\data\\")
    file_path = os.path.join(data_path, filename)

    file_details = {}
    # Getting the file details
    with open(file_path, "r") as f:
        while True:
            line = f.readline().strip()

            if line.startswith("NAME"):
                file_details["Name"] = line.split(":")[1]

            elif line.startswith("DIMENSION"):
                file_details["Dimensions"] = int(line.split(":")[1])

            elif line.startswith("EDGE_WEIGHT_TYPE"):
                file_details["Edge Weight Type"] = line.split(":")[1].strip()

            elif line.startswith("NODE_COORD_SECTION"):
                break

            else:
                continue

        if file_details["Edge Weight Type"] != "EUC_2D":
            print(
                "Edge weight types are not Euclidean distances in 2D. Please try with another file."
            )

        else:
            # Creating a dictionary of the nodes along with their coordinates
            cities = {}
            for i in range(file_details["Dimensions"]):
                node, x, y = map(float, f.readline().strip().split())
                cities[int(node)] = (x, y)

        return cities


def distance(city1, city2):
    return round(math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2), 2)


def initialize_pheromone_matrix(n, initial_value):
    return [[initial_value for _ in range(n)] for _ in range(n)]


def calculate_probability(city, unvisited, pheromone, distance, alpha, beta):
    denominator = sum([(pheromone[city-1][j-1] ** alpha) * (distance[city-1][j-1] ** (-beta)) for j in unvisited])
    probabilities = [(pheromone[city-1][j-1] ** alpha) * (distance[city-1][j-1] ** (-beta)) / denominator for j in unvisited]
    return probabilities


def select_next_city(univisted, probabilities):
    return random.choices(univisted, weights=probabilities, k=1)[0]


def update_pheromone(pheromone, delta_pheromone, rho):
    n = len(pheromone)
    for i in range(n):
        for j in range(n):
            pheromone[i][j] = (1 - rho) * pheromone[i][j] + delta_pheromone[i][j]


def apply_traffic_fluctuation(base_distances, num_nodes, traffic_factors):
    traffic_factors_matrix = np.ones((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            base_factor = np.random.choice(traffic_factors)
            traffic_factors_matrix[i, j] = base_factor
            traffic_factors_matrix[j, i] = base_factor

    fluctuated_distances = np.copy(base_distances)
    fluctuated_distances *= traffic_factors_matrix

    return fluctuated_distances


def tsp_aco(cities, distances, tour_start, tour_current_city, num_ants, max_iterations, alpha, beta, rho, initial_pheromone):
    n = len(distances[0])
    pheromone = initialize_pheromone_matrix(n, initial_pheromone)
    best_tour = None
    best_tour_length = float("inf")

    for iteration in range(max_iterations):
        delta_pheromone = initialize_pheromone_matrix(n, 0)

        for ant in range(num_ants):
            if tour_current_city != None:
                start_city = tour_current_city
            else:
                start_city = random.choice(cities)
            current_city = start_city
            unvisited = cities.copy()

            if tour_start == None:
                unvisited.remove(current_city)

            tour = [current_city]

            while unvisited:
                probabilities = calculate_probability(current_city, unvisited, pheromone, distances, alpha, beta)
                next_city = select_next_city(unvisited, probabilities)
                tour.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city

            if tour_current_city != tour_start:
                tour.append(tour_start)
            else:
                tour.append(start_city)

            tour_length = sum([distances[tour[i]-1][tour[i + 1]-1] for i in range(len(tour)-1)])
            
            if tour_length < best_tour_length:
                best_tour_length = tour_length
                best_tour = tour

        for i in range(len(tour)-1):
            delta_pheromone[best_tour[i]-1][best_tour[i + 1]-1] += 1 / best_tour_length

        update_pheromone(pheromone, delta_pheromone, rho)

    return best_tour, best_tour_length


def dynamic_tsp_aco(city_info, traffic_factors, initial_aco_params, sub_tour_aco_params):
    cities = list(city_info.keys())
    base_distances = np.array([[distance(city_info[c1], city_info[c2]) for c2 in cities] for c1 in cities])
    num_nodes = len(cities)

    # Run the ACO algorithm and obtain the best initial tour and its length
    initial_tour, initial_tour_length = tsp_aco(cities, base_distances, None, None, **initial_aco_params)

    print(f"\nBest initial tour: {initial_tour}")
    print(f"\nBest initial tour length: {initial_tour_length}")

    tour_start = initial_tour[0]
    print(f"\n{'-'*40}\n")
    print(f"Starting tour at city {tour_start}\n")

    current_tour = initial_tour.copy()
    tour_current_city = current_tour[0]
    tour_unvisited = list(set(current_tour))
    tour_unvisited.remove(tour_current_city)
    tour_visited_cities = []
    tour_visited_cities.append(tour_current_city)

    initial_tour_length_tracker = 0.0
    current_tour_length_tracker = 0.0

    for i in range(1, len(initial_tour)-1):

        # calculate the remaining tour length based on base_distances
        base_remaining_tour_length = sum([base_distances[current_tour[c]-1][current_tour[c + 1]-1] for c in range(i-1, num_nodes)])

        # apply traffic factors
        print("Applying traffic factors")
        fluctuated_distances = apply_traffic_fluctuation(base_distances, num_nodes, traffic_factors)

        # calculate the remaining tour length based on fluctuated_distances
        fluctuated_remaining_tour_length = sum([fluctuated_distances[current_tour[c]-1][current_tour[c + 1]-1] for c in range(i-1, num_nodes)])

        # calculate the percentage change bewteen base_remaining_tour_length and fluctuated_remaining_tour_length
        percentage_change = ((fluctuated_remaining_tour_length - base_remaining_tour_length) / base_remaining_tour_length) * 100

        if percentage_change > 10:
            print(f"Difference in remaining tour length has increased by more than 10%.\nRecalculating new sub tour from current node.")

            sub_tour, sub_tour_length = tsp_aco(tour_unvisited, fluctuated_distances, tour_start, tour_current_city, **sub_tour_aco_params)

            if sub_tour_length < fluctuated_remaining_tour_length:
                print(f"Found better optimal sub tour.")
                current_tour = current_tour[:i-1] + sub_tour
                print(f"New current tour: {current_tour}")
            else:
                print(f"Could not find better optimal sub tour. Continuing with previous tour.")

        tour_next_city = current_tour[i]
        print(f"\n{'-'*40}\n")
        print(f"Visiting next city : city {tour_next_city}\n")
        tour_visited_cities.append(tour_next_city)
        tour_unvisited.remove(tour_next_city)

        current_tour_length_tracker += fluctuated_distances[tour_current_city - 1][tour_next_city - 1]
        initial_tour_length_tracker += fluctuated_distances[initial_tour[i-1] - 1][initial_tour[i] - 1]

        tour_current_city = tour_next_city

    print(f"\n{'-'*40}\n")
    print(f"Visiting final city : city {tour_next_city}\n")
    

    print(f"Final tour: {current_tour}")
    print(f"\nFinal tour (fluctuated distances) length: {current_tour_length_tracker}")
    print(f"\nInital tour (fluctuated distance) length: {initial_tour_length_tracker}")


def main():
    city_info = readFile()
    traffic_factors = [1.0, 1.05, 1.2, 1.5]

    # Define the parameters for the initial run
    initial_aco_params = {
        'num_ants': 50,
        'max_iterations': 150,
        'alpha': 1.5,
        'beta': 5,
        'rho': 0.75,
        'initial_pheromone': 1.0
    }

    # Define the parameters for the sub-tours
    sub_tour_aco_params = {
        'num_ants': 30,
        'max_iterations': 100,
        'alpha': 1.0,
        'beta': 5,
        'rho': 0.75,
        'initial_pheromone': 1.0
    }

    dynamic_tsp_aco(city_info, traffic_factors, initial_aco_params, sub_tour_aco_params)


if __name__ == "__main__":
    main()
