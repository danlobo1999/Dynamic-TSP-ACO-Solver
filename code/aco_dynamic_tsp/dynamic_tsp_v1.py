import math
import numpy as np
from aco_dynamic_tsp.tsp_aco import tsp_aco

def distance(city1, city2):
    return round(math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2), 2)


def apply_traffic_fluctuation(base_distances, traffic_factors=[0, 0.1, 0.25, 0.5], seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    traffic_matrix = np.random.choice(traffic_factors, size=base_distances.shape)
    fluctuated_distance_matrix = base_distances * (1 + traffic_matrix)

    return np.round(fluctuated_distance_matrix, 2)


def dynamic_tsp_aco(city_info, traffic_factors, threshold, initial_aco_params, sub_tour_aco_params, seed):
    cities = list(city_info.keys())
    base_distances = np.array([[distance(city_info[c1], city_info[c2]) for c2 in cities] for c1 in cities])
    num_nodes = len(cities)

    # Run the ACO algorithm and obtain the initial tour and its length
    initial_tour, initial_tour_length = tsp_aco(cities, base_distances, None, None, **initial_aco_params)

    print(f"\nInitial tour: {initial_tour}")
    print(f"\nInitial tour length: {initial_tour_length}")

    tour_start = initial_tour[0]
    print(f"\n{'-'*40}\n")
    print(f"Starting tour at city {tour_start}\n")

    current_tour = initial_tour.copy()
    tour_current_city = current_tour[0]
    tour_unvisited = list(set(current_tour))
    tour_unvisited.remove(tour_current_city)

    initial_tour_length_tracker = 0.0
    current_tour_length_tracker = 0.0


    for i in range(1, len(initial_tour)):

        # calculate the remaining tour length based on base_distances
        base_remaining_tour_length = sum([base_distances[current_tour[c]-1][current_tour[c + 1]-1] for c in range(i-1, num_nodes)])

        # apply traffic factors
        print("Applying traffic factors")
        fluctuated_distances = apply_traffic_fluctuation(base_distances, traffic_factors, seed)

        # calculate the remaining tour length based on fluctuated_distances
        fluctuated_remaining_tour_length = sum([fluctuated_distances[current_tour[c]-1][current_tour[c + 1]-1] for c in range(i-1, num_nodes)])

        if i == num_nodes:
            print("Next city is the final city.")
            print(f"\n{'-'*60}\n")
            print(f"Visiting final city : city {tour_start}")
            current_tour_length_tracker += fluctuated_distances[tour_current_city - 1][tour_start - 1]
            initial_tour_length_tracker += fluctuated_distances[initial_tour[i-1] - 1][initial_tour[i] - 1]
            print(f"\n{'-'*60}\n")
            break

        # calculate the percentage change bewteen base_remaining_tour_length and fluctuated_remaining_tour_length
        percentage_change = ((fluctuated_remaining_tour_length - base_remaining_tour_length) / base_remaining_tour_length) * 100

        if percentage_change > threshold:
            print(f"Difference in remaining tour length has increased by more than {threshold}%. Recalculating new sub tour from current node.")

            sub_tour, sub_tour_length = tsp_aco(tour_unvisited, fluctuated_distances, tour_start, tour_current_city, **sub_tour_aco_params)

            if sub_tour_length < fluctuated_remaining_tour_length:
                current_tour = current_tour[:i-1] + sub_tour
                print(f"Found a better sub tour. New current tour: {current_tour}")
            else:
                print(f"Could not find a better sub tour. Continuing with previous tour.")
        else:
            print("Difference in remaining tour length has increased by less than 10%. Continuing with previous tour.")

        tour_next_city = current_tour[i]
        print(f"\n{'-'*60}\n")
        print(f"Visiting next city : city {tour_next_city}\n")
        tour_unvisited.remove(tour_next_city)

        current_tour_length_tracker += fluctuated_distances[tour_current_city - 1][tour_next_city - 1]
        initial_tour_length_tracker += fluctuated_distances[initial_tour[i-1] - 1][initial_tour[i] - 1]

        tour_current_city = tour_next_city
    
    print(f"Initial tour: {initial_tour}")
    print(f"Inital tour length (base distances): {initial_tour_length}")
    print(f"Inital tour length (fluctuated distances): {initial_tour_length_tracker}")
    print(f"\nOptimized tour: {current_tour}")
    print(f"Optimized tour length (fluctuated distances): {current_tour_length_tracker}")