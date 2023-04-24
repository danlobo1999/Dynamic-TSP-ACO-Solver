import random


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