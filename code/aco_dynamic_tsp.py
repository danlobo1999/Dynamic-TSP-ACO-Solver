import copy
import numpy as np
import random

class ACO:
    def __init__(self, n_ants, max_iter, rho, alpha, beta, dist_matrix, traffic_data):
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.dist_matrix = dist_matrix
        self.traffic_data = traffic_data
        self.n_cities = len(dist_matrix)
        self.pheromone_matrix = np.full((self.n_cities, self.n_cities), 1e-6)

    def calculate_heuristic(self, city, unvisited, time):
        eta = []
        for next_city in unvisited:
            traffic_factor = self.traffic_data[city][next_city](time)
            x = self.dist_matrix[city][next_city]
            eta.append(1 / (self.dist_matrix[city][next_city] * traffic_factor))
        return eta

    def select_next_city(self, city, unvisited, time):
        # eta = self.calculate_heuristic(city, unvisited, time)
        # tau = self.pheromone_matrix[city][unvisited]
        eta = np.array(self.calculate_heuristic(city, unvisited, time))
        tau = np.array([self.pheromone_matrix[city][next_city] for next_city in unvisited])
        probabilities = (tau**self.alpha) * (eta**self.beta)
        probabilities /= probabilities.sum()
        return np.random.choice(unvisited, p=probabilities)

    def update_pheromone(self, solutions):
        self.pheromone_matrix *= (1 - self.rho)
        for tour, cost in solutions:
            delta_tau = 1 / cost
            for i in range(len(tour) - 1):
                self.pheromone_matrix[tour[i]][tour[i + 1]] += delta_tau

    def generate_solution(self, start, visited, time):
        unvisited = [city for city in range(self.n_cities) if city not in visited and city != start]
        # tour = visited + [start]
        tour = copy.deepcopy(visited)
        cost = 0

        while len(unvisited) > 0:
            next_city = self.select_next_city(start, unvisited, time)
            cost += self.dist_matrix[start][next_city] * self.traffic_data[start][next_city](time)
            time += 1
            tour.append(next_city)
            unvisited.remove(next_city)
            start = next_city

        tour.append(tour[0])
        cost += self.dist_matrix[start][tour[0]] * self.traffic_data[start][tour[0]](time)

        return tour, cost

    def run_aco(self, start=None, visited=[]):
        best_tour = None
        best_cost = float('inf')

        for _ in range(self.max_iter):
            solutions = []
            for _ in range(self.n_ants):
                if start is None:
                    initial = random.choice([city for city in range(self.n_cities) if city not in visited])
                    tour, cost = self.generate_solution(initial, visited, len(visited))
                else:
                    tour, cost = self.generate_solution(start, visited, len(visited))
                solutions.append((tour, cost))

                if cost < best_cost:
                    best_tour = tour
                    best_cost = cost

            self.update_pheromone(solutions)

        return best_tour, best_cost

def dynamic_tsp_aco(dist_matrix, traffic_data, n_ants=10, max_iter=100, rho=0.1, alpha=1, beta=3):
    aco = ACO(n_ants, max_iter, rho, alpha, beta, dist_matrix, traffic_data)
    best_tour, _ = aco.run_aco()
    print(f"Initial tour: {best_tour}")

    # Simulating the salesman's progress through the path
    visited = []
    current_city = best_tour[0]
    total_cost = 0

    for i in range(1, len(best_tour) - 1):
        visited.append(current_city)
        next_city = best_tour[i]
        current_time = len(visited)

        # Update path based on current city, visited cities, and traffic factors
        sub_tour, sub_cost = aco.run_aco(start=current_city, visited=visited)
        updated_tour = visited + sub_tour
        total_cost += aco.dist_matrix[current_city][next_city] * aco.traffic_data[current_city][next_city](current_time)

        best_tour = updated_tour
        current_city = next_city

    total_cost += aco.dist_matrix[current_city][best_tour[0]] * aco.traffic_data[current_city][best_tour[0]](len(visited) + 1)
    best_tour.append(best_tour[0])

    return best_tour, total_cost

# Example usage:

# Define your distance matrix (static) and traffic data (dynamic) here.
# dist_matrix: A 2D matrix representing the distances between cities
# traffic_data: A 2D matrix representing the traffic factors as functions of time

dist_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

traffic_data = [
    [lambda t: 1, lambda t: 1 + 0.1 * t, lambda t: 1 - 0.05 * t, lambda t: 1],
    [lambda t: 1 - 0.1 * t, lambda t: 1, lambda t: 1 + 0.15 * t, lambda t: 1],
    [lambda t: 1 + 0.05 * t, lambda t: 1 - 0.15 * t, lambda t: 1, lambda t: 1],
    [lambda t: 1, lambda t: 1, lambda t: 1, lambda t: 1]
]
best_tour, total_cost = dynamic_tsp_aco(dist_matrix, traffic_data)
print("Best tour:", best_tour)
print("Total cost:", total_cost)
