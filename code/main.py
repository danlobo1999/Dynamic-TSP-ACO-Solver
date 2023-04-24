from aco_dynamic_tsp.helper_functions import readFile
from aco_dynamic_tsp.dynamic_tsp_v1 import dynamic_tsp_aco

def main():
    # filename = "test.tsp"
    filename = "berlin20.tsp"
    city_info = readFile(filename)
    traffic_factors = [0, 0.1, 0.25, 0.5]
    threshold = 10
    seed = 42

    # Define the parameters for the initial run
    initial_aco_params = {
        'num_ants': 60,
        'max_iterations': 200,
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

    dynamic_tsp_aco(city_info, traffic_factors, threshold, initial_aco_params, sub_tour_aco_params, seed)


if __name__ == "__main__":
    main()