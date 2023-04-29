from aco_dynamic_tsp.helper_functions import readFile
from aco_dynamic_tsp.dynamic_tsp import dynamic_tsp_aco

def main():

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

    traffic_factors = [0, 0.1, 0.25, 0.5]
    threshold = 10
    seed = 42

    filenames = ["test.tsp","berlin20.tsp","berlin52.tsp","eil51.tsp","eil76.tsp","pr76.tsp","kroA100.tsp","kroC100.tsp","a280.tsp"]

    for filename in filenames:
        city_info = readFile(filename)
        dynamic_tsp_aco(city_info, traffic_factors, threshold, initial_aco_params, sub_tour_aco_params, seed)
        print(f"\n\n{'-'*100}\n\n")


if __name__ == "__main__":
    main()