from functools import partial
from math import cos, sin
from multiprocessing import Pool
from os import cpu_count
from random import uniform
from time import time


def objective(x):
    return sum([sin(i) ** 3 + cos(i) ** 3 for i in x])


def check_constraints(x):
    return all((i >= 9 and i <= 99) for i in x)


def get_initial_solution(num_variables):
    while True:
        x = [uniform(9, 99) for _ in range(num_variables)]
        if check_constraints(x):
            return x


def neighbors(current_solution, neighborhood_size=10):
    neighborhood = []
    while len(neighborhood) < neighborhood_size:
        neighbor = [i + uniform(-0.01, 0.01) for i in current_solution]
        if check_constraints(neighbor):
            neighborhood.append(neighbor)
    return neighborhood


def find_min_condition(cost, best_cost):
    return cost < best_cost


def find_max_condition(cost, best_cost):
    return cost > best_cost


def list_in_list(target_array, list_of_arrays):
    for arr in list_of_arrays:
        if target_array == arr:
            return True
    return False


def tabu_search(
    initial_solution,
    find_min=False,
    max_iter=10000,
    tabu_size=10000,
    neighborhood_size=10,
):
    start_time = time()
    best_solution = initial_solution.copy()
    current_solution = initial_solution.copy()
    best_cost = objective(best_solution)
    tabu_list = [initial_solution.copy()]

    best_iter = 0
    total_iter = 0

    if find_min:
        condition = find_min_condition
        find_best_neighborhood = min
    else:
        condition = find_max_condition
        find_best_neighborhood = max

    while total_iter - best_iter <= max_iter:
        total_iter += 1
        neighborhood = neighbors(current_solution, neighborhood_size)
        current_solution = find_best_neighborhood(
            neighborhood, key=objective, default=current_solution.copy()
        )
        cost = objective(current_solution)
        if list_in_list(current_solution, tabu_list):
            continue

        if condition(cost, best_cost):
            best_solution = current_solution.copy()
            best_cost = cost
            best_iter = total_iter

        tabu_list.append(current_solution.copy())
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return initial_solution, best_solution, best_cost, time() - start_time


def parallel_tabu_search(
    num_variables: int,
    find_min: bool,
    max_iter: int,
    tabu_size: int,
    neighborhood_size: int,
):
    total_cpu = cpu_count()
    pool = Pool(total_cpu)
    initial_solutions = [get_initial_solution(num_variables) for _ in range(total_cpu)]
    function_to_run = partial(
        tabu_search,
        find_min=find_min,
        max_iter=max_iter,
        tabu_size=tabu_size,
        neighborhood_size=neighborhood_size,
    )
    results = pool.map(function_to_run, initial_solutions)
    pool.close()
    pool.join()
    if find_min:
        initial_solution, best_solution, best_cost, _ = min(results, key=lambda x: x[2])
    else:
        initial_solution, best_solution, best_cost, _ = max(results, key=lambda x: x[2])
    return (
        initial_solution,
        best_solution,
        sum([result[2] for result in results]) / len(results),
        best_cost,
        sum([result[3] for result in results]) / len(results),
    )


def start(
    num_variables: int,
    find_min: bool,
    max_iter: int,
    tabu_size: int,
    neighborhood_size: int,
):
    initial_solution, best_solution, mean_cost, best_cost, elapsed_time = (
        parallel_tabu_search(
            num_variables=num_variables,
            find_min=find_min,
            max_iter=max_iter,
            tabu_size=tabu_size,
            neighborhood_size=neighborhood_size,
        )
    )
    print("Initial Solution:", initial_solution)
    print("Initial Cost:", objective(initial_solution))
    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)
    print("Mean Cost:", mean_cost)
    print("Elapsed time:", elapsed_time, "seconds")


if __name__ == "__main__":
    start(
        num_variables=20,
        find_min=False,
        max_iter=10000,
        tabu_size=10000,
        neighborhood_size=10,
    )
