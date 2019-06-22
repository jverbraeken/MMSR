import pickle
from multiprocessing import Process, Manager
from os import path
from typing import List, Tuple

import math
import pandas as pd

from system1 import get_discounted_recipes
from system2 import get_candidate_similar_recipes


def jaccard_similarity(liked_recipe, candidate_recipe) -> float:
    A = 0
    B = 0
    AB = 0
    for (e1, e2) in zip(liked_recipe, candidate_recipe):
        if e1:
            A += 1
        if e2:
            B += 1
        if e1 and e2:
            AB += 1
    return float(AB) / (A + B - AB)


def filter_recipes(recipes, newdata2, i, L):
    L[i] = list(map(
        lambda liked_recipe_candidate: (liked_recipe_candidate[0], liked_recipe_candidate[1], jaccard_similarity(newdata2[:, liked_recipe_candidate[0]], newdata2[:, liked_recipe_candidate[1]]))
        , recipes))
    print("Finished thread: " + str(i))


def get_liked_recipes() -> List[int]:
    with open("liked_recipes.json") as file:
        liked_recipes = list(range(1000))  # set(json.load(file))
    return liked_recipes


def get_recipe_matrix() -> List[List[int]]:
    data = pd.read_csv("epi_r.csv")
    data = data.drop(["title",
                      "rating",
                      "calories",
                      "protein",
                      "fat",
                      "sodium",
                      "#cakeweek",
                      "#wasteless",
                      "22-minute meals",
                      "3-ingredient recipes",
                      "30 days of groceries",
                      "advance prep required",
                      "alabama",
                      "alaska",
                      "alcoholic"], axis=1)
    recipe_matrix = data.transpose().as_matrix()
    return recipe_matrix


def get_similarity_between_candidates_pairs(candidate_similar_recipes: List[Tuple], recipe_matrix: List[List[int]]) -> List[Tuple[int, int, float]]:
    if len(candidate_similar_recipes) > 10000:
        with Manager() as manager:
            L = manager.dict()
            processes = []
            print("Starting processes")
            NUM_PROCESSES = 8
            for i in range(NUM_PROCESSES):
                print("Starting process: " + str(i))
                p = Process(target=filter_recipes, args=(candidate_similar_recipes[
                                                         math.floor((len(
                                                             candidate_similar_recipes) / NUM_PROCESSES) * i):
                                                         math.floor(
                                                             (len(candidate_similar_recipes) / NUM_PROCESSES) * (
                                                                     i + 1))],
                                                         recipe_matrix, i, L))
                processes.append(p)
                p.start()

            for proc in processes:
                proc.join()
            print("Finished processes")
            values = [item for sublist in L.values() for item in sublist]
            # with open("tmp", 'w') as file:
            #     file.write(str([value[2] for value in values]))
    else:
        values = list(map(lambda liked_recipe_candidate: (liked_recipe_candidate[0], liked_recipe_candidate[1],
                                                          jaccard_similarity(
                                                              recipe_matrix[:, liked_recipe_candidate[0]],
                                                              recipe_matrix[:, liked_recipe_candidate[1]])),
                          candidate_similar_recipes))
    return values


def get_most_similar_recipes_to_liked_recipes(recipe_matrix, liked_recipes, B, R, num_buckets_per_band) -> List[Tuple]:
    similarities_file = "similarities_" + str(B) + "_" + str(R) + "_" + str(num_buckets_per_band) + "_range1000"
    if path.isfile(similarities_file):
        with open(similarities_file, 'rb') as file:
            similarities = pickle.load(file)
    else:
        candidate_similar_recipes = get_candidate_similar_recipes(recipe_matrix, liked_recipes, B, R,
                                                                  num_buckets_per_band)
        similarities = get_similarity_between_candidates_pairs(list(candidate_similar_recipes), recipe_matrix)
        with open(similarities_file, 'wb') as file:
            pickle.dump(similarities, file)
    sorted_similarities = sorted(similarities, key=lambda tuple: tuple[2], reverse=True)
    with open(similarities_file + "_pretty", 'w') as file:
        file.write(str(sorted_similarities))
    return sorted_similarities


if __name__ == '__main__':
    liked_recipes = get_liked_recipes()
    recipe_matrix = get_recipe_matrix()
    B = 4
    R = 6
    num_buckets_per_band = 25000000

    discounted_recipes = get_discounted_recipes()
    most_similar_recipes_to_liked_recipes = get_most_similar_recipes_to_liked_recipes(recipe_matrix, liked_recipes, B, R, num_buckets_per_band)
