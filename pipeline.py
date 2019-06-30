import codecs
import pickle
import time
from collections import defaultdict
from multiprocessing import Process, Manager
from os import path
from typing import List, Tuple, Dict

import math
import numpy as np
import pandas as pd

from system1 import get_discounted_recipes
from system2 import get_candidate_similar_recipes
from system3 import get_recommendations


def jaccard_similarity(liked_recipe: List[int], candidate_recipe: List[int]) -> float:
    """
    Calculate the Jaccard similarity between the ingredients in the liked_recipe and the candidate recipe
    """
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


def map_candidate_to_jaccard(recipes: List[Tuple[int, int]], recipe_matrix: pd.DataFrame, i: int, L: Dict) -> None:
    """
    Calculate Jaccard similarity between all candidate pairs
    """
    L[i] = list(map(
        lambda liked_recipe_candidate: (liked_recipe_candidate[0], liked_recipe_candidate[1], jaccard_similarity(recipe_matrix[:, liked_recipe_candidate[0]], recipe_matrix[:, liked_recipe_candidate[1]]))
        , recipes))
    print("Finished thread: " + str(i))


def get_recipe_matrix() -> pd.DataFrame:
    """
    Returns a 0/1 matrix that maps recipes to ingredients
    """
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


def get_similarity_between_candidates_pairs(candidate_similar_recipes: List[Tuple], recipe_matrix: pd.DataFrame) -> List[Tuple[int, int, float]]:
    """
    Returns a list of tuples[candidate_pair[0], candidate_pair[1], jaccard_similarity]
    """
    THRES_USE_THREADING = 500000
    if len(candidate_similar_recipes) > THRES_USE_THREADING:
        with Manager() as manager:
            L = manager.dict()
            processes = []
            print("Starting processes")
            NUM_PROCESSES = 2
            for i in range(NUM_PROCESSES):
                print("Starting process: " + str(i))
                p = Process(target=map_candidate_to_jaccard, args=(candidate_similar_recipes[
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


def get_most_similar_recipes_to_liked_recipes(recipe_matrix: pd.DataFrame, liked_recipes: List[int], B: int, R: int) -> List[Tuple]:
    """
    Returns a list of tuples[candidate_pair[0], candidate_pair[1], jaccard_similarity], sorted descendingly on similarity
    """
    similarities_file = "generated/similarities_" + str(B) + "_" + str(R) + "_range1000"
    if path.isfile(similarities_file):
        with open(similarities_file, 'rb') as file:
            similarities = pickle.load(file)
    else:
        start = time.time()
        candidate_similar_recipes = get_candidate_similar_recipes(recipe_matrix, liked_recipes, B, R)
        print("Time to get candidate similar recipes: " + str(time.time() - start))
        start = time.time()
        similarities = get_similarity_between_candidates_pairs(list(candidate_similar_recipes), recipe_matrix)
        print("Time to calculate similarity between all candidate pairs: " + str(time.time() - start))
        with open(similarities_file, 'wb') as file:
            pickle.dump(similarities, file)
    sorted_similarities = sorted(similarities, key=lambda tuple: tuple[2], reverse=True)
    # with open(similarities_file + "_pretty", 'w') as file:
    #     file.write(str(sorted_similarities))
    with open(similarities_file + "_simplified", 'w') as file:
        file.write(str([value[2] for value in sorted_similarities]))
        file.write("\n\n" + str(np.mean([value[2] for value in sorted_similarities])))
        file.write("\n\n" + str(len(sorted_similarities)))
        file.write("\n\n" + str(len(list(filter(lambda x: x >= 0.75, [value[2] for value in sorted_similarities])))))
    return sorted_similarities


if __name__ == '__main__':
    liked_recipes = list(range(10))
    recipe_matrix = get_recipe_matrix()

    B = 6
    R = 4

    num_recommendations = -1
    num_users = 50
    user_id = num_users + 1
    num_test_recipes_per_user = 0

    discounted_recipes = get_discounted_recipes()
    most_similar_recipes_to_liked_recipes = get_most_similar_recipes_to_liked_recipes(recipe_matrix, liked_recipes, B, R)
    recommended_recipes = get_recommendations(num_recommendations, num_users, user_id, num_test_recipes_per_user)

    map_discounted_recipe_to_rating = defaultdict(lambda: 0)
    for recipe in discounted_recipes:
        map_discounted_recipe_to_rating[int(recipe[0])] = recipe[1]

    map_most_similar_to_liked_recipe_to_rating = defaultdict(lambda: 0)
    for recipe in most_similar_recipes_to_liked_recipes:
        map_most_similar_to_liked_recipe_to_rating[recipe[1]] = recipe[2]

    map_recommended_recipe_to_rating = defaultdict(lambda: 0)
    for recipe in recommended_recipes:
        map_recommended_recipe_to_rating[recipe[0]] = (recipe[1][0][0] - 1) / 4

    scores = []
    for i in range(recipe_matrix.shape[1]):
        discounted_score = map_discounted_recipe_to_rating[i]  # system 1
        content_based_score = map_most_similar_to_liked_recipe_to_rating[i]  # system 2
        collaborative_score = map_recommended_recipe_to_rating[i]  # system 3
        ratio = min(1, num_users / 10000)
        scores.append((i, discounted_score * ((1 - ratio) * content_based_score + ratio * collaborative_score)))
    scores.sort(key=lambda k: k[1], reverse=True)

    titles = pd.read_csv("epi_r.csv", usecols=["title"])

    title_to_rating = [(titles.loc[score[0], "title"], score[1]) for score in scores]

    print(title_to_rating[:10])

    with codecs.open("results.txt", 'w', "utf-8") as file:
        file.write(str(title_to_rating).replace("), (", "),\n("))
