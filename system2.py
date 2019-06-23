import pickle
from collections import defaultdict
from os import path
from typing import Set, Tuple

import numpy as np
import pandas as pd


def _first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def get_candidate_similar_recipes(recipe_matrix, liked_recipes, B, R) -> Set[Tuple]:
    signatures_file = "generated/signatures_" + str(B) + "_" + str(R) + ".npy"
    if path.isfile(signatures_file):
        signatures = np.load(signatures_file)
    else:
        signatures = np.empty([B * R, recipe_matrix.shape[1]], dtype=int)
        # liked_recipe_signatures = np.empty([B * R, len(liked_recipes)], dtype=int)
        for i in range(B * R):
            print("Generating signatures for row: " + str(i))
            # for recipe in range(newdata2.shape[1]):
            np.random.seed(i)
            np.random.shuffle(recipe_matrix)
            # shuffled_liked_recipes = [np.array([np.copy(newdata2[:, recipe])]).T for recipe in liked_recipes]
            # shuffled_recipes = [np.array([np.copy(newdata2[:, recipe])]).T for recipe in range(newdata2.shape[1])]
            # for recipe in shuffled_recipes:
            #     np.random.seed(i)
            #     np.random.shuffle(recipe)
            # for recipe in shuffled_liked_recipes:
            #     np.random.seed(i)
            #     np.random.shuffle(recipe)
            signatures[i] = [_first_nonzero(recipe_matrix[:, recipe], 0).item(0) for recipe in
                             range(recipe_matrix.shape[1])]
            # liked_recipe_signatures[i] = \
            #     [first_nonzero(newdata2[:, recipe], 0).item(0) for recipe in liked_recipes]
        np.save(signatures_file, signatures)
    print("Signatures generated")

    recipe_per_bucket_per_band = [defaultdict(list) for _ in range(B)]
    for band in range(B):
        for i, column in enumerate(signatures.T):
            bucket = hash(tuple(column[band * R: band * R + R]))
            recipe_per_bucket_per_band[band][bucket].append(i)
    print("Buckets per band / liked recipe buckets per band generated")

    liked_recipe_signatures = np.empty([B * R, len(liked_recipes)], dtype=int)
    for i, liked_recipe in enumerate(liked_recipes):
        liked_recipe_signatures[:, i] = signatures[:, liked_recipe]
    liked_recipe_buckets_per_band = [[] for _ in range(B)]
    for band in range(B):
        for i, signature in enumerate(liked_recipe_signatures.T):
            liked_recipe_buckets_per_band[band].append((hash(tuple(signature[band * R: band * R + R])), liked_recipes[i]))

    candidate_pairs = set()
    for band, liked_recipe_buckets in enumerate(liked_recipe_buckets_per_band):
        for i, (bucket, liked_recipe_id) in enumerate(liked_recipe_buckets):
            for candidate in recipe_per_bucket_per_band[band][bucket]:
                if candidate not in liked_recipes and (liked_recipe_id, candidate) not in candidate_pairs:
                    candidate_pairs.add((liked_recipe_id, candidate))

    print("Candidate pairs generated")
    # with open("candidate_pairs.pickle", 'wb') as f:
    #     pickle.dump(candidate_pairs, f)
    return candidate_pairs


if __name__ == '__main__':
    with open("liked_recipes.json") as file:
        liked_recipes = list(range(1000))  # json.load(file)
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
    get_candidate_similar_recipes(recipe_matrix, liked_recipes, 1, 24)
