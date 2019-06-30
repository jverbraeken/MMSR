from collections import defaultdict
from os import path
from typing import Set, Tuple, List, Dict

import numpy as np
import pandas as pd


def _first_nonzero(arr) -> int:
    """
    Returns the index of the first row of arr which is non-zero
    """
    mask = arr != 0
    return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)


def get_signatures(recipe_matrix: pd.DataFrame, B: int, R: int) -> np.ndarray:
    """
    Returns a list of signatures for all recipes given the locality-sensitive hashing parameters B and R
    """
    signatures_file = "generated/signatures_" + str(B) + "_" + str(R) + ".npy"
    if path.isfile(signatures_file):
        signatures = np.load(signatures_file)
    else:
        signatures = np.empty([B * R, recipe_matrix.shape[1]], dtype=int)
        for i in range(B * R):
            print("Generating signatures for row: " + str(i))
            np.random.seed(i)
            np.random.shuffle(recipe_matrix)
            signatures[i] = [_first_nonzero(recipe_matrix[:, recipe]).item(0) for recipe in
                             range(recipe_matrix.shape[1])]
        np.save(signatures_file, signatures)
    print("Signatures generated")
    return signatures


def get_recipe_per_bucket_per_band(signatures: np.ndarray, B: int, R: int) -> List[Dict[int, List]]:
    """
    Returns a list with for every band a dictionary with for every bucket a list of recipes that are hashed into that bucket
    """
    recipe_per_bucket_per_band = [defaultdict(list) for _ in range(B)]
    for band in range(B):
        for i, column in enumerate(signatures.T):
            bucket = hash(tuple(column[band * R: band * R + R]))
            recipe_per_bucket_per_band[band][bucket].append(i)
    print("Buckets per band / liked recipe buckets per band generated")
    return recipe_per_bucket_per_band


def get_liked_recipe_buckets_per_band(signatures: np.ndarray, liked_recipes: List[int], B: int, R: int) -> List[List[int]]:
    """
    Returns the bands in which the liked recipes are hashed
    """
    liked_recipe_signatures = np.empty([B * R, len(liked_recipes)], dtype=int)
    for i, liked_recipe in enumerate(liked_recipes):
        liked_recipe_signatures[:, i] = signatures[:, liked_recipe]
    liked_recipe_buckets_per_band = [[] for _ in range(B)]
    for band in range(B):
        for i, signature in enumerate(liked_recipe_signatures.T):
            liked_recipe_buckets_per_band[band].append((hash(tuple(signature[band * R: band * R + R])), liked_recipes[i]))
    return liked_recipe_buckets_per_band


def get_candidate_pairs(recipe_per_bucket_per_band: List[Dict[int, List]], liked_recipe_buckets_per_band: List[List[int]], liked_recipes: List[int]) -> Set[Tuple[int, int]]:
    """
    Returns a set of candidate pairs given the output of the locality-sensitive hashing algorithm and the bands in which the liked recipes are located
    """
    candidate_pairs = set()
    for band, liked_recipe_buckets in enumerate(liked_recipe_buckets_per_band):
        for i, (bucket, liked_recipe_id) in enumerate(liked_recipe_buckets):
            for candidate in recipe_per_bucket_per_band[band][bucket]:
                if candidate not in liked_recipes and (liked_recipe_id, candidate) not in candidate_pairs:
                    candidate_pairs.add((liked_recipe_id, candidate))
    return candidate_pairs


def get_candidate_similar_recipes(recipe_matrix: pd.DataFrame, liked_recipes: List[int], B: int, R: int) -> Set[Tuple]:
    """
    Returns a set of candidate pairs of approximately similar recipes using min-hashing and locality-sensitive hashing
    """
    signatures = get_signatures(recipe_matrix, B, R)
    recipe_per_bucket_per_band = get_recipe_per_bucket_per_band(signatures, B, R)
    liked_recipe_buckets_per_band = get_liked_recipe_buckets_per_band(signatures, liked_recipes, B, R)
    candidate_pairs = get_candidate_pairs(recipe_per_bucket_per_band, liked_recipe_buckets_per_band, liked_recipes)

    print("Candidate pairs generated")
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
