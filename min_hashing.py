import json

import numpy as np
import pandas as pd

with open("liked_recipes.json") as file:
    liked_recipes = json.load(file)

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
newdata = data.transpose()
# newdata.to_csv("tmp.csv")
newdata2 = newdata.as_matrix()

print("Loaded data")


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


B = 4
R = 6
signatures = np.empty([B * R, newdata2.shape[1]], dtype=int)
liked_recipe_signatures = np.empty([B * R, len(liked_recipes)], dtype=int)
for i in range(B * R):
    np.random.seed(i)
    shuffled_recipes = [np.array([np.copy(newdata2[:, recipe])]).T for recipe in range(newdata2.shape[1])]
    shuffled_liked_recipes = [np.array([np.copy(newdata2[:, recipe])]).T for recipe in liked_recipes]
    for recipe in shuffled_recipes:
        np.random.seed(i)
        np.random.shuffle(recipe)
    for recipe in shuffled_liked_recipes:
        np.random.seed(i)
        np.random.shuffle(recipe)
    signatures[i] = [first_nonzero(shuffled_recipes[recipe], 0).item(0) for recipe in range(len(shuffled_recipes))]
    liked_recipe_signatures[i] = \
        [first_nonzero(shuffled_liked_recipes[recipe], 0).item(0) for recipe in range(len(shuffled_liked_recipes))]

print("Signatures generated")

NUM_BUCKETS_PER_BAND = 100000
buckets_per_band = [[[] for _ in range(NUM_BUCKETS_PER_BAND)] for band in range(B)]
liked_recipe_buckets_per_band = [[] for band in range(B)]
for band in range(B):
    for i, column in enumerate(signatures.T):
        buckets_per_band[band][hash(tuple(column[band * R: band * R + R])) % NUM_BUCKETS_PER_BAND].append(i)
    for i, signature in enumerate(liked_recipe_signatures.T):
        liked_recipe_buckets_per_band[band].append(
            (hash(tuple(signature[band * R: band * R + R])) % NUM_BUCKETS_PER_BAND, i))

print("Buckets per band / liked recipe buckets per band generated")

candidate_pairs = []
for band, liked_recipe_buckets in enumerate(liked_recipe_buckets_per_band):
    for (bucket, liked_recipe_id) in liked_recipe_buckets:
        for candidate in buckets_per_band[band][bucket]:
            if (liked_recipe_id, candidate) not in candidate_pairs and (candidate, liked_recipe_id) not in candidate_pairs:
                candidate_pairs.append((liked_recipe_id, candidate))

print("Candidate pairs generated")
print(candidate_pairs)
