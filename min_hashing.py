import json

import numpy as np
import pandas as pd


with open("liked_recipes.json") as file:
    liked_recipes = json.load(file)

data = pd.read_csv("epi_r.csv")
# data = data.drop(["title",
#                   "rating",
#                   "calories",
#                   "protein",
#                   "fat",
#                   "sodium",
#                   "#cakeweek",
#                   "#wasteless",
#                   "22-minute meals",
#                   "3-ingredient recipes",
#                   "30 days of groceries",
#                   "advance prep required",
#                   "alabama",
#                   "alaska",
#                   "alcoholic"], axis=1)
newdata = data.transpose()
# newdata.to_csv("tmp.csv")
newdata2 = newdata.as_matrix()
# liked_recipe_ingredients = [newdata2[recipe] for recipe in liked_recipes]


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


signatures = np.empty([24, newdata2.shape[1]], dtype=int)
# liked_recipe_signatures = np.empty([24, len(liked_recipes)], dtype=int)
for i in range(24):
    np.random.seed(i)
    shuffled_data = np.copy(newdata2)
    # shuffled_liked_recipes = [np.copy(recipe) for recipe in liked_recipe_ingredients]
    np.random.shuffle(shuffled_data)
    # for recipe in shuffled_liked_recipes:
    #     np.random.shuffle(recipe)
    signatures[i] = first_nonzero(shuffled_data, 0)
    # liked_recipe_signatures[i] = \
    #     [first_nonzero(shuffled_liked_recipes[recipe], 0).item(0) for recipe in range(len(shuffled_liked_recipes))]

buckets_per_band = [[[] for _ in range(10000)] for band in range(8)]
for band in range(8):
    for i, column in enumerate(signatures.T):
        buckets_per_band[band][hash(tuple(column[band * 3 : band * 3 + 3])) % 10000].append(i)

a = 5
