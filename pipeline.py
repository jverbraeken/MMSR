import json

import pandas as pd

from jaccard import get_discounted_recipes, get_jaccard_similarity
from min_hashing import get_candidate_similar_recipes

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

# discounted_recipes = get_discounted_recipes()
# candidate_similar_recipes = get_candidate_similar_recipes()
candidate_similar_recipes = [(295, 3598), (295, 4993), (295, 19109), (295, 579), (295, 865), (295, 1233), (295, 1895), (295, 1973), (295, 1995), (295, 2260), (295, 2476), (295, 2497), (295, 2609), (295, 2904), (295, 2961), (295, 3350), (295, 4199), (295, 4226), (295, 4265), (295, 4598), (295, 4901), (295, 5170), (295, 5278), (295, 5313), (295, 6233), (295, 6299), (295, 6518), (295, 6657), (295, 6734), (295, 6822), (295, 6890), (295, 7296), (295, 7495), (295, 7612), (295, 8144), (295, 8452), (295, 9053), (295, 9692), (295, 10647), (295, 11257), (295, 11506), (295, 11637), (295, 11761), (295, 11809), (295, 11864), (295, 11943), (295, 12064), (295, 12191), (295, 12504), (295, 12832), (295, 13452), (295, 13537), (295, 13556), (295, 14368), (295, 15101), (295, 15363), (295, 15507), (295, 15517), (295, 15684), (295, 16267), (295, 16915), (295, 17191), (295, 17435), (295, 18015), (295, 18749), (295, 18790), (295, 18907), (295, 18916), (295, 18954), (295, 19441), (295, 19700), (295, 2886)]

def compare_recipes(liked_recipe, candidate_recipe):
    count_liked = 0
    count_liked_candidate = 0
    for (e1, e2) in zip(liked_recipe, candidate_recipe):
        if e1 == 1:
            count_liked += 1
            if e2 == 1:
                count_liked_candidate += 1
    if count_liked_candidate == 0:
        return 0
    return count_liked_candidate / count_liked


similar_recipes = list(filter(
    lambda liked_recipe_candidate: compare_recipes(newdata2[:, liked_recipe_candidate[0]], newdata2[:, liked_recipe_candidate[1]]) > 0.4,
    candidate_similar_recipes))

a = 5