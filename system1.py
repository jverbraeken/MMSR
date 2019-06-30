import json
import time
from typing import Tuple, List

from whoosh import index
from whoosh.qparser import QueryParser


def get_jaccard_similarity(list1: List[str], list2: List[str]) -> float:
    """
    Returns the Jaccard similarity between the two lists, where list2 elements are checked for inclusion in list1 elements
    """
    count = 0
    for elem1 in list1:
        if any(elem2 in elem1 for elem2 in list2):
            count += 1
    return float(count) / (len(list1) + len(list2) - count)


def get_relative_similarity(list1: List[str], list2: List[str]) -> float:
    """
    Returns the fraction of elements in list2 that occur in list1, where list2 elements are checked for inclusion in list1 elements
    """
    count = 0
    for elem1 in list1:
        for elem2 in list2:
            if elem2 in elem1:
                count += 1
    return float(count) / len(list1)


def get_discounted_recipes() -> List[Tuple[str, float]]:
    """
    Returns a list of recipes of which at least one ingredient is discounted and also the fraction of ingredients that is discounted
    """
    with open("offers.json") as file:
        offers = json.load(file)
    with open("full_format_recipes.json") as file:
        recipes = json.load(file)
    similarity_metric = "relative"

    map_recipe_to_ingredients = {}
    for i, recipe in enumerate(recipes):
        if recipe == {}:
            continue
        map_recipe_to_ingredients[i] = recipe["ingredients"]

    ix = index.open_dir("indexdir")

    map_recipe_to_jaccard = {}

    if similarity_metric == "jaccard":
        calculate_similarity = get_jaccard_similarity
    elif similarity_metric == "relative":
        calculate_similarity = get_relative_similarity
    else:
        calculate_similarity = None
        print("Error: similarity_metric must be either \"jaccard\" or \"relative\"")
        exit(1)

    start_time = time.time()
    with ix.searcher() as searcher:
        query = QueryParser("ingredients", ix.schema).parse("(" + ") OR (".join(offers) + ")")
        results = searcher.search(query, limit=None)
        print("Time to search through index: " + str(time.time() - start_time))

        for result in results:
            recipe = result["title"]
            ingredients = map_recipe_to_ingredients[int(recipe)]
            map_recipe_to_jaccard[recipe] = calculate_similarity(ingredients, offers)
    result = sorted(map_recipe_to_jaccard, key=map_recipe_to_jaccard.get, reverse=True)
    result_as_tuple = [(recipe, map_recipe_to_jaccard[recipe]) for recipe in result]

    print("Total time to find discounted recipes: " + str(time.time() - start_time))
    return result_as_tuple


if __name__ == '__main__':
    result = get_discounted_recipes()
    print("Finished")
