import json
from typing import Tuple, List

from whoosh import index
from whoosh.qparser import QueryParser


def get_jaccard_similarity(list1: List, list2: List) -> float:
    count = 0
    for elem1 in list1:
        if any(elem2 in elem1 for elem2 in list2):
            count += 1
    return count / (len(list1) + len(list2) - count)


def get_discounted_recipes() -> List[Tuple[str, float]]:
    with open("offers.json") as file:
        offers = json.load(file)
    with open("full_format_recipes.json") as file:
        recipes = json.load(file)

    map_recipe_to_ingredients = {}
    for i, recipe in enumerate(recipes):
        if recipe == {}:
            continue
        map_recipe_to_ingredients[i] = recipe["ingredients"]

    ix = index.open_dir("indexdir")

    map_recipe_to_jaccard = {}

    with ix.searcher() as searcher:
        query = QueryParser("ingredients", ix.schema).parse("(" + ") OR (".join(offers) + ")")
        results = searcher.search(query, limit=None)

        for result in results:
            recipe = result["title"]
            ingredients = map_recipe_to_ingredients[int(recipe)]
            map_recipe_to_jaccard[recipe] = get_jaccard_similarity(ingredients, offers)
    return map_recipe_to_jaccard
    #result = sorted(map_recipe_to_jaccard, key=map_recipe_to_jaccard.get, reverse=True)
    #return [(recipe, map_recipe_to_jaccard[recipe]) for recipe in result]


if __name__ == '__main__':
    get_discounted_recipes()
    print("Finished")
