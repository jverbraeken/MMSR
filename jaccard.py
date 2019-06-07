import json

from whoosh import index
from whoosh.qparser import QueryParser

with open("offers.json") as file:
    offers = json.load(file)
with open("full_format_recipes.json") as file:
    recipes = json.load(file)

map_recipe_to_ingredients = {}
for recipe in recipes:
    if recipe == {}:
        continue
    map_recipe_to_ingredients[recipe["title"]] = recipe["ingredients"]

ix = index.open_dir("indexdir")

map_recipe_to_jaccard = {}

with ix.searcher() as searcher:
    query = QueryParser("ingredients", ix.schema).parse("(" + ") OR (".join(offers) + ")")
    results = searcher.search(query, limit=None)

    for result in results:
        recipe = result["title"]
        count = 0
        ingredients = map_recipe_to_ingredients[recipe]
        for ingredient in ingredients:
            if any(offer in ingredient for offer in offers):
                count += 1
        jaccard_similarity = count / (len(offers) + len(ingredients) - count)
        map_recipe_to_jaccard[recipe] = jaccard_similarity
result = sorted(map_recipe_to_jaccard, key=map_recipe_to_jaccard.get)