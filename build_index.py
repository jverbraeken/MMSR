import json
import os.path

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID

schema = Schema(title=ID(stored=True),
                ingredients=TEXT(analyzer=StemmingAnalyzer()))

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

ix = index.create_in("indexdir", schema)

with open("full_format_recipes.json") as file:
    recipes = json.load(file)

writer = ix.writer()
num_recipes = len(recipes)
i = 0
for i, recipe in enumerate(recipes):
    if i % 1000 == 0:
        print("Building the index: " + str(round(i * 100.0 / num_recipes)) + "%\n")
    if recipe == {}:
        continue
    if "title" not in recipe or "ingredients" not in recipe:
        b = 5
    writer.add_document(title=str(i), ingredients=" ".join(recipe["ingredients"]))
    i = i + 1
print("Committing changes...\n")
writer.commit()
print("Finished")
