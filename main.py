import json

from whoosh import index
from whoosh.qparser import QueryParser

with open("offers.json") as file:
    offers = json.load(file)

ix = index.open_dir("indexdir")

with ix.searcher() as searcher:
    query = QueryParser("ingredients", ix.schema).parse("(" + ") OR (".join(offers) + ")")
    results = searcher.search(query, limit=None)

    for result in results:
        print(result["title"] + "\n")
