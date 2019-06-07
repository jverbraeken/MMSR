import json

offers = [
    "carrot",
    "selery",
    "lettuce",
    "turkey breast"
]

with open("offers.json", 'w') as file:
    json.dump(offers, file)

likes = [
    1, 5, 8, 9, 10
]

with open("liked_recipes.json", 'w') as file:
    json.dump(likes, file)


