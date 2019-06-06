import json

offers = [
    "carrot",
    "selery",
    "lettuce",
    "turkey breast"
]

with open("offers.json", 'w') as file:
    json.dump(offers, file)
