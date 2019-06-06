import abc
from urllib.request import urlopen as url

from bs4 import BeautifulSoup as bs


# must be passed raw html
class Recipe:
    __metaclass__ = abc.ABCMeta
    title = ''
    date = ''
    desc = ''
    ingredients = []
    directions = []
    categories = []

    @abc.abstractstaticmethod
    def get_title(self, page):
        pass

    @abc.abstractstaticmethod
    def get_ingredients(self, page):
        pass

    @abc.abstractstaticmethod
    def get_directions(self, page):
        pass

    @abc.abstractstaticmethod
    def get_categories(self, page):
        pass

    @abc.abstractstaticmethod
    def get_date(self, page):
        pass

    @abc.abstractstaticmethod
    def get_desc(self, page):
        pass

    def build_recipie(self, page):
        self.title = self.get_title(page)
        self.ingredients = self.get_ingredients(page)
        self.directions = self.get_directions(page)
        self.categories = self.get_categories(page)
        self.date = self.get_date(page)
        self.desc = self.get_desc(page)

    def __init__(self, page):
        print('attempting to build from: ' + page)
        try:
            self.build_recipie(bs(url(page), 'html.parser'))
        except Exception as x:
            print('Could not build from %s, %s' % (page, x))


class FN_Recipe(Recipe):
    def get_title(self, page):
        return page.find('div', {'class': 'tier-3 title'}).text.encode('utf-8').strip()

    def get_ingredients(self, page):
        return [i.text.encode('utf-8').strip() for i in
                page.find_all('div', {'class': 'ingredients'})[0].find_all('li')]

    def get_directions(self, page):
        return [i.text.encode('utf-8').strip() for i in
                page.find_all('ul', {'class': 'recipe-directions-list'})[0].find_all('li')]

    def get_categories(self, page):
        return [i.text.encode('utf-8').strip() for i in page.find_all('ul', {'class': 'categories'})[0].find_all('li')]

    def get_date(self, page):
        try:
            page_s = str(page)
            return page_s[page_s.index('OrigPubDate') + 14:page_s.index('OrigPubDate') + 24]
        except:
            return ''

    def get_desc(self, page):
        return [dd['content'] for dd in page.find_all('meta', {'itemprop': 'description'})][-1].encode('utf-8').strip()


# No scripting shit apparently needed
class EP_Recipe(Recipe):
    rating = None
    calories = None
    sodium = None
    fat = None
    protein = None

    def get_date(self, page):
        try:
            return page.find('meta', {'itemprop': 'datePublished'})['content']
        except:
            return None

    def get_desc(self, page):
        try:
            return page.find('div', {'itemprop': 'description'}).find('p').text
        except:
            return None

    def get_directions(self, page):
        return [i.text.strip() for i in page.find_all('li', {'class': "preparation-step"})]

    def get_ingredients(self, page):
        return [i.text.strip() for i in page.find_all('li', {'itemprop': "ingredients"})]

    def get_categories(self, page):
        return [i.text for i in page.find_all('dt', {'itemprop': "recipeCategory"})]

    def get_title(self, page):
        return page.find('h1', {'itemprop': 'name'}).text

    def get_rating(self, page):
        try:
            return float(page.find_all('span', {'class': 'rating'})[-1].text.split('/')[0]) * 5 / 4
        except:
            return None

    def build_recipie(self, page):
        super(EP_Recipe, self).build_recipie(page)
        self.rating = self.get_rating(page)
        self.calories = self.get_calories(page)
        self.sodium = self.get_sodium(page)
        self.fat = self.get_fat(page)
        self.protein = self.get_protein(page)

    def get_calories(self, page):
        try:
            return float(page.find('span', {'class': 'nutri-data', 'itemprop': 'calories'}).text)
        except:
            return None

    def get_sodium(self, page):
        try:
            return float(page.find('span', {'class': 'nutri-data', 'itemprop': 'sodiumContent'}).text.split(' ')[0])
        except:
            return None

    def get_fat(self, page):
        try:
            return float(page.find('span', {'class': 'nutri-data', 'itemprop': 'fatContent'}).text.split(' ')[0])
        except:
            return None

    def get_protein(self, page):
        try:
            return float(page.find('span', {'class': 'nutri-data', 'itemprop': 'proteinContent'}).text.split(' ')[0])
        except:
            return None


if __name__ == '__main__':
    import pickle
    import json

    # ep_urls = pickle.load(open('epi_urls.checkpoint','rb'))
    # ep_urls = [str(i) for i in ep_urls]
    # p = multiprocessing.Pool(4)
    # output = p.map(EP_Recipe,ep_urls)
    # pickle.dump(output,open('epi_recipes.final','wb'))
    data = pickle.load(open('epi_recipes.final', 'rb'))
    ar = []
    for i in data:
        ar.append(i.__dict__)
    pickle.dump(ar, open('epi_recipe_dict_form.dict', 'wb'))

    with open('result.json', 'w') as fp:
        json.dump(ar, fp)

{"directions": [
    "1. Place the stock, lentils, celery, carrot, thyme, and salt in a medium saucepan and bring to a boil. Reduce heat to low and simmer until the lentils are tender, about 30 minutes, depending on the lentils. (If they begin to dry out, add water as needed.) Remove and discard the thyme. Drain and transfer the mixture to a bowl; let cool.",
    "2. Fold in the tomato, apple, lemon juice, and olive oil. Season with the pepper.",
    "3. To assemble a wrap, place 1 lavash sheet on a clean work surface. Spread some of the lentil mixture on the end nearest you, leaving a 1-inch border. Top with several slices of turkey, then some of the lettuce. Roll up the lavash, slice crosswise, and serve. If using tortillas, spread the lentils in the center, top with the turkey and lettuce, and fold up the bottom, left side, and right side before rolling away from you."],
 "fat": 7.0, "date": "2006-09-01T04:00:00.000Z",
 "categories": ["Sandwich", "Bean", "Fruit", "Tomato", "turkey", "Vegetable", "Kid-Friendly", "Apple", "Lentil",
                "Lettuce", "Cookie"], "calories": 426.0, "desc": null, "protein": 30.0, "rating": 2.5,
 "title": "Lentil, Apple, and Turkey Wrap ",
 "ingredients": ["4 cups low-sodium vegetable or chicken stock", "1 cup dried brown lentils",
                 "1/2 cup dried French green lentils", "2 stalks celery, chopped", "1 large carrot, peeled and chopped",
                 "1 sprig fresh thyme", "1 teaspoon kosher salt", "1 medium tomato, cored, seeded, and diced",
                 "1 small Fuji apple, cored and diced", "1 tablespoon freshly squeezed lemon juice",
                 "2 teaspoons extra-virgin olive oil", "Freshly ground black pepper to taste",
                 "3 sheets whole-wheat lavash, cut in half crosswise, or 6 (12-inch) flour tortillas",
                 "3/4 pound turkey breast, thinly sliced", "1/2 head Bibb lettuce"], "sodium": 559.0}
