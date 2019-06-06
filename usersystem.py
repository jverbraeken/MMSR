import pandas as pd 
import numpy as np
import surprise

def create_mockdata(data):
    # check number of empty values for each column
    #print(recipes.isnull().sum())
    recipe_dictionary = dict(enumerate(recipes.title, start=0))
    new_dataframe = pd.DataFrame(columns=['userid', 'recipeid', 'rating'])
    new_dataframe['userid'] = np.random.randint(0, 1000, size=100000)
    new_dataframe = new_dataframe.sort_values(by=['userid'])
    new_dataframe['recipeid'] = np.random.randint(0, high=len(recipes.title), size=100000)
    new_dataframe['rating'] = np.random.randint(0, 2, size=100000)
    new_dataframe = new_dataframe.reset_index(drop=True)
    #new_dataframe['recipeid'] = np.random.randint(0, )

    #new_dataframe = pd.DataFrame(columns=[['userid', 'recipeid', 'rating']])
    #new_dataframe['userid'] = np.random.randint(1, 100000, new_dataframe.shape[0])
    return new_dataframe

def matrix_creation(data):
    return 0

def build_model(data, unique_recipes, userid):
    reader = surprise.Reader(rating_scale = (0, 1))
    data = surprise.Dataset.load_from_df(data, reader)
    svd_model = surprise.SVDpp()
    result = svd_model.fit(data.build_full_trainset())
    pred = svd_model.predict(userid='50', iid='52')
    score = pred.est 
    print(score)

    # Recommend 
    

recipes = pd.read_csv('epi_r.csv')
unique_recipes = recipes['title'].unique()


user_recipe = create_mockdata(recipes)
user_to_index = dict(zip(user_recipe['userid'], range(len(user_recipe['userid']))))
recipe_to_index = dict(zip(unique_recipes, range(len(unique_recipes))))
index_to_user = dict(zip(user_to_index.values(), user_to_index.keys()))      
index_to_recipe = dict(zip(recipe_to_index.values(), recipe_to_index.keys()))




num_rows = len(user_recipe['userid'])
num_col = len(unique_recipes)


users_to_recipes = user_recipe.groupby('userid')['recipeid'].unique()
# which movie IDs were rated by user 1?
users_to_recipes[1]

print(len(users_to_recipes))
