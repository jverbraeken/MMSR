# Importing the libraries
import pandas as pd
import numpy as np
import surprise
from itertools import product
from collections import defaultdict
from google_images_search import GoogleImagesSearch
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from tqdm import tqdm 
import time
import math

'''PARAMETERS'''
N = 5 # Number of recommendations for the user
number_of_users = 50
userid = number_of_users + 1 # user id 

#Create mock data of users with ratings for recipes
def create_data(data):
        userid = [i for i in range(0, 50)]
        #recipeid = data.index
        recipeid = [i for i in range(0,100)]
        new_dataframe = pd.DataFrame(list(product(userid, recipeid)), columns=['userid', 'recipeid'])
        new_dataframe['rating'] = np.random.randint(-2, 5, size=(len(list(product(userid, recipeid)))))
        #new_dataframe = pd.DataFrame(np.random.randint(0,5,size=(100, len(data.index))), columns=[data.index])
        return new_dataframe

def ask_user_input(unique_recipes, user_item_matrix, userid):
        print('Please give a rating from 0.5 - 5 for the following 10 recipes: INSERT [RATING], THEN PRESS [ENTER]')
        for i in range(0, 10):
            answer = input(unique_recipes[i] + ': ')
            user_item_matrix = user_item_matrix.append(pd.DataFrame([[userid, i, answer]], columns=['userid', 'recipeid', 'rating']))
        return user_item_matrix

recipes = pd.read_csv('epi_r.csv')
unique_recipes = recipes.title
print('Preparing User Item Matrix ..... ')
#user_recipe = create_data(recipes)
user_recipe = pd.read_csv('recipe_ratings.csv', skiprows=1, names=['userid', 'recipeid', 'rating'])
user_recipe = user_recipe.loc[user_recipe['userid'] < userid]
user_recipe = ask_user_input(unique_recipes, user_recipe, userid)
user_recipe.reset_index(drop=True)
# Prepare the data to perform matrix factorization
reader = surprise.Reader(rating_scale=(0.5,5))
data = surprise.Dataset.load_from_df(user_recipe,reader)

# Algorithm
class MatrixFactorization(surprise.AlgoBase):

    def __init__(self,learning_rate,num_iterations,num_factors):
        self.alpha = learning_rate
        self.num_iterations = num_iterations
        self.num_factors = num_factors

    def fit(self,train):
        P = np.random.rand(train.n_users, self.num_factors).astype('float128') # initialize the model
        Q = np.random.rand(train.n_items, self.num_factors).astype('float128')
        for _ in tqdm(range(self.num_iterations)): # gradient descent 
            for u,i,r_ui in train.all_ratings():
                error = r_ui - np.dot(P[u,:],Q[i,:])
                temp = np.copy(P[u,:])
                P[u,:] += self.alpha * error * Q[i,:] 
                Q[i,:] += self.alpha * error * temp

        self.P = P
        self.Q = Q
        self.trainset = train

    def estimate(self,u,i):
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            checkNan = np.dot(self.P[u],self.Q[i])
            if np.isnan(checkNan):
                return self.trainset.global_mean
            else:
                return np.dot(self.P[u,:],self.Q[i,:])
        else: #if unknown return general average
            return self.trainset.global_mean

    def top_n_recommendations(self, n, recommendations, recipes):
        top_n = defaultdict(list)
        for i in recommendations:
            top_n[i[1]].append((i[3], recipes.iloc[i[1]])) # for now leave actual rating out. Some our empty
        top_n=sorted(top_n.items(), key = lambda k:k[1], reverse=True)
        return top_n[:n]

# Find optimal parameters based on the Grid Search
# Due to the long runtime, we have already added the optimal parameters for this model
'''
print('Performing the Grid Search .....')
gridsearch = surprise.model_selection.GridSearchCV(MatrixFactorization, 
                param_grid={'learning_rate':[0.0005, 0.05], 'num_iterations':[200, 1000],
                    'num_factors':[10, 100]},measures=['rmse'], cv=5)
gridsearch.fit(data)

best_params = gridsearch.best_params['rmse']
bestModel = MatrixFactorization(learning_rate=best_params['learning_rate'],
                num_iterations=best_params['num_iterations'],num_factors=best_params['num_factors'])
print('Best RSME with GridSearch: ',gridsearch.best_score['rmse'])
print('Best parameters with GridSearch: ', best_params)
'''
bestModel = MatrixFactorization(learning_rate=0.0005,
                num_iterations=200,num_factors=10)
# k-fold cross validation to find the best model and compute the recommendation
recommendation_per_fold = defaultdict(list)
print('\n User model created. Performing 10-fold CV and finding the optimal model for recommendations .....')
kSplit = surprise.model_selection.KFold(n_splits=10,shuffle=True)
for train,test in kSplit.split(data):
    map_id_to_raw = defaultdict(list)

    bestModel.fit(train)
    predictions = bestModel.test(test)
    accuracy = surprise.accuracy.rmse(predictions,verbose=True)
    recommendation_per_fold[accuracy].append(bestModel)

best_fold = min(recommendation_per_fold)
print('Found best model with RMSE: ', best_fold)

print('Use the model to calculate the top ', N, ' recommendations for the user ......')
estimated_ratings = []
for i in range(0, len(unique_recipes)): # Estimates a rating for each recipe for the user
        estimated_rating = recommendation_per_fold[best_fold][0].predict(userid, i, r_ui=user_recipe['rating'].loc[(user_recipe['userid'] == userid) & (user_recipe['recipeid'] == i)])
        estimated_ratings.append((estimated_rating.uid, estimated_rating.iid, estimated_rating.r_ui, estimated_rating.est))

recommendations = bestModel.top_n_recommendations(N, estimated_ratings, unique_recipes)
print('Recommendation for the user:\n')
for i in recommendations:
    print(i[1][0])

'''
print('YOUR TOP RECOMMENDATIONS IN DESCENDING ORDER')
# API credentials replaced with 'xxxxxx' due to privacy reasons. You can use your own 
# Google API credentials to run this part.

gis = GoogleImagesSearch('xxxxxxx', 'xxxxxxxx')
for recommendation in recommendations:
    for i in recommendation[1]:
        print('Recipe: ', i[1], ' has estimated rating: ', i[0])
        my_bytes_io = BytesIO()
        gis.search({'q': i[1], 'num': 1})

fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
for j in range(0, 1):
    for i in recommendations[j]:
        ranking = j + 1
        rank = 'Rank ' + str(ranking)
        my_bytes_io.seek(0)
        # take raw image data
        raw_image_data = gis.results()[j].get_raw_data()
        # writes raw image data to the object
        gis.results()[j].copy_to(my_bytes_io, raw_image_data)
        # go back to address 0 so PIL can read it from start to finish
        my_bytes_io.seek(0)
        temp_img = Image.open(my_bytes_io)
        d = ImageDraw.Draw(temp_img)
        d.text((10,10), rank, font=fnt, fill=(255, 255, 0))
        d.text((10, 40), i[1], font=fnt, fill=(255, 255, 0))
        temp_img.show()
'''