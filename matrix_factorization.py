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

'''PARAMETERS'''
userid = 2 # user id 
N = 2 # Number of recommendations for the user

#Create mock data of users with ratings for recipes
def create_data(data):
        userid = [i for i in range(0, 50)]
        #recipeid = data.index
        recipeid = [i for i in range(0,100)]
        new_dataframe = pd.DataFrame(list(product(userid, recipeid)), columns=['userid', 'recipeid'])
        new_dataframe['rating'] = np.random.randint(0, 5, size=(len(list(product(userid, recipeid)))))
        #new_dataframe = pd.DataFrame(np.random.randint(0,5,size=(100, len(data.index))), columns=[data.index])
        return new_dataframe

recipes = pd.read_csv('epi_r.csv')
unique_recipes = recipes.title
user_recipe = create_data(recipes)

# Prepare the data to perform matrix factorization
user_recipe_train = user_recipe[user_recipe['rating'] != 0]
reader = surprise.Reader(rating_scale=(1,4))
data = surprise.Dataset.load_from_df(user_recipe_train,reader)

# Algorithm
class MatrixFactorization(surprise.AlgoBase):

    def __init__(self,learning_rate,num_iterations,num_factors):
        self.alpha = learning_rate
        self.num_iterations = num_iterations
        self.num_factors = num_factors

    def fit(self,train):
        P = np.random.rand(train.n_users, self.num_factors) # initialize the model
        Q = np.random.rand(train.n_items, self.num_factors)

        for _ in range(self.num_iterations): # gradient descent
            for u,i,r_ui in train.all_ratings():
                error = r_ui - np.dot(P[u],Q[i])
                temp = P[u,:] 
                P[u,:] +=  self.alpha * error * Q[i] 
                Q[i,:] +=  self.alpha * error * temp 

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
            top_n[i[1]].append((i[3], i[2], recipes.iloc[i[1]]))
        top_n=sorted(top_n.items(), key = lambda k:k[1], reverse=True)
        return top_n[:n]

# Find optimal parameters based on the Grid Search
# Due to the long runtime, we have already added the optimal parameters for this model
gridsearch = surprise.model_selection.GridSearchCV(MatrixFactorization, 
                param_grid={'learning_rate':[0.0005], 'num_iterations':[200],
                    'num_factors':[10]},measures=['rmse'], cv=5)
gridsearch.fit(data)

best_params = gridsearch.best_params['rmse']
bestModel = MatrixFactorization(learning_rate=best_params['learning_rate'],
                num_iterations=best_params['num_iterations'],num_factors=best_params['num_factors'])
print('Best RSME with GridSearch: ',gridsearch.best_score['rmse'])
print('Best parameters with GridSearch: ', best_params)

# k-fold cross validation to find the best model and compute the recommendation
recommendation_per_fold = defaultdict(list)
kSplit = surprise.model_selection.KFold(n_splits=10,shuffle=True)
for train,test in kSplit.split(data):
    map_id_to_raw = defaultdict(list)
    estimated_ratings = []

    bestModel.fit(train)
    predictions = bestModel.test(test)
    accuracy = surprise.accuracy.rmse(predictions,verbose=True)
    
    for i in range(0, N):
        estimated_rating = bestModel.predict(user_recipe.userid.iloc[i],user_recipe.recipeid.iloc[i], r_ui=user_recipe.rating.iloc[i])
        estimated_ratings.append((estimated_rating.uid, estimated_rating.iid, estimated_rating.r_ui, estimated_rating.est))
    recommendation_per_fold[accuracy].append((estimated_ratings))

print('Best prediction with RMSE: ', min(recommendation_per_fold))
recommendations = bestModel.top_n_recommendations(N, recommendation_per_fold[min(recommendation_per_fold)][0], unique_recipes)
print('Recommendation for user:\n', recommendations)

print('YOUR TOP RECOMMENDATIONS IN DESCENDING ORDER')

gis = GoogleImagesSearch('AIzaSyCR4WYKfqWeUrxztz5gec9yUg8kH2K9-pA', '010356926349476305199:jim1wf2c680')
for recommendation in recommendations:
    for i in recommendation[1]:
        print('Recipe: ', i[2], ' has estimated rating: ', i[0])
        my_bytes_io = BytesIO()
        gis.search({'q': i[2], 'num': 1})

fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
for j in range(0, len(recommendations)):
    for i in recommendations[j][1]:
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
        d.text((10, 40), i[2], font=fnt, fill=(255, 255, 0))
        temp_img.show()
    