import pandas as pd
import numpy as np
import surprise
from itertools import product
from collections import defaultdict

'''PARAMETERS'''
userid = 2
N = 10

def create_data(data):
        userid = [i for i in range(0, 50)]
        #recipeid = data.index
        recipeid = [i for i in range(0,100)]
        new_dataframe = pd.DataFrame(list(product(userid, recipeid)), columns=['userid', 'recipeid'])
        new_dataframe['rating'] = np.random.randint(0, 5, size=(len(list(product(userid, recipeid)))))
        #new_dataframe = pd.DataFrame(np.random.randint(0,5,size=(100, len(data.index))), columns=[data.index])
        return new_dataframe

class MatrixFactorization(surprise.AlgoBase):

    def __init__(self,learning_rate,num_iterations,num_factors):
        self.alpha = learning_rate
        self.num_iterations = num_iterations
        self.num_factors = num_factors

    def fit(self,train):
        P = np.random.rand(train.n_users, self.num_factors) # initialize the model
        Q = np.random.rand(train.n_items, self.num_factors)

        for iteration in range(self.num_iterations): # gradient descent
            for u,i,r_ui in train.all_ratings():
                error = r_ui - np.dot(P[u],Q[i]) #Here we consider the squared error because the estimated rating can be either higher or lower than the real rating.
                temp = P[u,:] # update variable at the same time 
                P[u,:] +=  self.alpha * error * Q[i] 
                Q[i,:] +=  self.alpha * error * temp 

        self.P = P
        self.Q = Q
        self.trainset = train

    #returns estimated rating for user u and item i
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
        #top_n[].sort(key = lambda x: x[3], reverse = True)
        return top_n

recipes = pd.read_csv('epi_r.csv')
unique_recipes = recipes.title
user_recipe = create_data(recipes)
unique_users = user_recipe['userid'].unique()

user_to_index = dict(zip(unique_users, range(len(unique_users))))
recipe_to_index = dict(zip(unique_recipes, range(len(unique_recipes))))
index_to_user = dict(zip(user_to_index.values(), user_to_index.keys()))      
index_to_recipe = dict(zip(recipe_to_index.values(), recipe_to_index.keys()))

user_recipe_train = user_recipe[user_recipe['rating'] != 0]
reader = surprise.Reader(rating_scale=(1,4))
data = surprise.Dataset.load_from_df(user_recipe_train,reader)

gridsearch = surprise.model_selection.GridSearchCV(MatrixFactorization, 
                param_grid={'learning_rate':[0.0005], 'num_iterations':[200],
                    'num_factors':[10]},measures=['rmse'], cv=5)
gridsearch.fit(data)

best_params = gridsearch.best_params['rmse']
bestModel = MatrixFactorization(learning_rate=best_params['learning_rate'],
                num_iterations=best_params['num_iterations'],num_factors=best_params['num_factors'])
print('Best RSME with GridSearch: ',gridsearch.best_score['rmse'])
print('Best parameters with GridSearch: ', best_params)

recommendation_per_fold = defaultdict(list)
# k-fold cross validation to evaluate the best model. 
kSplit = surprise.model_selection.KFold(n_splits=10,shuffle=True)
for train,test in kSplit.split(data):
    map_id_to_raw = defaultdict(list)
    estimated_ratings = []

    bestModel.fit(train)
    predictions = bestModel.test(test)
    accuracy = surprise.accuracy.rmse(predictions,verbose=True)

    #for id_value in bestModel.trainset.all_items():
    #    map_id_to_raw[id_value].append(bestModel.trainset.to_raw_iid(id_value))
    
    for i in range(0, 10):
        estimated_rating = bestModel.predict(user_recipe.userid.iloc[i],user_recipe.recipeid.iloc[i], r_ui=user_recipe.rating.iloc[i])
        estimated_ratings.append((estimated_rating.uid, estimated_rating.iid, estimated_rating.r_ui, estimated_rating.est))
    recommendation_per_fold[accuracy].append((estimated_ratings))
    #recommendation_per_fold[accuracy].append(top_n)

print('Best prediction with RMSE: ', min(recommendation_per_fold))
#print(
hoi = bestModel.top_n_recommendations(5, recommendation_per_fold[min(recommendation_per_fold)][0], unique_recipes)#, '\n\n')
#print('Recommendation for user ', userid)
#for i in recommendation_per_fold[min(recommendation_per_fold)][1]:
#    print(i)





