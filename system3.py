from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import surprise
# from google_images_search import GoogleImagesSearch
# from IPython.display import display
from tqdm import tqdm


# Algorithm
class MatrixFactorization(surprise.AlgoBase):

    def __init__(self, learning_rate, num_iterations, num_factors, **kwargs):
        super().__init__(**kwargs)
        self.alpha = learning_rate
        self.num_iterations = num_iterations
        self.num_factors = num_factors

    def fit(self, train):
        P = np.random.rand(train.n_users, self.num_factors).astype('float64')  # initialize the model
        Q = np.random.rand(train.n_items, self.num_factors).astype('float64')
        for _ in tqdm(range(self.num_iterations)):  # gradient descent
            for u, i, r_ui in train.all_ratings():
                error = r_ui - np.dot(P[u, :], Q[i, :])
                temp = np.copy(P[u, :])
                P[u, :] += self.alpha * error * Q[i, :]
                Q[i, :] += self.alpha * error * temp

        self.P = P
        self.Q = Q
        self.trainset = train

    def estimate(self, u, i):
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            checkNan = np.dot(self.P[u], self.Q[i])
            if np.isnan(checkNan):
                return self.trainset.global_mean
            else:
                return np.dot(self.P[u, :], self.Q[i, :])
        else:  # if unknown return general average
            return self.trainset.global_mean

    @staticmethod
    def top_n_recommendations(n, recommendations, recipes):
        print('Use the model to calculate the top ', n, ' recommendations for the newly added user ......')
        top_n = defaultdict(list)
        for i in recommendations:
            top_n[i[1]].append((i[3], recipes.iloc[i[1]]))  # for now leave actual rating out. Some are empty
        top_n = sorted(top_n.items(), key=lambda k: k[1], reverse=True)
        if n == -1:
            return top_n
        else:
            return top_n[:n]


# Create mock data of users with ratings for recipes
def create_data():
    userid = [i for i in range(0, 50)]
    # recipeid = data.index
    recipeid = [i for i in range(0, 100)]
    new_dataframe = pd.DataFrame(list(product(userid, recipeid)), columns=['userid', 'recipeid'])
    new_dataframe['rating'] = np.random.randint(-2, 5, size=(len(list(product(userid, recipeid)))))
    # new_dataframe = pd.DataFrame(np.random.randint(0,5,size=(100, len(data.index))), columns=[data.index])
    return new_dataframe


def ask_user_input(unique_recipes, user_item_matrix, userid):
    print('Please give a rating from 0.5 - 5 for the following 10 recipes: INSERT [RATING], THEN PRESS [ENTER]')
    for i in range(0, 10):
        answer = input(unique_recipes[i] + ': ')
        user_item_matrix = user_item_matrix.append(pd.DataFrame([[userid, i, answer]], columns=['userid', 'recipeid', 'rating']))
    return user_item_matrix

<<<<<<< HEAD:system3.py

def get_recommendations(num_recommendations, num_users, user_id, num_test_recipes_per_user):
=======
def evaluate_relevance(data):
    print('Evaluating the relevance of the ranking on test data .....')
    data = data.sort_values(by=['userid', 'predicted'], ascending=False)
    dcg = {5: [], 10: [], 15: [], 20: []}
    idcg = {5: [], 10: [], 15: [], 20: []}
    p = [5, 10, 15, 20]
    unique_users = data['userid'].unique()
    for user in unique_users:
        user_predicted_set = data[['userid', 'actual']].loc[data['userid'] == user]
        user_actual_set = data[['userid', 'actual']].loc[data['userid'] == user]

        sub_dcg = 0
        sub_ndcg = 0
        for rank in p:
            user_predicted_subset = user_predicted_set.reset_index(drop=True)
            user_predicted_subset = user_predicted_subset.loc[:rank-1,]
        
            user_actual_subset = user_actual_set.sort_values(by=['userid', 'actual'], ascending=False)
            user_actual_subset = user_actual_subset.reset_index(drop=True)
            user_actual_subset = user_actual_subset.loc[:rank-1,]

            sub_dcg = calculate_DCG(user_predicted_subset, 'predicted')
            sub_ndcg = calculate_DCG(user_actual_subset, 'actual')
            dcg[rank].append(sub_dcg)
            idcg[rank].append(sub_ndcg)
    
    #print('DCG: ', dcg)
    #print('IDCG: ', idcg)
    for rank in p:
        print('NDCG for rank ', rank, ': ', np.mean(dcg[rank])/np.mean(idcg[rank]))

def calculate_DCG(data, column_name):
    dcg = 0
    for i in range(1, len(data)+1): 
        relevance = 0 if ((data.iloc[i-1]['actual'] < 4) or (data.index.isin([i-1]).any() == False)) else 1
        dcg += relevance / (math.log2(i+1))
    print(data)
    print(column_name, ' dcg: ', dcg)
    return dcg

def perform_Gridsearch(data):
    print('Performing the Grid Search .....')
    gridsearch = surprise.model_selection.GridSearchCV(MatrixFactorization, 
                    param_grid={'learning_rate':[0.001, 0.05, 0.01, 0,1], 'num_iterations':[i+100 for i in range(0,1000)],
                        'num_factors':[i for i in range(2,10)]},measures=['rmse'], cv=5)
    gridsearch.fit(data)

    best_params = gridsearch.best_params['rmse']
    bestModel = MatrixFactorization(learning_rate=best_params['learning_rate'],
                    num_iterations=best_params['num_iterations'],num_factors=best_params['num_factors'])
    print('Best RSME with GridSearch: ',gridsearch.best_score['rmse'])
    print('Best parameters with GridSearch: ', best_params)
    return bestModel

def print_recommended_images(recommendations):
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

def get_recommendations(num_recommendations, num_users, user_id, num_recipes_per_user):
>>>>>>> added ndcg and evaluation metrics:matrix_factorization.py
    recipes = pd.read_csv('epi_r.csv')
    unique_recipes = recipes.title
    print('Preparing User Item Matrix ..... ')
    user_recipe = pd.read_csv('recipe_ratings.csv', skiprows=1, names=['userid', 'recipeid', 'rating'])

    train_validation = pd.DataFrame()
    test = pd.DataFrame()
    user_recipe_user_id = user_recipe.loc[:, "userid"]

    for i in range(1, num_users + 1):
        train_validation_indices = user_recipe_user_id[user_recipe_user_id == i].index[num_test_recipes_per_user:]
        train_validation = train_validation.append(user_recipe.loc[train_validation_indices, :])
        test_index = user_recipe_user_id[user_recipe_user_id == i].index[:num_test_recipes_per_user, ]
        test = test.append(user_recipe.loc[test_index, :])

    train_validation = ask_user_input(unique_recipes, train_validation, user_id)
    train_validation = train_validation.reset_index(drop=True)
    # Prepare the data to perform matrix factorization
    reader = surprise.Reader(rating_scale=(0.5, 5))
    data = surprise.Dataset.load_from_df(train_validation, reader)

    # Find optimal parameters based on the Grid Search
    # Due to the long runtime, we have already added the optimal parameters for this model
<<<<<<< HEAD:system3.py
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
                                    num_iterations=200, num_factors=10)
=======
    # bestModel = perform_Gridsearch(data)

    bestModel = MatrixFactorization(learning_rate=0.001,
                                    num_iterations=200, num_factors=4)
>>>>>>> added ndcg and evaluation metrics:matrix_factorization.py
    # k-fold cross validation to find the best model and compute the recommendation
    recommendation_per_fold = defaultdict(list)
    print('\n User model created. Performing 5-fold CV to find the optimal model for recommendations .....')
    kSplit = surprise.model_selection.KFold(n_splits=5, shuffle=True)
    for train, validation in kSplit.split(data):
        bestModel.fit(train)
        predictions = bestModel.test(validation)
        accuracy = surprise.accuracy.rmse(predictions, verbose=True)
        recommendation_per_fold[accuracy].append(bestModel)

    best_fold = min(recommendation_per_fold)
    print('Found best model with RMSE: ', best_fold)

    print('Evaluate if the model is correct with use of the test set .....')
    test_results = pd.DataFrame(columns=['userid', 'recipeid', 'actual', 'predicted'])
    for i in range(0, len(test)):
        result = recommendation_per_fold[best_fold][0].predict(test.iloc[i]['userid'], test.iloc[i]['recipeid'], r_ui=test.iloc[i]['rating'])
<<<<<<< HEAD:system3.py
        test_results.append((test.iloc[i]['userid'], test.iloc[i]['recipeid'], test.iloc[i]['rating'], result.est))
    # TODO: apply the DCG on the above dataset

    print('Use the model to calculate the top ', num_recommendations, ' recommendations for the user ......')
=======
        test_results = test_results.append(
            pd.DataFrame([[test.iloc[i]['userid'], test.iloc[i]['recipeid'], test.iloc[i]['rating'], result.est]], 
            columns=['userid', 'recipeid', 'actual', 'predicted']))
    test_results = test_results.reset_index(drop=True)
    evaluate_relevance(test_results)

    # Recommend on the newly added user
>>>>>>> added ndcg and evaluation metrics:matrix_factorization.py
    estimated_ratings = []
    best_fold_model = recommendation_per_fold[best_fold][0]
    for i in range(len(unique_recipes)):  # Estimates a rating for each recipe for the user
        estimated_rating = best_fold_model.predict(user_id, i)
        estimated_ratings.append((user_id, i, estimated_rating.r_ui, estimated_rating.est))

    recommendations = bestModel.top_n_recommendations(num_recommendations, estimated_ratings, unique_recipes)
    print('Recommendation for the user:\n')
    for i in recommendations:
        print(i[1][0])

<<<<<<< HEAD:system3.py
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
    return recommendations

=======
    #print_recommended_images(recommendations)
>>>>>>> added ndcg and evaluation metrics:matrix_factorization.py

if __name__ == '__main__':
    num_recommendations = 5
    num_users = 50
    user_id = num_users + 1
    num_test_recipes_per_user = 5  # Maybe I'll remove this later on. It's only for the experiments of the DCG

<<<<<<< HEAD:system3.py
    result = get_recommendations(num_recommendations, num_users, user_id, num_test_recipes_per_user)
=======
    result = get_recommendations(num_recommendations, num_users, user_id, num_recipes_per_user)
>>>>>>> added ndcg and evaluation metrics:matrix_factorization.py
    print("Finished")
