import pandas as pd
import numpy as np
import sys
import argparse
import csv

def creat_dicts(dataset):
    """
    Creates dictionary for users and for movies
    Parameters:
    -----------------------------------
    dataset: contains movie_id, customer_id, and movie_rating
    Returns:
    -----------------------------------
    usr_mv: dictionary of users and movies they've watched
    mve_usr: dictionary of movies and who has watched them
    """
    keys1 = set(dataset['user'])
    usr_mv = dict.fromkeys(keys1) # first dict
    keys2 = set(dataset['movie'])
    mve_usr = dict.fromkeys(keys2) # second dict
    for movie, user, rating in zip(dataset['movie'], dataset['user'], dataset['rating']):
        try:
            usr_mv[int(user)][int(movie)] = rating
        except:
            usr_mv[int(user)] = {int(movie): rating}
        
        try:
            mve_usr[int(movie)][int(user)] = rating
        except:
            mve_usr[int(movie)] = {int(user): rating}
return usr_mv, mve_usr

def open_file(filename):
    """
    Re-formatting of file
    Parameters:
    -----------------------------------
    filename: string parameter indicating file to format
    Returns:
    -----------------------------------
    drame: re-formatted dataframe
    """
    dframe = pd.read_csv(filename, sep = ",", header = None)
    dframe.columns=("movie", "user", "rating")
    return dframe

def average_rating(usr, dict):
    """
    Calculates average rating a user gives
    Parameters:
    -----------------------------------
    usr: user id
    dict: dictionary including movies they have watched
    Returns:
    -----------------------------------
    avg: average rating of usr
    """
    nest = dict[usr]
    avg = float(np.mean(list(nest.values())))
    return avg

def get_similarity(user1, user2, d1):
    """
    Calculates similarity between two users
    Parameters:
    -----------------------------------
    user1: user id
    user2: user id
    d1: dictionary indicating movies watched
    Returns:
    -----------------------------------
    similarity metric
    """
    items_user1 = d1[user1].keys()
    items_user2 = d1[user2].keys()
    common = set(items_user1).intersection(set(items_user2))
    mean_user1, mean_user2 = average_rating(user1, d1), average_rating(user2, d1)
    if bool(common)==False:
        return 0
    else:
        numer, denom1, denom2 = 0, 0, 0
        for item in common:
            try:
                numer += (d1[user1][item]-mean_user1)*(d1[user2][item] - mean_user2)
                denom1 += (d1[user1][item] - mean_user1)**2
                denom2 +=  (d1[user2][item] - mean_user2)**2
            except:
                pass

if denom1==0 or denom2==0:
    return 0
        else:
            return float(numer/np.sqrt(denom1*denom2))

def rating_for_movie(testuser, movie, dict_users, dict_movies): # dict movies is training set dictionary movie -> user -> rating
    """
    Calculates predicted rating for a movie
    Parameters:
    -----------------------------------
    testuser: user id
    movie: movie id to predict testuser's rating for
    dict_users: dictionary of users as keys and movies they've watched
    dict_movies: dictionary of movies as keys and viewers of that movie
    Returns:
    -----------------------------------
    num_pred: predicted rating for movie
    """ 
    users_rated = dict_movies[movie].keys()
    tot_sum = 0
    mult = 0
    for user in users_rated:
        try:
            tot_sum += abs(get_similarity(testuser, user, dict_users))
            mult += get_similarity(testuser, user, dict_users)*(dict_users[user][movie] - average_rating(user, dict_users))
        except:
            pass
    try:
        num_pred = float(average_rating(testuser, dict_users)) + (1/tot_sum)*mult
    except:
        num_pred = float(average_rating(testuser, dict_users))
    return num_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    args = parser.parse_args()
    
    traindata = open_file(args.train)
    testdata = open_file(args.test)
    
    # traindata = open_file('TrainingRatings.txt')
    # testdata = open_file('TestingRatings.txt')
    
    dicts = creat_dicts(traindata)
    userdict = dicts[0]
    moviedict = dicts[1]
    mse = []
    
    with open('predictions.txt', 'w') as txtfile:
        for row in testdata.values:
            try:
                writer = csv.writer(txtfile, delimiter = ' ')
                if int(row[1]) not in userdict and row[0] in moviedict: # new user but known movie
                    pred = np.mean(moviedict[row[0]].values())
                    txtfile.write("{0} {1} {2} {3}\n".format(row[0], int(row[1]), row[2], pred))
                    mse.append(row[2]-pred)
                elif int(row[1]) in userdict and row[0] not in moviedict:
                    pred = average_rating(int(row[1], userdict))
                    txtfile.write("{0} {1} {2} {3}\n".format(row[0], int(row[1]), row[2], pred))
                    mse.append(row[2]-pred)
                elif int(row[1]) not in userdict and row[0] not in moviedict:
                    pred = np.mean([moviedict[user].values() for user in moviedict])
                    txtfile.write("{0} {1} {2} {3}\n".format(row[0], int(row[1]), row[2], pred))
                    mse.append(row[2]-pred)
                else:
                    pred = rating_for_movie(int(row[1]), row[0], userdict, moviedict)
                    txtfile.write("{0} {1} {2} {3}\n".format(row[0], int(row[1]), row[2], pred))
                    mse.append(row[2]-pred)
            except:
                pass

mae = np.mean([abs(i) for i in mse])
    rmse = np.sqrt(np.mean([i**2 for i in mse]))
    sys.stdout.write("MAE = {0:.4f}\n".format(mae))
    sys.stdout.write("RMSE = {0:.4f}\n".format(rmse))









