import numpy as np
import sqlite3
import time

# Seed ensures reproducibility when initialising relevant data structures and shuffling training data
np.random.seed(42)

"""
This Python script implements a recommendation system through Matrix Factorisation using no external libraries other than numpy.

--OUTCOME
    The system should be capable of predicting the rating of a given user U for an item I, trained on past user-item data.
    
--COLLABORATIVE FILTERING APPROACH
    Collaborative filtering leverages the "wisdom of the crowd" to perform recommendation tasks. This assumes that customers 
    who have agreed in the past will likely agree in the future.
    
--MATRIX FACTORISATION [1]
    n_u = number of users
    n_i = number of items
    f = number of latent factors
    R = ratings matrix
    P = User embedding matrix
    Q = Item embedding matrix

    Matrix factorisation models aim to learn 2 embedding matrices (P and Q) through the training data.
    The product of P and Q should give a good approximation of R. P has a size of (u x f), and Q has a size of (f x i).
    This process is refined by accounting for the global mean of the training data, along with user and item biases.

    Each row in P represents the features of users (factors). Each row in Q represents the features of items.
    The learning process of matrix factorisation models is automated, so it does not require information about the 
    users and items prior to training.

    A prediction of user U for an item I is the result of the dot-product between u's factor vector and i's factor vector, 
    which are both obtained from the embedding matrices.

    
--LEARNING ALGORITHM
    Each epoch in the training process should aim to minimise the error between the predicted and actual ratings on the 
    training set. Since the mae is used on submissions, this metric is calculated at the end of each epoch to understand 
    the performance of the model with the relevant parameters.

    Stochastic gradient descent (SGD) is a fundamental optimization algorithm commonly employed to minimize the objective function 
    in matrix factorization-based recommendation systems. At each iteration, the user and item factor vectors are adjusted based on the prediction error, 
    learning rate, and regularization parameter. The learning rate governs the step size taken during parameter updates, 
    while the regularization term acts as a penalty factor to prevent overfitting by shrinking the factors towards zero. 
    Dynamic strategies, such as learning rate decay or adaptive learning rates like AdaGrad and Adam, are often employed to adjust the learning rate 
    during training dynamically. Additionally, convergence is typically assessed based on changes in the objective function 
    or parameter updates falling below a predefined threshold. Proper initialization of the factor vectors, 
    often achieved through random initialization from a suitable distribution, is crucial to prevent convergence to suboptimal solutions.
    Recommender systems work best when accounting for biases, so user and item biases will be incorporated into the SGD approach.

--PSEUDOCODE [2]
    NOTE prediction = global_mean + user_bias + item_bias + (dot product of P[u] and Q[i])

    NOTE Pseudocode used for the SGD Approach
    Training parameters will be decided by performing grid search on the 100k training set

    Training input = ratings(N_user, N_item) with empty values removed
    The following parameters are chosen through grid search
    N_factor = ?
    Learning rate lr = ?
    Regularization rp = ?

    Randomly initialise item matrix q(N_item, N_factor) 
    Randomly initialise user matrix p(N_user, N_factor)
    Randomly initialise user bias vector r(N_user)
    Randomly initialise item bias vector s(N_items)

    loop for N_iterations
        make a randomly shuffled set of (u,i) pairs
            loop on each (u,i) pair
                predicted_rating = q_i . p_u + r_u + s_i
                error = actual_rating - predicted_rating

                p(u,*) += lr * ( error * q(i, *) - rp * p(u,*) ) -adjust the given user factor vector to reduce loss
                q(i,*) += lr * ( error * p(u, *) - rp * q(i,*) ) -adjust the given item factor vector to reduce loss

                r_u += lr * (error - rp * r_u) -adjust the given user bias to reduce loss
                s_i += lr * (error - rp * s_i) -adjust the given item bias to reduce loss
"""


# Function to perform matrix factorisation (see [2] for details on pseudocode and design choices)
# Max user and item IDs passed as parameters to reduce computation overhead
def MATRIX_FACTORIZATION(K, learning_rate, reg_param, max_user_id, max_item_id, user_item_train_dict, test_data, num_epochs, val_data=None):
    global_mean = np.mean([r for (_, _), r in user_item_train_dict.items()]) # global mean to incorporate into training

    # variables initialised as per [1]
    # user and item biases initialised with 0s because random values introduces
    # additional randomness to training, affecting convergence
    bu = np.zeros(max_user_id+1) ; bi = np.zeros(max_item_id+1)

    # embedding matrices initialised with random values
    P = np.random.normal(scale=1./K, size=(max_user_id+1, K))
    Q = np.random.normal(scale=1./K, size=(K, max_item_id+1))

    # pre-trained embedding matrices and biases can be loaded from dat files to speed up training process
    # P = np.loadtxt("matP.dat", delimiter=",", skiprows=1)
    # Q = np.loadtxt("matQ.dat", delimiter=",", skiprows=1)
    # bu = np.loadtxt("bu.dat")
    # bi = np.loadtxt("bi.dat")
    
    prev_mae = np.inf
    for it in range(1,num_epochs+1):
        print(f'Starting Epoch {it}')
        errors = [] # track errors to print mae after each epoch
        items = list(user_item_train_dict.items())
        np.random.shuffle(items) # shuffle for randomness
        start = time.time()

        # NOTE Algorithm implemented based on pseudocode described in [2] 
        for count,((u,i), r) in enumerate(items,start=1):
            # compute prediction and calculate error
            pred = global_mean + bu[u] + bi[i] + np.dot(P[u, :], Q[:, i])
            err = r - pred
            
            # include biases in training process, and dynamically update relevant entries
            bu[u] += learning_rate * (err - reg_param * bu[u])
            bi[i] += learning_rate * (err - reg_param * bi[i])
            P[u, :] += learning_rate * (err * Q[:, i] - reg_param * P[u, :])
            Q[:, i] += learning_rate * (err * P[u, :] - reg_param * Q[:, i])
            errors.append(abs(err)) # used to calculate mae of training epoch 
            
            # Print update when halfway for debugging
            if count == 9302667: print(f"50% of training complete for Epoch {it}")

        mae = np.mean(errors)
        # adaptive learning rates
        # if mae < prev_mae: learning_rate *= 0.95
        prev_mae = mae

        # print time and mae for each epoch (debugging)
        print(f"Epoch {it} complete ({int(time.time()-start)}s), MAE: {mae}")

        # Early stopping is a technique that avoids overfitting, and is used on 20% of the training data
        # This suspends training if the MAE increases after the epoch
        if val_data:
            val_user_ids, val_item_ids = zip(*val_data.keys())
            val_ratings = np.array(list(val_data.values()))
            val_pred = global_mean + bu[val_user_ids] + bi[val_item_ids] + np.sum(P[val_user_ids] * Q[val_item_ids], axis=1)
            val_err = val_ratings - val_pred
            val_mae = np.mean(np.abs(val_err))
            if val_mae >= prev_mae:
                print("Validation MAE increased, stopping early.")
                break
            prev_mae = val_mae

    print("Training complete with mae: ",prev_mae)

    # NOTE predictions generated based on equation outlined in top comment
    print('---------------------GENERATE AND SAVE PREDICTIONS---------------------')
    predicted_ratings = []
    for user, item, timestamp in test_data:
        u = user-1 ; i = item-1
        # use same formula from training to calculate predictions
        pred_r = global_mean + bu[u] + bi[i] + np.dot(P[u, :], Q[:, i])
        pred_r = np.clip(pred_r, 0.5, 5) # clip and round predictions for improved accuracy
        predicted_ratings.append((user, item, np.round(pred_r), timestamp))

    # relevant parts of this method can be saved and loaded to avoid redundant computation
    # np.savetxt('matP.dat',P)
    # np.savetxt('matQ.dat',Q)
    # np.savetxt('bu.dat',bu)
    # np.savetxt('bi.dat',bi)
    return predicted_ratings


######################################
########### HELPER METHODS ###########
###################################### 

# Used to load training data from database file
# NOTE the db_example.py file is used to 
def load_from_db(filename):
    with sqlite3.connect(filename) as conn:
        c = conn.cursor()
        c.execute('SELECT UserID, ItemID, Rating FROM example_table')
        data = c.fetchall()
    return data

# Used to load testing and validation data from csv file
def load_from_csv(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return np.char.split(np.array(lines, dtype=str), sep=',').tolist()

# Used to write predictions array to file
# Assumed format of elements in pred array: [user,item,rating,timestamp]
def serialize(output,pred):
    with open(output,'w') as file:
        for u,i,r,t in pred:
            file.write(f"{u},{i},{r},{t}\n")
    print("Predictions saved to:",output)

# NOTE Only used in training to estimate MAE of submission
# The mae of the submission file from handing in was compared with the 
# generated predictions to deduce if those predictions would recieve a 
# similar mae when submitting.
def mae(path1,path2):
	results1 = open(path1).read().splitlines()
	results2 = open(path2).read().splitlines()

	results1 = list(map(lambda x: np.float16(x.split(',')[2]),results1))
	results2 = list(map(lambda x: np.float16(x.split(',')[2]),results2))
	differences = [abs(predicted - actual) for predicted, actual in zip(results1,results2)]
	mae_val = sum(differences)/len(differences)
	print('mae:',mae_val); return mae_val

######################################
########### HELPER METHODS ###########
######################################


print('--------------------------------SETUP--------------------------------')

print('Load and prepare train data')
train_data = load_from_db('train_20M_withratings.db')
# Store ratings matrix as a dictionary to save memory
# only consider positive ratings
user_item_train_dict = {(user_id, item_id): rating 
                        for user_id, item_id, rating in train_data if rating > 0}
# max user and item IDs used for initialising attributes for recommendation system
max_user_id = max(user_id for user_id, _, _ in train_data)
max_item_id = max(item_id for _, item_id, _ in train_data)

print('Load and prepare test data')
test_data = load_from_csv("test_20M_withoutratings.csv")
# Reformat test data rows in the form [user,item,timestamp] for predictions
# Ignore test data that contains ratings (applicable when using portion of train data for validation)
test_data = [list(map(int, values)) for values in test_data if len(values) == 3]


"""
The commented code below executes grid search with appropriate parameter ranges.
Grid search is a technique used to systematically explore a range of hyperparameters
to find the optimal combination that yields the best performance.

An essential aspect of grid search is its iterative nature, where insights gained from evaluating
one set of hyperparameters inform subsequent selections. This iterative refinement process ensures 
continuous improvement in model performance, and helps gain insights into how hyperparameter variations
can impact the system's convergence and prediction accuracy.

Large learning rates can result in overflow issues and prevent the model from converging
to an optimal solution. Similarly, large regularization parameters may lead to underfitting
of the model by excessively penalizing the weights, resulting in poor performance.

To mitigate these issues, some parameters are constrained to not exceed certain values
during the grid search process. For instance, learning rates and regularization parameters
are capped to ensure stability and prevent divergence of the optimization process.

Since the method prints out the Mean Absolute Error (MAE) after each epoch during training,
a default value of 100 epochs was passed for sufficient debugging time. This allows for
early detection of any anomalies or irregularities in the training process, such as sudden
increases in the MAE, which may indicate issues with parameter settings or convergence.

The output of the training process is observed iteratively to identify instances where the MAE
increases, signaling potential problems with the model's performance or parameter configuration.
Based on these observations, adjustments to the training parameters, such as learning rates
or regularization strengths, are made dynamically to improve the model's performance.

This iterative tuning process helps in determining the optimal choice of parameters to train
the model with, ensuring that it achieves the best possible performance on the given dataset.

Training a dataset with 20M entries proves to be computationally expensive, so the grid search
operations are carried out on the 100k dataset to obtain results faster. The Hold-out approach
involves using a subset of the training data to estimate generalisation capabilities. 10% of the 
training files was used to evaluate the performance of the recommendation system.
"""
# print('----------------------------GRID SEARCH----------------------------')
# Ks = np.arange(40,250,20)
# lrs = np.arange(0.01,0.03,0.001)
# rgs = np.arange(0.01,0.05,0.002)

# best_mae = np.inf
# best_params = []
# for K in Ks:
#     for lr in lrs:
#         for rg in rgs:
#             print(f'Training with K:{K}, learning rate:{lr}, reg param:{rg}')
#             predictions = MATRIX_FACTORIZATION(K=K,
#                     learning_rate=0.002,
#                     reg_param=0.05,
#                     max_user_id=max_user_id,
#                     max_item_id=max_item_id,
#                     user_item_train_dict=user_item_train_dict,
#                     user_item_pairs=test_data,
#                     num_epochs=5)
#             serialize('ratings.csv',predictions)
#             mae_gscv = mae('ratings.csv','20m_val.csv')
#             if mae_gscv < best_mae:
#                 print(f'New combination found - K:{K}, learning rate:{lr}, reg param:{rg}')
#             best_mae = mae
#             best_params = [K,lr,rg]

print('---------------------------MODEL TRAINING----------------------------')
predictions = MATRIX_FACTORIZATION(K=200,
                    learning_rate=0.005,
                    reg_param=0.01,
                    max_user_id=max_user_id,
                    max_item_id=max_item_id,
                    user_item_train_dict=user_item_train_dict,
                    test_data=test_data,
                    num_epochs=100)
serialize('submission.csv',predictions)


# NOTE This is the db_example.py file obtained to assist in building the database
# The database code was used only on files for training the code
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2021/02/11
# Project : Teaching
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, logging, os, shutil, subprocess, sqlite3, traceback, random

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

'''

# setup a ECS VM (which will run RedHat Enterprise 8 and therefore need Python 3.9)
# copy comp3208_100k_train_withratings.csv file to the same folder as db_example.py

sudo yum -y install gcc gcc-c++ python39-devel
sudo yum install python39 python39-pip
sudo python3.9 -m pip install --upgrade pip
sqlite3 comp3208.db
	CREATE TABLE IF NOT EXISTS example_table (UserID INT, ItemID INT, Rating FLOAT, PredRating FLOAT);
	.quit
python3.9 db_example.py

'''

if __name__ == '__main__':

	logger.info( 'loading training set and creating sqlite3 database' )

	# connect to database (using sqlite3 lib built into python)
	conn = sqlite3.connect( 'comp3208_example.db')

	#
	# comp3208-test-small.csv
	#
	readHandle = codecs.open( 'comp3208_100k_train_withratings.csv', 'r', 'utf-8', errors = 'replace' )
	listLines = readHandle.readlines()
	readHandle.close()

	c = conn.cursor()
	c.execute( 'CREATE TABLE IF NOT EXISTS example_table (UserID INT, ItemID INT, Rating FLOAT, PredRating FLOAT)' )
	conn.commit()

	c.execute( 'DELETE FROM example_table' )
	conn.commit()

	for strLine in listLines :
		if len(strLine.strip()) > 0 :
			# userid, itemid, rating, timestamp
			listParts = strLine.strip().split(',')
			if len(listParts) == 4 :
				# insert training set into table with a completely random predicted rating
				c.execute( 'INSERT INTO example_table VALUES (?,?,?,?)', (listParts[0], listParts[1], listParts[2], random.random() * 5.0) )
			else :
				raise Exception( 'failed to parse csv : ' + repr(listParts) )
	conn.commit()

	c.execute( 'CREATE INDEX IF NOT EXISTS example_table_index on example_table (UserID, ItemID)' )
	conn.commit()

	# run SQL to compute MAE
	c.execute('SELECT AVG(ABS(Rating-PredRating)) FROM example_table WHERE PredRating IS NOT NULL')
	row = c.fetchone()
	nMSE = float( row[0] )

	logger.info( 'example MAE for random prediction = ' + str(nMSE) )

	# run SQL to compute MAE against a fixed average rating
	c.execute('SELECT AVG(ABS(Rating-3.53)) FROM example_table WHERE PredRating IS NOT NULL')
	row = c.fetchone()
	nMSE = float( row[0] )

	logger.info( 'example MAE for user average of 3.53 prediction = ' + str(nMSE) )

	# close database connection
	c.close()
	conn.close()
"""
