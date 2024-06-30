import numpy as np
import math
from collections import defaultdict

################ METHODS #################

def load_data(file_path):
    # Data is loaded as a csv file
    data = open(file_path, 'r', encoding='utf-8').read().splitlines()
    # Format for rows in test data: (user id, item id, timestamp)
    # Format for rows in train data: (user id, item id, rating, timestamp)
    data = list(map(lambda x: x.split(','),data))
    return data

def nbhood_quality(size, is_user):
    if size == 0: return 0
    if size > 40: return 1
    
    if is_user:
        return 0.5 + 0.5 * (1 - np.exp(-0.16 * size))
    else:
        return 0.2 + 0.8 * (1 - np.exp(-0.16 * size))

def build_user_item_matrix(csv_data):
    user_item_ratings_train = defaultdict(dict)
    csv_array = np.array(csv_data)
    # Filter out rows with incorrect format
    valid_rows = csv_array[np.array([len(row) == 4 for row in csv_array])]
    for user_id, item_id, rating in valid_rows[:, :3].astype(np.float16):
        # slicing ensures that only the first three columns are considered
        # unpack the values directly to assign rating to appropriate cell
        user_item_ratings_train[user_id][item_id] = rating
    return user_item_ratings_train

def serialize_predictions(output_file, predictions):
    with open(output_file, "w") as file:
        for (user_id, item_id), (pred, timestamp) in predictions.items():
            # write data to file in same format as data in train file
            file.write(f"{user_id},{item_id},{pred},{timestamp}\n")
    print("Predictions saved to:", output_file)

# NOTE: The user-item matrix passed into cosine similarity methods is normalised to improve accuracy
# Similarity equation: similarity = i1_i2_dot_product / ((math.sqrt(i1_euclidian_d)) * (math.sqrt(i2_euclidian_d)))

def cosine_similarity_item(n_items, user_item_matrix):
    similarity_matrix = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            # find set of users who have reviewed both item i and item j
            mask = ~np.isnan(user_item_matrix[:, i]) & ~np.isnan(user_item_matrix[:, j])
            users_who_rated_both = np.nonzero(mask)[0]
            
            # 0 similarity if users have less than one item in common
            if len(users_who_rated_both) <= 1:
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
                continue
            
            i1_scores = user_item_matrix[users_who_rated_both, i]
            i2_scores = user_item_matrix[users_who_rated_both, j]
            numerator = np.dot(i1_scores, i2_scores)
            denominator = np.sqrt(np.sum(i1_scores ** 2)) * np.sqrt(np.sum(i2_scores ** 2))
            similarity = numerator / denominator # calculate similarity with formula mentioned above
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    np.fill_diagonal(similarity_matrix, 1)
    return similarity_matrix

def cosine_similarity_user(n_users, user_item_matrix):
    similarity_matrix = np.zeros((n_users, n_users))
    for i in range(n_users):
        for j in range(i + 1, n_users):
            # find set of items that both users have rated
            items_both_rated = np.nonzero(np.logical_and(~np.isnan(user_item_matrix[i]), ~np.isnan(user_item_matrix[j])))[0]

            # handle case where users have no items in common with ratings
            if len(items_both_rated) == 0:
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
                continue

            i1_scores = user_item_matrix[i, items_both_rated]
            i2_scores = user_item_matrix[j, items_both_rated]
            numerator = np.dot(i1_scores, i2_scores)
            denominator = np.sqrt(np.sum(i1_scores ** 2)) * np.sqrt(np.sum(i2_scores ** 2))
            similarity = numerator / denominator
            
            similarity_matrix[i, j] = similarity # flip i and j to speed computation
            similarity_matrix[j, i] = similarity

    np.fill_diagonal(similarity_matrix, 1)
    return similarity_matrix

def item_prediction(item_sim,user_item_matrix,i,j,n=115,not_nan_val=-50):
    numerator = 0
    denominator = 0

    # get the similarity row for the specific item
    # replace nan similarities with not_nal_val based on predicate
    similarities = item_sim[j, :]
    similarities = [not_nan_val if np.isnan(x) else similarities[i] for i, x in enumerate(user_item_matrix[i, :])]
    similarities = [not_nan_val if ((v > 0.3) or v==0 or v==not_nan_val) else v for v in similarities]

    # select indexes where value is not -1
    # extract the indexes of positive values and sort them in descending order
    sorted_indices = sorted(
        (index for index, value in enumerate(similarities) if value != not_nan_val),
        key=lambda i: abs(similarities[i]), reverse=True)

    sorted_indices = np.array(sorted_indices[:n]) # only keep n highest ratings
    top_n_indices = sorted_indices[sorted_indices != j] # exclude item j from the list if included
    nborhood = [item_sim[index,j] for index in top_n_indices]

    for nbor in top_n_indices:
        sim = item_sim[nbor,j]
        r = user_item_matrix[i,nbor]
        numerator += sim*r
        denominator += abs(sim)

    if denominator*numerator==0:
        return 0,0.3
    
    pred = numerator/denominator # compute predictions and handle cases where prediction is nan
    pred = 0 if np.isnan(pred) else pred
    return pred, nbhood_quality(len(nborhood),is_user=False) # return estimated quality with prediction

def user_prediction(user_sim,user_item_matrix,i,j,n=74):
    numerator = 0
    denominator = 0

    # get similarity scores for all users relative to user_id
    # filter users who have rated item_id and use this to find similarity scores
    users_who_rated_item = np.where(~np.isnan(user_item_matrix[:, j]))[0]
    useful_similarities = user_sim[i, :][users_who_rated_item]

    # neighborhood includes users with similarity greater than 0.3
    nbhood_indices = np.where(((useful_similarities > 0.3)) & (useful_similarities != 0))[0]
    sorted_user_indexes = np.argsort(-np.abs(useful_similarities[nbhood_indices])) 
    sorted_user_ids = users_who_rated_item[nbhood_indices][sorted_user_indexes]
    sorted_indices = np.array(sorted_user_ids[:n])
    # exclude the item itself from the list if included
    top_n_indices = sorted_indices[sorted_indices != i]

    # calculate estimated quality
    nborhoodB = [user_sim[index,i] for index in top_n_indices]
    quality = nbhood_quality(len(nborhoodB),is_user=True)

    # use neighborhood to calculate prediction
    for nb in top_n_indices:
        sim = user_sim[nb,i]
        r = user_item_matrix[nb,j]
        numerator += sim*r
        denominator += abs(sim)

    if denominator*numerator==0:
        return 0,0.7
    
    return numerator/denominator,quality



def mae(path1,path2):
	results1 = open(path1).read().splitlines()
	results2 = open(path2).read().splitlines()

	results1 = list(map(lambda x: np.float16(x.split(',')[2]),results1))
	results2 = list(map(lambda x: np.float16(x.split(',')[2]),results2))
	differences = [abs(predicted - actual) for predicted, actual in zip(results1,results2)]
	mae_val = sum(differences)/len(differences)
	print('mae:',mae_val)


################ METHODS #################

################ RECOMMENDATION SYSTEM IMPLEMENTATION #################

################ LOAD DATA AND OBTAIN ALL IDS #################

# Load CSV file as list of strings
print(f"Loading training and testing data")
train_data = load_data("train_100k_withratings.csv")
test_data = load_data("test_100k_withoutratings.csv")

print("Parse rows to use for predictions")
test_data = [list(map(int, values)) for values in test_data if len(values) == 3]

all_user_ids = list(range(1, 944)) # hard code user and item ids
all_item_ids = list(range(1, 1683))
all_user_ids_np = np.array(all_user_ids) # Convert to NumPy arrays for efficient indexing
all_item_ids_np = np.array(all_item_ids)

n_users = len(all_user_ids)
n_items = len(all_item_ids)

################ USER ITEM MATRIX COMPUTATION #################

# Parse training data and create user-item matrix
print(f"Use training data to build user by item matrix")
user_item_ratings_dict = build_user_item_matrix(train_data)

# Initialize user-item matrix with zeros
user_item_matrix_train = np.full((n_users, n_items),np.nan)

print(f"Populate sparse matrix for user-item matrix")
# Find the indices of the items in the sorted item IDs array
for i, usr in enumerate(all_user_ids):
    items_for_user = user_item_ratings_dict[usr]
    # Using binary search to find the insertion points of items in the sorted array
    j_indices = np.searchsorted(all_item_ids_np, list(items_for_user.keys()))
    user_item_matrix_train[i, j_indices] = list(items_for_user.values())


# Calculate the mean rating for each user, ignoring null/nans basically whatever I end up choosing
user_means = np.nanmean(user_item_matrix_train, axis=1)

# Subtract the user's mean rating from their ratings in the matrix to normalise
normalized_user_item_matrix = user_item_matrix_train.copy()
nan_mask = np.isnan(user_item_matrix_train) # avoid nan values when normalising
for u in range(n_users):
    normalized_user_item_matrix[u, ~nan_mask[u]] -= user_means[u]

################ SIMILARITY MATRIX COMPUTATION #################

# Compute similarity matrices
print(f"Generating similarity matrices for predictions")
user_item_matrix_train = normalized_user_item_matrix
user_sim_matrix = cosine_similarity_user(n_users,user_item_matrix_train)
item_sim_matrix = cosine_similarity_item(n_items,user_item_matrix_train)

################ PREDICTION CALCULATIONS AND SERIALIZATION #################

# Predict ratings for test data
print(f"Predicting ratings using training data")
predicted_ratings = {}
for user_id, item_id, timestamp in test_data:
    # Create boolean masks to check if user_id and item_id are present in the training data
    user_index_mask = (all_user_ids_np == user_id)
    item_index_mask = (all_item_ids_np == item_id)
    
    if not np.any(user_index_mask) or not np.any(item_index_mask):
        continue # Skip items not present in the training set
 
    # Find the index of the user_id and item_id in the training data arrays
    user_index = np.where(user_index_mask)[0][0]
    item_index = np.where(item_index_mask)[0][0]

    # Quality values are dependent on size of neighborhood: small neighborhood downweights rating
    user_pred, user_quality = user_prediction(user_sim_matrix,user_item_matrix_train,user_id-1,item_id-1) + user_means[user_id-1]
    item_pred, item_quality = item_prediction(item_sim_matrix,user_item_matrix_train,user_id-1,item_id-1) + user_means[user_id-1]

    # Use predictions and relevant quality to calculate weighted prediction
    pred = ((user_pred * user_quality) + (item_pred * item_quality)) / (user_quality + item_quality)
    predicted_ratings[(user_id, item_id)] = (round(pred), timestamp) # round pred to reduce mae


serialize_predictions("F:/SCT_CWK1/submission/submission.csv",predicted_ratings)
mae("F:/SCT_CWK1/submission/submission.csv","F:/SCT_CWK1/submission/FinaluserItemPrediction.csv")

################ RECOMMENDATION SYSTEM IMPLEMENTATION #################
