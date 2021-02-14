import pandas as pd
import numpy as np
import scipy
from scipy.linalg import sqrtm

data = pd.read_csv("./data/votes.csv", sep="|")
n_votes = len(data)
data['user.id'] -= 1
data['item.id'] -= 1

data['user.id'] = data['user.id'].astype('str')
data['item.id'] = data['item.id'].astype('str')
users = data['user.id'].unique() #list of all users
movies = data['item.id'].unique() #list of all movies

print("Number of users", len(users))
print("Number of movies", len(movies))
print(data.head())

test = pd.DataFrame(columns=data.columns)
train = pd.DataFrame(columns=data.columns)
test_ratio = 0.2 #fraction of data to be used as test set.
for u in users:
    temp = data[data['user.id'] == u]
    n = len(temp)
    test_size = int(test_ratio*n)

temp = temp.sort_values('timestamp').reset_index()
temp.drop('index', axis=1, inplace=True)
    
dummy_test = temp.iloc[n-1-test_size :]
dummy_train = temp.iloc[: n-2-test_size]
    
test = pd.concat([test, dummy_test])
train = pd.concat([train, dummy_train])

def create_utility_matrix(data, formatizer = {'user':0, 'item': 1, 'value': 2}):
    
    """
        :param data:      Array-like, 2D, nx3
        :param formatizer:pass the formatizer
        :return:          utility matrix (n x m), n=users, m=items
    """
        
    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value']  
    
    userList = data.iloc[:,userField].tolist()
    itemList = data.iloc[:,itemField].tolist()
    valueList = data.iloc[:,valueField].tolist()
    
    users = list(set(data.iloc[:,userField]))
    items = list(set(data.iloc[:,itemField]))
    
    users_index = {users[i]: i for i in range(len(users))}
    
    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}
    
    for i in range(0,len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i]
        
    pd_dict[item][users_index[user]] = value
    
    X = pd.DataFrame(pd_dict)
    X.index = users
        
    itemcols = list(X.columns)
    items_index = {itemcols[i]: i for i in range(len(itemcols))}
    # users_index gives us a mapping of user_id to index of user
    # items_index provides the same for items
     
    return X, users_index, items_index

def svd(train, k):
    utilMat = np.array(train)
    
    # the nan or unavailable entries are masked
    mask = np.isnan(utilMat)
    masked_arr = np.ma.masked_array(utilMat, mask)
    item_means = np.mean(masked_arr, axis=0)
    
    # nan entries will replaced by the average rating for each item
    utilMat = masked_arr.filled(item_means)
    
    x = np.tile(item_means, (utilMat.shape[0],1))
    
     # we remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utilMat = utilMat - x
    
    # The magic happens here. U and V are user and item features
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)
    
    # we take only the k most significant features
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    
    s_root=sqrtm(s)
    
    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)
    
    UsV = UsV + x
    
    print("svd done")
    return UsV

def rmse(true, pred):
    # this will be used towards the end
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)# to test the performance over a different number of features


no_of_features = [8,10,12,14,17]

utilMat, users_index, items_index = create_utility_matrix(train)

for f in no_of_features: 
    svdout = svd(utilMat, k=f)
    pred = [] #to store the predicted ratings
    for _,row in test.iterrows():
        user = row['user.id']
        item = row['item.id']
        u_index = users_index[user]
        if item in items_index:
            i_index = items_index[item]
            pred_rating = svdout[u_index, i_index]
        else:
            pred_rating = np.mean(svdout[u_index, :])
        pred.append(pred_rating)
        print(str(f)+" > "+str(pred_rating))
        
print(rmse(test['rating'], pred))