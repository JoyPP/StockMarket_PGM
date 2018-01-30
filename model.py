from sklearn.svm import SVC
import numpy as np
import math
import scipy.stats as stats

def get_bullish_set(dataset, svm_model, threshold = 0.5):
    '''
    return bullish dataset from dataset which probability not less than threshold by svm_model
    :return:
    '''
    x_train, y_train = dataset
    y_pred = svm_model.predict_proba(x_train)
    bullish_idx = np.where(svm_model.classes_>0)[0][0]
    y_pred = y_pred[:, bullish_idx]

    idx = np.where(y_pred >= threshold)[0]

    return x_train[idx], y_train[idx]


def ExpertPValure(dataset, svm_model, p_bl, alpha, threshold = 0.5):
    '''
    dataset is one user's dataset
    test whether user is expert and his p_value if he is
    '''
    # get bullish dataset
    x_train, y_train = get_bullish_set(dataset, svm_model, threshold)

    c_u, i_u = 0, 0
    for i, x in enumerate(x_train):
        if y_train[i]: # really stock price up
            c_u += 1
        else:
            i_u += 1
    if c_u + i_u > 0:
        p_u = float(c_u) / (c_u + i_u)
    else:
        p_u = 0
    if p_u <= p_bl:
        return 0, 0   # non_expert
    else:
        c_bl = p_bl * (c_u + i_u)
        i_bl = (1 - p_bl) * (c_u + i_u)
        test_stat, p = stats.chisquare([c_u, i_u], [c_bl, i_bl])
        if p > alpha:
            return 0, 0 # non_expert
        else:
            return 1, p

def Best_Threshold(dataset, svm_model, p_bl, alpha):
    '''
    input varying threshold for svm_model to gain the best (minimal) p_value
    '''
    threshold = np.arange(0.05, 1, 0.05)
    best_p = None
    best_t = None
    for t in threshold:
        exp, p_value = ExpertPValure(dataset, svm_model, p_bl, alpha, t)
        if exp and (best_p is None or best_p > p_value):
            best_p = p_value
            best_t = t
    return best_t, best_p




class Joint_All:
    '''
    learn a single SVM from training data, apply it to test dataset, rank tweets according to the SVM score
    '''
    def __init__(self, train_data, hold_data, test_data):
        self.train_data, self.train_user = train_data
        self.hold_data, self.hold_user = hold_data
        self.test_data, self.test_user = test_data
        self.clf = SVC(probability=True)

    def train(self):
        '''
        return the ranking list and baseline precision
        '''
        # learning SVM from training and hold data
        x_train, y_train = self.train_data
        self.clf.fit(x_train, y_train)
        x_hold, y_hold = self.hold_data
        self.clf.fit(x_hold, y_hold)

        # counting the number of positive examples in training data
        c_bl = len(np.where(np.array(y_train) > 0)[0]) + len(np.where(np.array(y_hold) > 0)[0])
        total = len(y_train) + len(y_hold)
        p_bl = float(c_bl) / total

        # apply SVM to test data
        rank_list = list()  # list of (y_u, y_pred)
        x_test, y_test = self.test_data
        y_pred = self.clf.predict_proba(x_test)
        bullish_idx = np.where(self.clf.classes_>0)[0][0]
        y_pred = y_pred[:, bullish_idx] # get the probability of predicting as bullish
        rank_list.extend(map(lambda a,b: (a,b), y_test, y_pred))
        rank_list.sort(key=lambda tup: tup[1], reverse=True)
        final_list = map(lambda x:x[0], rank_list)
        return final_list, p_bl


class Per_User:
    '''
        1, learn SVM_u for each user in training dataset;
        2, optimize the classification threshold (grid search) resulting in best p-value for each classifier in hold-out dataset.
    '''
    def __init__(self, train_data, hold_data, test_data):
        self.train_data, self.train_user = train_data
        self.hold_data, self.hold_user = hold_data
        self.test_data, self.test_user = test_data

    def train(self, p_bl, alpha):
        '''
        get a dict of experts.  {userid: {"threshold": t, "p_value": p_value}}
        '''
        self.expert_id = list()
        p_list = list()
        rank_list = list()
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        for userid, (num_exp_train, bef_exp_train) in self.train_user.items():
            if num_exp_train and self.hold_user.has_key(userid) and self.test_user.has_key(userid):
                # test user's prediction
                x_u = x_train[bef_exp_train:bef_exp_train+num_exp_train]
                y_u = y_train[bef_exp_train:bef_exp_train+num_exp_train]
                if len(set(y_u)) < 2:
                    continue    #the user only predict one class
                # learn SVM_u for each user from training dataset
                clf = SVC(probability=True)
                clf.fit(x_u, y_u)
                # optimize the classification threshold (grid search) resulting in best p-value for each classifier in hold-out dataset
                best_threshold, p_value = Best_Threshold(self.hold_data, clf, p_bl, alpha)
                #best_threshold, p_value = np.random.uniform(), np.random.uniform()
                if best_threshold is not None:
                    # data for user[userid]
                    num_exp_test, bef_exp_test = self.test_user[userid]
                    if num_exp_test:
                        x_u, y_u = x_test[bef_exp_test: bef_exp_test+num_exp_test], y_test[bef_exp_test: bef_exp_test+num_exp_test]
                        pos_x, pos_y = get_bullish_set((x_u, y_u), clf, best_threshold)
                        rank_list.append((pos_y.tolist(), p_value))
                        self.expert_id.append(userid)
        rank_list.sort(key=lambda tup:tup[1])   # p_value from small to large
        if len(rank_list):
            final_list = reduce(lambda x,y: (x[0]+y[0],), rank_list)[0]
        else:
            final_list = []
        return final_list



class Joint_Experts:
    '''
    train a single joint SVM model from the tweets of experts (Per_User model)
    '''
    def __init__(self, train_data, hold_data, test_data, expert_list = None):
        self.train_data, self.train_user = train_data
        self.hold_data, self.hold_user = hold_data
        self.test_data, self.test_user = test_data
        self.expert_id = expert_list
        self.clf = SVC(probability=True)

    def train(self, p_bl, alpha):
        if self.expert_id is None:
            per_user = Per_User((self.train_data, self.train_user),(self.hold_data, self.hold_user),(self.test_data, self.test_user))
            per_user.train(p_bl, alpha)
            self.expert_id = per_user.expert_id

        if len(self.expert_id) == 0:
            return []

        score_list = list()
        rank_list = list()
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        # learn a single joint SVM model from tweets of experts in train dataset
        for userid in self.expert_id:
            num_exp_train, bef_exp_train = self.train_user[userid]
            x_u, y_u = x_train[bef_exp_train: bef_exp_train+num_exp_train], y_train[bef_exp_train: bef_exp_train+num_exp_train]
            self.clf.fit(x_u, y_u)
        # apply SVM model to tweets of experts in test dataset
        bullish_idx = np.where(self.clf.classes_ > 0)[0][0]
        for userid in self.expert_id:
            num_exp_test, bef_exp_test = self.test_user[userid]
            if num_exp_test == 0:
                continue

            x_u, y_u = x_test[bef_exp_test: bef_exp_test + num_exp_test], y_test[bef_exp_test: bef_exp_test + num_exp_test]
            y_pred = self.clf.predict_proba(x_u)[:,bullish_idx]

            rank_list.extend(map(lambda a, b: (a, b), y_u.tolist(), y_pred))
        rank_list.sort(key=lambda tup: tup[1], reverse=True)
        final_list = map(lambda x: x[0], rank_list)
        return final_list

