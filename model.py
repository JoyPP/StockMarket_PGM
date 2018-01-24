from sklearn.svm import SVC
import numpy as np
import math

x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([0, -1, 0, 1])
clf = SVC(probability=True)
clf.fit(x, y)
clf.predict([[-0.8, -1], [0.4,0.5]])

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


def ChiSquareTest(c_u, i_u, c_bl, i_bl):
    '''
    return the chi square test statistic
    '''
    test_stat = math.pow(c_u - c_bl, 2) / c_bl + math.pow(i_u - i_bl, 2) / i_bl
    return test_stat


def ExpertPValure(dataset, svm_model, p_bl, alpha, threshold = 0.5):
    '''
    dataset is one user's dataset
    test whether user is expert and his p_value if he is
    '''
    # get bullish dataset
    x_train, _ = get_bullish_set(dataset, svm_model, threshold)

    c_u, i_u = 0, 0
    for i, x in enumerate(x_train):
        if y[i]: # really stock price up
            c_u += 1
        else:
            i_u += 1
    p_u = float(c_u) / (c_u + i_u)
    if p_u <= p_bl:
        return 0, 0   # non_expert
    else:
        c_bl = p_bl * (c_u + i_u)
        i_bl = (1 - p_bl) * (c_u + i_u)
        p = ChiSquareTest(c_u, i_u, c_bl, i_bl)
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
        self.train_data = train_data
        self.hold_data = hold_data
        self.test_data = test_data
        self.clf = SVC(probability=True)

    def train(self):
        '''
        return the ranking list and baseline precision
        '''
        # learning SVM from training data
        for userid, (x_u, y_u) in zip(self.train_data):
            # counting the number of positive examples in training data
            c_bl = len(np.where(np.array(y_u)>0)[0])
            total = len(y_u)
            self.clf.fit(x_u, y_u)
        p_bl = float(c_bl) / total
        # apply SVM to test data
        rank_list = list()  # list of (x_u, y_u, y_pred)
        for userid, (x_u, y_u) in zip(self.test_data):
            y_pred = self.clf.predict_proba(x_u)
            bullish_idx = np.where(self.clf.classes_>0)[0][0]
            y_pred = y_pred[:, bullish_idx]
            rank_list.extend(map(lambda a,b,c: (a,b,c), x_u, y_u, y_pred))
            rank_list.sort(key=lambda tup: tup[2], reverse=True)
        return rank_list, p_bl


class Per_User:
    '''
        1, learn SVM_u for each user in training dataset;
        2, optimize the classification threshold (grid search) resulting in best p-value for each classifier in hold-out dataset.
    '''
    def __init__(self, train_data, hold_data, test_data):
        self.train_data = train_data
        self.hold_data = hold_data
        self.test_data = test_data

    def train(self, p_bl, alpha):
        '''
        get a dict of experts.  {userid: {"threshold": t, "p_value": p_value}}
        '''
        self.expert_id = list()
        p_list = list()
        rank_list = list()
        for userid, (x_train, y_train) in zip(self.train_data):
            if self.hold_data.has_key(userid) and self.test_data.has_key(userid):
                # learn SVM_u for each user from training dataset
                clf = SVC(probability=True)
                clf.fit(x_train, y_train)
                # optimize the classification threshold (grid search) resulting in best p-value for each classifier in hold-out dataset
                best_threshold, p_value = Best_Threshold(self.hold_data, clf, p_bl, alpha)
                if best_threshold is not None:
                    self.expert_id.append(userid)
                    p_list.append(p_value)
                    # data for user[userid]
                    data_u = get_bullish_set(self.test_data[userid], clf, best_threshold)
                    rank_list.append(data_u)
        rank_list = list(np.take(rank_list, np.argsort(p_list)))
        return rank_list



class Joint_Experts:
    '''
    train a single joint SVM model from the tweets of experts (Per_User model)
    '''
    def __init__(self, train_data, hold_data, test_data):
        self.train_data = train_data
        self.hold_data = hold_data
        self.test_data = test_data
        self.clf = SVC(probability=True)

    def train(self, p_bl, alpha):
        per_user = Per_User(self.train_data, self.hold_data, self.test_data)
        per_user.train(p_bl, alpha)
        self.expert_id = per_user.expert_id

        score_list = list()
        rank_list = list()
        for userid in self.expert_id:
            x, y = self.train_data[userid]
            self.clf.fit(x, y)
        for userid in self.expert_id:
            data_u = get_bullish_set(self.test_data[userid], self.clf)
            x_u, y_u = data_u
            score_u = self.clf.score(x_u, y_u)
            score_list.append(score_u)
            rank_list.append(data_u)
        rank_list = list(np.take(rank_list, np.argsort(score_list)))
        return rank_list.reverse()  # from large to small score


