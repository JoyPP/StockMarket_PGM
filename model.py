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
    bullish_idx = np.where(svm_model.classes_)[0][0]
    y_pred = y_pred[:, bullish_idx]

    idx = np.where(y_pred >= threshold)[0]

    return x_train[idx], y_train[idx]


def ChiSquareTest(c_u, i_u, c_bl, i_bl):
    test_stat = math.pow(c_u - c_bl, 2) / c_bl + math.pow(i_u - i_bl, 2) / i_bl
    return test_stat


def ExpertPValure(dataset, svm_model, p_bl, alpha, threshold = 0.5):
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




class Join_All:
    '''
    learn a single SVM from training data, apply it to test dataset, rank tweets according to the SVM score
    '''
    def __init__(self, train_data, hold_data, test_data):
        self.train_data = train_data
        self.hold_data = hold_data
        self.test_data = test_data
        self.clf = SVC()

    def train(self):
        x_train, y_train = self.train_data
        self.clf.fit(x_train, y_train)

    def test(self):
        x_test, y_test = self.test_data
        score = self.clf.score(x_test, y_test)


class Per_User:
    '''
        1, learn SVM_u for each user in training dataset;
        2, optimize the classification threshold (grid search) resulting in best p-value for each classifier in hold-out dataset.
    '''
    def __init__(self, train_data, hold_data, test_data):
        self.train_data = train_data
        self.hold_data = hold_data
        self.test_data = test_data

    def train(self):
        '''
        get a dict of experts.  {userid: {"threshold": t, "p_value": p_value}}
        '''
        self.expert_id = dict()
        p_list = list()
        rank_list = list()
        for userid, (x_train, y_train) in zip(self.train_data):
            if self.hold_data.has_key(userid) and self.test_data.has_key(userid):
                # learn SVM_u for each user from training dataset
                clf = SVC(probability=True)
                clf.fit(x_train, y_train)
                # optimize the classification threshold (grid search) resulting in best p-value for each classifier in hold-out dataset
                best_threshold, p_value = Best_Threshold(self.hold_data, clf, p_lb, alpha)
                if best_threshold is not None:
                    self.expert_id[userid] = dict()
                    p_list.append(p_value)
                    # data for user[userid]
                    data_u = get_bullish_set(self.test_data[userid], clf, best_threshold)
                    rank_list.append(data_u)
        rank_list = list(np.take(rank_list, np.argsort(p_list)))
        return rank_list



class Joint_Experts:
    def __init__(self, train_data, hold_data, test_data):
        self.train_data = train_data
        self.hold_data = hold_data
        self.test_data = test_data

    def train(self):
        pass

    def test(self):
        pass


