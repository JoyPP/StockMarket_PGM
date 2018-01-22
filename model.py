from sklearn.svm import SVC
from sklearn.feature_selection import chi2
import numpy as np

x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([0, -1, 0, 1])
clf = SVC(probability=True)
clf.fit(x, y)
clf.predict([[-0.8, -1], [0.4,0.5]])

def ExpertPValure(dataset, svm_model, p_bl, alpha, threshold = None):
    x_train, y_train = dataset
    y_pred = svm_model.predict_proba(x)

    c_u, i_u = 0, 0
    for i, x in enumerate(x_train):
        if y_pred[i]:   # predicted bullish
            if y[i]: # really stock price up
                c_u += 1
            else:
                i_u += 1
        else:
            continue
    p_u = float(c_u) / (c_u + i_u)
    if p_u <= p_bl:
        return 0    # non_expert
    else:
        c_bl = p_bl * (c_u + i_u)
        i_bl = (1 - p_bl) * (c_u + i_u)
        p = Chisqueare(c_u, i_u, c_bl, i_bl)
        if p > alpha:
            return 0 # non_expert
        else:
            return 1, p



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
        threshold = np.arange(0.1,1,0.1)
        for userid, (x_train, y_train) in zip(self.train_data):
            if self.hold_data.has_key(userid) and self.test_data.has_key(userid):
                clf = SVC(probability=True)
                clf.fit(x_train, y_train)
                ExpertPValure(self.hold_data, clf, p_lb, alpha)
                y_pred = clf.predict_proba()





    def test(self):
        pass


class Joint_Experts:
    def __init__(self, train_data, hold_data, test_data):
        self.train_data = train_data
        self.hold_data = hold_data
        self.test_data = test_data

    def train(self):
        pass

    def test(self):
        pass


