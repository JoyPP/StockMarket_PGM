from model import *
from data_loader import data_loader
import os
import cPickle
import time

def analysis(rank_list, threshold):
    pass

msg_dir = 'stocktwits_samples/'
price_dir = 'stock_prices/'
if os.path.exists(msg_dir) and os.path.isdir(msg_dir):
    symbols = [f[:-5] for f in os.listdir(msg_dir) if f.endswith('.xlsx')]

#symbols = ['JPM']
start = time.time()
train, hold, test = data_loader(symbols,  msg_dir = msg_dir,  price_dir = price_dir)
print 'loading data costs time = ', time.time() - start

alpha = 0.05
joint_all = Joint_All(train, hold, test)
p_bl = joint_all.train()
joint_all_list = joint_all.test()   # joint_all_list = joint_all.rank_list
print 'the baseline prob = ', p_bl
with open('joint_all.pkl','w') as f:
    cPickle.dump(joint_all, f)

per_user = Per_User(train, hold, test)
per_user_list = per_user.train(p_bl, alpha) # per_user_list = per_user.rank_list
print '#experts = ', len(per_user.expert_id)
print '#per_user_tweets = ', len(per_user_list)
with open('per_user.pkl','w') as f:
    cPickle.dump(per_user, f)

joint_expert = Joint_Experts(train, hold, test, p_bl, alpha, per_user.expert_id)
joint_expert.train()
joint_expert_list = joint_expert.test()     # joint_expert_list = joint_expert.rank_list
print '#joint_expert_tweets = ', len(per_user_list)
with open('joint_expert.pkl','w') as f:
    cPickle.dump(joint_expert, f)


print 'finish'


