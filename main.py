from model import *
from data_loader import data_loader
import os
import cPickle
import time
from plot import plot

def analysis_prob(rank_list, threshold):
    rank_list = np.array(rank_list)
    precision = list()
    for th in threshold:
        idx = np.where(rank_list[:,1] >= th)[0]
        k = len(idx)
        rl = rank_list[:k,0]
        correct = len(np.where(rl > 0)[0])
        precision.append((k, float(correct)/k))
    return precision

def analysis_exp(rank_list, num_experts):
    precision = list()
    for i in num_experts:
        rl = reduce(lambda x,y: x+y, rank_list[:i])
        rl = np.array(rl)
        correct = len(np.where(rl > 0)[0])
        precision.append((len(rl), float(correct)/len(rl)))
    return precision


msg_dir = 'stocktwits_samples/'
price_dir = 'stock_prices/'
if os.path.exists(msg_dir) and os.path.isdir(msg_dir):
    symbols = [f[:-5] for f in os.listdir(msg_dir) if f.endswith('.xlsx')]

#symbols = ['JPM']
start = time.time()
train, hold, test = data_loader(symbols,  msg_dir = msg_dir,  price_dir = price_dir)
print 'loading data costs time = ', time.time() - start

alpha = 0.05
# joint all model
joint_all = Joint_All(train, hold, test)
p_bl = joint_all.train()
joint_all_list = joint_all.test()   # joint_all_list = joint_all.rank_list

joint_all_precision = analysis_prob(joint_all_list, np.arange(0.95,0,-0.05))
print 'the baseline prob = ', p_bl
with open('joint_all.pkl','w') as f:
    cPickle.dump(joint_all, f)
    cPickle.dump(joint_all_precision, f)

per_user = Per_User(train, hold, test)
per_user_list = per_user.train(p_bl, alpha) # per_user_list = per_user.rank_list
per_user_precision = analysis_exp(per_user_list, range(1,21))
print '#experts = ', len(per_user.expert_id)
print '#per_user_tweets = ', len(per_user_list)
with open('per_user.pkl','w') as f:
    cPickle.dump(per_user, f)

joint_expert = Joint_Experts(train, hold, test, p_bl, alpha, per_user.expert_id)
joint_expert.train()
joint_expert_list = joint_expert.test()     # joint_expert_list = joint_expert.rank_list

joint_expert_precision = analysis_prob(joint_expert_list, np.arange(0.95,0,-0.05))
print '#joint_expert_tweets = ', len(per_user_list)
with open('joint_expert.pkl','w') as f:
    cPickle.dump(joint_expert, f)
    cPickle.dump(joint_expert_precision, f)

plot(joint_all_precision, per_user_precision, joint_expert_precision, 'pic.png')

print 'finish'


