from model import *
from data_loader import data_loader
import os

msg_dir = 'stocktwits_samples/'
price_dir = 'stock_prices/'
if os.path.exists(msg_dir) and os.path.isdir(msg_dir):
    symbols = [f[:-5] for f in os.listdir(msg_dir) if f.endswith('.xlsx')]

#symbols = ['JPM']
train, hold, test = data_loader(symbols,  msg_dir = msg_dir,  price_dir = price_dir)

alpha = 0.05
joint_all = Joint_All(train, hold, test)
joint_all_list, p_bl = joint_all.train()
print 'the baseline prob = ', p_bl

per_user = Per_User(train, hold, test)
per_user_list = per_user.train(p_bl, alpha)
print '#experts = ', len(per_user.expert_id)
print '#per_user_tweets = ', len(per_user_list)

joint_expert = Joint_Experts(train, hold, test, per_user.expert_id)
joint_expert_list = joint_expert.train(p_bl, alpha)
print '#joint_expert_tweets = ', len(per_user_list)

print 'finish,,,'