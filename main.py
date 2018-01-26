from model import *
from data_loader import data_loader


symbol = ['GOOGL']
train, hold, test = data_loader(symbol, msg_dir = 'stocktwits_samples/', price_dir = 'stock_prices/')

alpha = 0.05
joint_all = Joint_All(train, hold, test)
joint_all_rank, p_bl = joint_all.train()

per_user = Per_User(train, hold, test)
per_user_list = per_user.train(p_bl, alpha)

joint_expert = Joint_Experts(train, hold, test, per_user.expert_id)
joint_expert_list = joint_expert.train(p_bl, alpha)


print 'finish,,,'