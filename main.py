from model import *
from data_loader import file_processing


symbol = 'BHP'
train_data, hold_data, test_data = file_processing(symbol)

alpha = 0.05
joint_all = Joint_All(train_data, hold_data, test_data)
joint_all_rank, p_bl = joint_all.train()

per_user = Per_User(train_data, hold_data, test_data)
per_user.train(p_bl, alpha)

joint_expert = Joint_Experts(train_data, hold_data, test_data)
joint_expert.train(p_bl, alpha)
