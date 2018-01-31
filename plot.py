import matplotlib.pyplot as plt

def plot(joint_all_precision, per_user_precision, joint_expert_precision, pkl_name):
    '''
    plot precision of and save it into pkl_path
    :param pkl_path: path to save the fig
    :return:
    '''
    l1, = plt.plot(joint_all_precision[:, 0], joint_all_precision[:,1], 'r', label='joint all')
    l2, = plt.plot(per_user_precision[:, 0], per_user_precision[:, 1], 'b', label='per user')
    l3, = plt.plot(joint_expert_precision[:, 0], joint_expert_precision[:, 1], 'g', label='joint experts')
    plt.legend(handles=[l1, l2, l3])
    plt.title('Empirical model comparison')
    plt.xlabel('Tweets')
    plt.ylabel('Precision')
    plt.savefig(pkl_name)