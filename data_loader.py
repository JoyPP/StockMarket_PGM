import os
import string
import pytz
import fasttext
import pandas as pd
import numpy as np
import cPickle
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from openpyxl.reader.excel import load_workbook


def fasttext_model_pretraining():
    '''
    use all extracted text to train fasttext model, saved into model.bin
    :return:
    '''
    directory = 'summary/'
    if os.path.isdir(directory):
        files = [directory+f for f in os.listdir(directory) if f.endswith('.xlsx')]
    else:
        print 'Summary directory Not Found!'
        return
    alltext = 'alltext.txt'
    if os.path.exists(alltext):
        os.remove(alltext)
    translator = string.maketrans(string.punctuation, " " * len(string.punctuation))
    for f in files:
        print 'file is ', f
        wb = load_workbook(filename=f)
        ws = wb.get_active_sheet()
        row, col = ws.max_row, ws.max_column
        col_dict = {1:'A', 2:'B', 3:'C',4:'D', 5:'E', 6:'F', 7: 'G', 8:'H', 9:'I', 10:'J',11:'K',12:'L',13:'M',14:'N'}
        col_range = [1] + range(5, col + 1)
        tmp = ""
        for i in range(2, row+1):
            for j in col_range:
                val = ws[col_dict[j]+str(i)].value
                if val is not None:
                    tmp += val + '\n'
            if i%100 == 0:
                with open(alltext, 'a') as fd:
                    fd.write(tmp.encode('utf-8').translate(translator))
                tmp = ""
        with open(alltext, 'a') as fd:
            fd.write(tmp.encode('utf-8').translate(translator))
    model = fasttext.skipgram(alltext, 'model')
    return model

def get_sen_label(texts):
    '''
    process the labels (texts)
    :param texts:
    :return: list of labels
    '''
    # eliminate whitespace and replace chinese characters into english character
    texts = texts.lower().strip().replace(" ", "").replace("\xef\xbc\x88", "(").replace("\xef\xbc\x89", ")").replace("\xef\xbc\x9b", ";").replace("\xef\xbc\x9a", ":").split(";")
    labels = list()
    for text in texts:
        '''
        if text in ['statement', 'analysis', 'comparison', 'pos', 'neg', 'neural', 'none']: # if label belongs to statement, analysis, comparison, sentiment or none
            labels.append(text)
        elif text.startswith('ep'): # if label is event prediciton, mark as 'ep_pos', 'ep_neg' or 'ep_neu'
            labels.append('ep_' + text[3:6])
        elif text.startswith('mp'): # if label is market prediction, mark as 'mp_direction_time_prediction e.g.: 'mp_up_1d'
            start = text.find("(")
            end = text.find(")")
            labels.append('mp_'+text[start+1:end].replace(":","_"))
        else:
            print 'cannot recognize label:', text
        '''
        if text in ['statement', 'analysis', 'comparison', 'neural', 'none']:
            labels.append('statement')
        elif text in ['pos', 'neg']:
            labels.append(text + '_none')
        elif text.startswith('ep'): # event prediciton, label as 'pos_none', 'neg_none' and 'neu_none'
            labels.append(text[3:6] + '_none')
        elif text.startswith('mp'): # market prediction, label 'mp(up:1d)' as 'pos_1d'
            start = text.find("(")
            end = text.find(")")
            labels.append(text[start + 1:end].replace(":", "_").replace('up','pos').replace('down','neg').replace('stay','neu'))
    return labels

def summary_preprocessing(symbol, model, directory = 'dataset/'):
    '''
    load summaries for the symbol
    :param symbol: stock symbol
    :return:
        summary_info: list saving summary, one dictionary {label: sentence_matrix} for each summary
        time_info: list saving time (transfer to US/Eastern timezone,
        author_info saving summary, time, author information respectively
    '''
    print 'Loading data for', symbol
    # read article information from summary_file
    summary_file = directory + symbol + '.xlsx'
    wb = load_workbook(filename=summary_file)
    ws = wb.get_active_sheet()
    row, col = ws.max_row, ws.max_column    # get max row and column number
    col_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
                13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T'}
    col_range = [1] + range(5, col + 1, 2) # related column number

    # transfer all punctuations to whitespace
    translator = string.maketrans(string.punctuation, " " * len(string.punctuation))

    # transfer timezone from utc to US/Eastern timezone
    utc = pytz.utc
    eastern = pytz.timezone('US/Eastern')
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'

    # list to save summary vector, US/Eastern time, and author information respectively
    summary_info, time_info, author_info = [], [], []
    # read summary_file
    row = 200
    for i in range(2, row + 1):
        # read summary and save its vector matrix into the summary_info
        summary = dict()
        for j in col_range:
            val = ws[col_dict[j]+str(i)].value
            if val is not None:
                val = val.encode('utf-8').translate(translator)
                if j == 1:  # title
                    summary["statement"] = val
                else: # summary
                    text = ws[col_dict[j+1]+str(i)].value
                    if text is None:    # missing labelled
                        labels = ['statement']
                    else:
                        labels = get_sen_label(text.encode('utf-8'))
                    for label in labels:
                        if label in summary.keys():
                            summary[label] += ' ' + val
                        else:
                            summary[label] = val
            else:
                break
        if 'statement' in summary.keys():
            words = summary['statement'].split()
            summary['statement'] = [model[w] for w in words]
            summary_info.append(summary['statement'])
        else:
            summary_info.append([[0]*(model.dim)])  # add an empty word list

        # read time and transfer it to US/Eastern Timezone and save it into the time_info
        t = ws['C'+str(i)].value.encode('utf-8')
        YY, MM, DD, hh, mm, ss = int(t[:4]), int(t[5:7]), int(t[8:10]), int(t[11:13]), int(t[14:16]), int(t[17:19])
        utc_dt = datetime(YY, MM, DD, hh, mm, ss, tzinfo=utc)
        est = utc_dt.astimezone(eastern)
        t = est.strftime(fmt)
        # preprocessing time to US/Eastern timezone, only show date
        # if the article is published before market closing, mark it current date
        # if the article is published after market closing, mark it next day
        # (next trading day to be considered)
        if (t.endswith('EDT-0400') and (t[11:19] <= '16:00:00')) or (
            t.endswith('EDT-0500') and (t[11:19] <= '17:00:00')):
            t = t[:10]
        else:
            t = (est + timedelta(days=1)).strftime(fmt)[:10]
        time_info.append(t) # only save the date

        # read author information and save it into the author_info
        author = ws['D'+str(i)].value.encode('utf-8')
        author_info.append(author)
    summary_info.reverse() #  time sequence
    time_info.reverse()
    author_info.reverse()
    return summary_info, time_info, author_info

def sen_padding(summary_info, feature_dim = 100, max_len = 40):
    '''
    padding each sentence to the given length (max_len)
    :param summary_info: sentences matrix needs to be padded
    :param max_len: length after padding
    :return: padded sentence matrix
    '''
    for i, s in enumerate(summary_info):
        if len(s) > max_len:
            summary_info[i] = s[:max_len]
        else:
            summary_info[i].extend([[0] * feature_dim] * (max_len - len(s)))
    return summary_info


def trend(target_price, basic_price, threshold = 0.01):
    '''
    :param target_price: price after time interval
    :param basic_price: price when predicting
    :param threshold: threshold of price changes
    :return: label of trend. 0: stay; 1: down; 2: up
    '''
    percent = (float(target_price) -  float(basic_price)) / float(basic_price)
    if percent >= threshold:
        return 1
    elif percent <= -threshold:
        return -1
    else:
        return 0

def price_preprocessing(symbol, time_info, time_interval, directory = 'dataset/'):
    # read stock price information from price_file
    price_file = directory + symbol + '.csv'
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    price_data = pd.read_csv(price_file, parse_dates=[0], index_col='Date', date_parser=dateparse, usecols=[0,4])
    price_data = price_data['Close']
    # get price after timedelta days from each of the time_info
    fmt = '%Y-%m-%d'
    targets_dict = {}
    time_intervals = set([7, 14, 21] + [time_interval])
    idx = price_data.index
    for t in time_info:
        for delta in time_intervals:
            target_time = (datetime.strptime(t, fmt) + timedelta(days=delta)).strftime(fmt)
            # if market is not open at t, then t-1
            while t not in idx:
                t = (datetime.strptime(t, fmt) - timedelta(days=1)).strftime(fmt)
            # if market is not open at target_time, then target_time+1
            while target_time not in idx:
                target_time = (datetime.strptime(target_time, fmt) + timedelta(days=1)).strftime(fmt)

            # set label
            label = trend(price_data[target_time], price_data[t])

            # save label into targets_dict
            if str(delta) in targets_dict:
                targets_dict[str(delta)].append(label)
            else:
                targets_dict[str(delta)] = list([label])
    return targets_dict

def author_vec_loader(author_link):
    '''
    get list of vectors corresponding to author_links
    :param author_link: list of author homepage links
    :return: list of vectors
    '''
    author_file = 'author.pkl'
    # get author_dict
    if os.path.exists(author_file):
        with open(author_file, 'r') as f:
            author_dict = cPickle.load(f)
    else:
        author_excel = 'author_info.xlsx'
        author_dict = dict()
        wb = load_workbook(filename=author_excel)
        ws = wb.get_active_sheet()
        row = ws.max_row
        col = ['B','C','D','E','F','G','H']
        for i in range(2, row + 1):
            vec = list()
            for j in col:
                vec.append(int(ws[j+str(i)].value))
            author_dict[ws['A'+str(i)].value.encode('utf-8')] = vec
        # save author_dict into pkl file
        with open(author_file, 'w') as f:
            cPickle.dump(author_dict, f)

    # match author with its vector
    author_vec = list()
    for link in author_link:
        author_vec.append(author_dict[link])
    return author_vec


def shuffle_samples(train_samples, train_labels, test_samples, test_labels):
    '''
    type(inputs) == 'list', shuffle then return list
    :param train_samples:
    :param train_labels:
    :param test_samples:
    :param test_labels:
    :return:
    '''
    # transfer to numpy.array
    train_samples = np.array(train_samples, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    test_samples = np.array(test_samples, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_num = train_samples.shape[0]
    test_num = test_samples.shape[0]
    print train_samples.shape, test_samples.shape

    # shuffling data
    print 'train data shuffling'
    train_idx = np.arange(train_num, dtype=np.int32)
    np.random.shuffle(train_idx)
    np.take(train_samples, train_idx, axis=0, out=train_samples)
    np.take(train_labels, train_idx, axis=0, out=train_labels)
    train_samples = train_samples.tolist()
    train_labels = train_labels.tolist()

    print 'test data shuffling'
    test_idx = np.arange(test_num, dtype=np.int32)
    np.random.shuffle(test_idx)
    np.take(test_samples, test_idx, axis=0, out=test_samples)
    np.take(test_labels, test_idx, axis=0, out=test_labels)
    test_samples = test_samples.tolist()
    test_labels = test_labels.tolist()

    return train_samples, train_labels, test_samples, test_labels

def data_division(data, batch_size, window_size, shuffle=False):
    '''
    data division with no-padding data
    :param data: tuple of (inputs, targets)
    :param batch_size:
    :param window_size:
    :return: training and test dataset (#batch, 2, batch_size, window_size, feature_dim)
    '''
    # inputs: (n_data, feature_dim)
    # targets: (n_data,)
    inputs, targets = data

    xsample, ysample = [], []
    for index in range(len(inputs) - window_size + 1):
        xsample.append(inputs[index: index + window_size])  # (#sample, window_size, feature_dim)
        ysample.append(targets[index + window_size - 1])    # (#sample, )

    n_sample = len(xsample)
    n_batch = n_sample // batch_size
    xsample = xsample[n_sample - n_batch*batch_size:]   # keep the newest data
    ysample = ysample[n_sample - n_batch*batch_size:]

    # divide into training and test dataset
    row = round(0.8 * n_batch)
    train_samples = xsample[:int(row)*batch_size]
    train_labels = ysample[:int(row)*batch_size]

    test_samples = xsample[int(row)*batch_size:]
    test_labels = ysample[int(row)*batch_size:]

    if shuffle:
        train_samples, train_labels, test_samples, test_labels = shuffle_samples(train_samples, train_labels, test_samples, test_labels)

    train_dataset = []
    for b in range(int(row)):
        x = train_samples[b*batch_size: (b+1)*batch_size]    # (#train_batch, batch_size, window_size, feature_dim)
        y = train_labels[b*batch_size: (b+1)*batch_size]   # (#train_batch, batch_size, 1)
        train_dataset.append((x, y))

    test_dataset = []
    for b in range(n_batch-int(row)):
        x = test_samples[b*batch_size: (b+1)*batch_size]    # (test_batch, batch_size, window_size, feature_dim)
        y = test_labels[b*batch_size: (b+1)*batch_size]   # (test_batch, batch_size, 1)
        test_dataset.append((x, y))

    return train_dataset, test_dataset


def data_loader_for_each_symbol(symbol, directory, max_len = 40, batch_size = 16, time_interval = 7, window_size = 10):
    '''
    :param symbol: symbol of the stock
    :param model: fasttext model
    :param directory: directory of file saving summary and price
    :param time_interval: prediction time interval
    :return: train_dataset, test_dataset consisting of tuples of (inputs, targets)
    '''

    if not os.path.exists('model.bin'):
        model = fasttext_model_pretraining()
    else:
        model = fasttext.load_model('model.bin')
    feature_dim = model.dim

    # get summary information for each symbol
    summary_info, time_info, author_link = summary_preprocessing(symbol, model, directory)

    # padding summary with fix len
    summary_info = sen_padding(summary_info, feature_dim, max_len)
    print '#data = ', len(summary_info)
    # get label (up or down) for each symbol
    targets_dict = price_preprocessing(symbol, time_info, time_interval, directory)

    # get author's vector
    author_vec = author_vec_loader(author_link)

    # divide data into training and test dataset and return
    return data_division((summary_info, targets_dict[str(time_interval)]), batch_size, window_size)


def data_loader(symbols, directory):
    train_dataset, test_dataset = [], []
    for symbol in symbols:
        train, test = data_loader_for_each_symbol(symbol, directory)
        train_dataset.extend(train)
        test_dataset.extend(test)
    return train_dataset, test_dataset