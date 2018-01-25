
import nltk
import re
import string
import cPickle

def stemmer(x):
    ps = nltk.PorterStemmer()
    try:
        x = ps.stem(x)
    except:
        x = ""
    return x

def Text2Wordlist(raw_text, stopwords_list = list()):
    '''
    get the word list of raw_text
    :param raw_text: one post
    :param stopwords_list: list of stopwords
    :return: list of words in raw_text
    '''
    raw_text = raw_text.lower().replace('&#39;','\'').replace('\xe2\x80\x99','\'').replace("&quot;",'\"') \
                .replace('i\'m','i am').replace('they\'re','they are').replace('we\'re','we are').replace('here\'s', 'here is') \
                .replace('there\'s','there is').replace('there\'re', 'there are').replace('\'ll',' will').replace('\'ve',' have') \
                .replace('can\'t', 'cannot').replace('won\'t','will not').replace('haven\'t','have not').replace('hasn\'t','has not')
    words = nltk.word_tokenize(raw_text.decode('utf-8'))
    if len(stopwords_list):
        words = list(filter(lambda x: x not in stopwords_list, words))

    words = list(filter(lambda x: x != "", map(lambda x: stemmer(x), words)))
    return words

def vocab_filter(vocab, threshold):
    sorted_list = sorted(vocab.items(),key=lambda d:d[1], reverse = True)

    vocab = dict()
    map(lambda x: vocab.update({x[0]: x[1]}), filter(lambda x: x[1] > threshold, sorted_list))
    vocab_size = len(vocab)
    print "vocab size = ", vocab_size
    keys = vocab.keys()
    new_vocab = dict({'': 0})
    for k, v in zip(keys, range(1, vocab_size + 1)):
        new_vocab.update({k: v})
    with open('vocab_' + str(threshold) +'.pkl', 'w') as f:
        cPickle.dump(new_vocab, f)
    return new_vocab

def getVocab(raw_text, threshold = 0, stopwords_list = list()):
    vocab = dict()
    words = Text2Wordlist(raw_text, stopwords_list)
    for w in words:
        if vocab.has_key(w):
            vocab[w] += 1
        else:
            vocab.update({w:1})
    with open('vocab_original.pkl', 'w') as f:
        cPickle.dump(vocab, f)
    vocab = vocab_filter(vocab, threshold)
    return vocab


def getIdx(word_list, vocab, max_length):
	index = list()
	keys = vocab.keys()
	for word in word_list:
		if word in keys:
			index.append(vocab[word])
	index += [0] * (max_length - len(index))
	return index[:max_length]

def Article2Index(texts, min_length, max_length, vocab, stopwords_list = list()):
    '''
    return list of indexes of texts
    :param texts:
    :param min_length:
    :param max_length:
    :param vocab:
    :param stopwords_list:
    :return:
    '''
    word_lists = map(lambda x: Text2Wordlist(x, stopwords_list), texts)
    indexes = map(lambda x: getIdx(x, vocab, max_length), word_lists)
    indexes = filter(lambda x: x[min_length]!=0, indexes)
    return indexes

if __name__ == '__main__':
    with open('Text.txt', 'r') as f:
        train_texts = f.read()
    raw_texts = ''
    stopwords_list = list(string.punctuation)
    min_freq = 5
    vocab = getVocab(train_texts, stopwords_list)
    idxs = Article2Index(raw_texts, 2, 5, vocab, stopwords_list)

