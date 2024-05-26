import sys
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')

from nltk import word_tokenize
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

word_emb_path = 'datasets/glove.840B.300d.txt'

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']


# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK

def mask_vector(max_size, arr):
    # print (arr,arr.shape)
    if arr.shape[0] > max_size:
       output = [1]*max_size
    else:
       len_zero_value = max_size -  arr.shape[0]
       output = [1]*arr.shape[0] + [0]*len_zero_value

    return np.array(output)

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    #for line in tqdm_notebook(f, total=embedding_vocab):
    for line in tqdm(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()


class CoMic:
    def __init__(self, path_dir, task, words = True):
        
        if path_dir is None:
            print("Dataset files path is not specified. Please specify first.")
            exit(0)
        else:
            sys.path.append(str(path_dir))
        
        DATA_PATH = str(path_dir)
        #try:
        train_set = json.load(open(f'{DATA_PATH}/training_features_binary_allCaps.json'))
        validation_set = json.load(open(f'{DATA_PATH}/val_features_binary_allCaps.json'))
        test_set = json.load(open(f'{DATA_PATH}/test_features_binary_allCaps.json'))

        self.train = train = []
        self.dev = dev = []
        self.test = test = []
        self.word2id = word2id

        for tr in train_set:

            mid = train_set[tr]['IMDBid']

            image = np.load('Features/i3D_vecs/' + mid + "_rgb.npy") + np.load('Features/i3D_vecs/' + mid + "_flow.npy")
            mask_img = mask_vector(36, image)

            try:
                audio = np.load('Features/vgg_vecs/' + mid + '_vggish.npy')
            except:
                audio = np.array(128 * [0.0])
            mask_aud = mask_vector(63, audio)
            
            label = train_set[tr]['y']

            if words:
                _words = train_set[tr]['words']

                actual_words = []
                words_ = []

                for idx, word in enumerate(_words):
                    actual_words.append(word)
                    words_.append(word2id[word])

                __words = np.asarray(words_)

                train.append(((__words, image, mask_img, audio, mask_aud, actual_words), label, None))
            else:
                __words = train_set[tr]['indexes']

                train.append(((__words, image, mask_img, audio, mask_aud, __words), label, None))

        for val in validation_set:

            mid = validation_set[val]['IMDBid']

            image = np.load('Features/i3D_vecs/' + mid + "_rgb.npy") + np.load('Features/i3D_vecs/' + mid + "_flow.npy")
            mask_img = mask_vector(36, image)

            try:
                audio = np.load('Features/vgg_vecs/' + mid + '_vggish.npy')
            except:
                audio = np.array(128 * [0.0])
            mask_aud = mask_vector(63, audio)
            
            label = validation_set[val]['y']

            if words:
                _words = validation_set[val]['words']

                actual_words = []
                words_ = []

                for idx, word in enumerate(_words):
                    actual_words.append(word)
                    words_.append(word2id[word])

                __words = np.asarray(words_)

                dev.append(((__words, image, mask_img, audio, mask_aud, actual_words), label, None))
            else:
                __words = validation_set[val]['indexes']

                dev.append(((__words, image, mask_img, audio, mask_aud, __words), label, None))

        for ts in test_set:

            mid = test_set[ts]['IMDBid']

            image = np.load('Features/i3D_vecs/' + mid + "_rgb.npy") + np.load('Features/i3D_vecs/' + mid + "_flow.npy")
            mask_img = mask_vector(36, image)

            try:
                audio = np.load('Features/vgg_vecs/' + mid + '_vggish.npy')
            except:
                audio = np.array(128 * [0.0])
            mask_aud = mask_vector(63, audio)
            
            label = np.array(test_set[ts]['y'])
            #label = np.nan_to_num(label)

            if words:
                _words = test_set[ts]['words']

                actual_words = []
                words_ = []

                for idx, word in enumerate(_words):
                    actual_words.append(word)
                    words_.append(word2id[word])

                __words = np.asarray(words_)

                test.append(((__words, image, mask_img, audio, mask_aud, actual_words), label, None))
            else:
                __words = test_set[ts]['indexes']
                __words = np.asarray(__words)

                test.append(((__words, image, mask_img, audio, mask_aud, __words), label, None))

        word2id.default_factory = return_unk

        """except:
            ValueError('Files does not exits in the root specified. Please verify it.')"""

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id
        elif mode == "val":
            return self.dev, self.word2id
        elif mode == "test":
            return self.test, self.word2id
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class Hate:
    def __init__(self, path_dir, task, words = True):
        
        if path_dir is None:
            print("Dataset files path is not specified. Please specify first.")
            exit(0)
        else:
            sys.path.append(str(path_dir))
        
        DATA_PATH = str(path_dir)
        
        file = f'{DATA_PATH}/dict_hsdv_batch_250_labels.json'
        data = json.load(open(file))

        train_aux, test_split, y_train_aux, y_test = train_test_split(list(data.keys()), list(data.values()), test_size = 0.1)
        train_split, val_split, y_train, y_val = train_test_split(train_aux, y_train_aux, test_size = 0.1)
        print(len(train_split))
        print(len(val_split))
        print(len(test_split))
        print(len(data))

        self.train = train = []
        self.dev = dev = []
        self.test = test = []
        self.word2id = word2id
        
        for dat in data:
            label = np.array([data[dat]])
            
            img = np.load(f'{path_dir}/features/I3D_vecs/' + dat + "_rgb.npy") + np.load(f'{path_dir}/features/I3D_vecs/' + dat + "_flow.npy")
            #image.append(img)
            mask_img = mask_vector(26, img)

            aud = np.load(f'{path_dir}/features/vgg_vecs/' + dat + '_vggish.npy')
            #audio.append(aud)
            mask_aud = mask_vector(67, aud)
            
            _words = []
            txt = open(f'{path_dir}/features/TextoFiles/{dat}.txt', 'r', encoding = 'utf8')
            for line in txt.readlines():
                if line != '\n':
                    l = line.split('\n')
                    _words.extend(word_tokenize(l[0]))
            txt.close()

            actual_words = []
            words_ = []

            for idx, word in enumerate(_words):
                actual_words.append(word)
                words_.append(word2id[word])

            __words = np.asarray(words_)

            if dat in train_split:
                train.append(((__words, img, mask_img, aud, mask_aud, actual_words), label, None))
            elif dat in val_split:
                dev.append(((__words, img, mask_img, aud, mask_aud, actual_words), label, None))
            elif dat in test_split:
                test.append(((__words, img, mask_img, aud, mask_aud, actual_words), label, None))

        word2id.default_factory = return_unk

        """except:
            ValueError('Files does not exits in the root specified. Please verify it.')"""

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id
        elif mode == "val":
            return self.dev, self.word2id
        elif mode == "test":
            return self.test, self.word2id
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()