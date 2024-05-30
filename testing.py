
from data_loader import get_loader
import sys

dataset = 'UR_FUNNY'
batch_size = 16
len0 = 8074 # MOSI: 1283 - MOSEI: 16327 - UR-FUNNY: 8074 - MUSTARD: 412

"""train_data, len1 = get_loader(data = 'CoMic', data_dir = '...', mode = 'train', task = 'Binary', batch_size = batch_size, 
                                                          seq_l = 768, words = True, shuffle = True)

val_data, len2 = get_loader('CoMic', 'Binary/processed_data', mode = 'val', task = 'Binary', batch_size = batch_size, 
                                                          seq_l = 768, words = True, shuffle = True)"""

'''test_data, len3 = get_loader('CoMic', sys.path[0], mode = 'test', task = '---', batch_size = batch_size, seq_length = 128,
                             words = False, shuffle = False)'''

test_data, len3 = get_loader('Hate', 'hsdv_batch_250', mode = 'test', task = 'Binary', batch_size = batch_size, 
                                                          seq_length = 128, words = True, shuffle = False)

def train(split):
    """
    Trains the model using the optimizer for a single epoch.
    :param model: pytorch model
    :param optimizer:
    :return:
    """

    batch_idx = 1
    total_loss = 0

    idx = 0

    for batch in test_data:

        #t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask, masked_audio, masked_img = batch
        t, v, a, y = batch

        batch_x = t
        batch_image = v
        batch_audio = a
        y = y
        """l = l
        bert_sent = bert_sent
        bert_sent_type = bert_sent_type
        bert_sent_mask = bert_sent_mask"""

        #print(f'shape text: {t}')
        #print(f'shape text: {bert_sent}')
        print(f'\nshape text: {t.shape}')
        print(f'shape image: {v.shape}')
        print(f'shape audio: {a.shape}')
        print(f'labels: {y}')
        print(f'labels: {y.shape}\n')

        '''if idx == 1:
            break
        idx += 1
'''
train(test_data)