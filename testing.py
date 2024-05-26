
from data_loader import get_loader

batch_size = 1

"""train_data, len1 = get_loader(data = 'CoMic', data_dir = 'Binary/processed_data', mode = 'train', task = 'Binary', batch_size = batch_size, 
                                                          seq_l = 768, words = True, shuffle = True)

val_data, len2 = get_loader('CoMic', 'Binary/processed_data', mode = 'val', task = 'Binary', batch_size = batch_size, 
                                                          seq_l = 768, words = True, shuffle = True)"""

"""test_data, len3 = get_loader('CoMic', 'Binary/processed_data', mode = 'test', task = 'Binary', batch_size = batch_size, 
                                                          seq_l = 768, words = False, shuffle = False)"""

test_data, len3 = get_loader('Hate', 'hsdv_batch_250', mode = 'test', task = 'Binary', batch_size = batch_size, 
                                                          seq_l = 768, words = True, shuffle = False)

def train():
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

        t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask, masked_audio, masked_img = batch

        batch_x = t
        batch_image = v
        batch_audio = a
        y = y
        l = l
        bert_sent = bert_sent
        bert_sent_type = bert_sent_type
        bert_sent_mask = bert_sent_mask

        print(f'shape text: {t}')
        print(f'shape text: {bert_sent}')
        print(f'shape image: {v.shape}')
        print(f'shape audio: {a.shape}')
        print(f'labels: {y}\n')

        if idx == 1:
            break
        idx += 1

train()
