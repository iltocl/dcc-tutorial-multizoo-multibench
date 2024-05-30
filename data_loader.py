import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from create_dataset import CoMic, Hate, PAD

run_on = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(run_on)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer_es = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

def masking(docs_ints, seq_length = 768):

    # getting the correct rows x cols shape
    masks = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        #mask[i, :len(row)] = 1
        masks[i, -len(row):] = 1

    return masks

class MSADataset(Dataset):
    def __init__(self, data, data_dir, mode, task:str, words:bool, seq_length:int = 768):

        ## Fetch dataset
        if "comic" == str(data).lower():
            dataset = CoMic(data_dir, task, words)
            self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        elif 'hate' == str(data).lower():
            dataset = Hate(data_dir, task, words)
            self.bert = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', output_hidden_states=True, output_attentions=True)
        else:
            print("Dataset not defined correctly")
            exit()

        '''if "mosi" in str(data_dir).lower():
            dataset = MOSI(sdk_dir, data_dir)
        elif "mosei" in str(data_dir).lower():
            dataset = MOSEI(sdk_dir, data_dir)
        elif "ur_funny" in str(data_dir).lower():
            dataset = UR_FUNNY(data_dir)'''
        
        
        self.data, self.word2id = dataset.get_data(mode)
        self.len1 = len(self.data)

        #self.visual_size = self.data[0][0][1].shape[1]
        #self.acoustic_size = self.data[0][0][2].shape[1]
        self.embedding_size = 300
        hidden_size = 128
        output_size = seq_length
        dp = 0.1

        self.word2id = self.word2id

        self.rnn_audio = nn.LSTM(128, hidden_size, num_layers=1, bidirectional=False, batch_first = True)
        self.rnn_img = nn.LSTM(1024, hidden_size, num_layers=1, bidirectional=False, batch_first = True)

        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Dropout(dp)
        )

        self.mlp_audio = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Dropout(dp)
        )

        self.mlp_text = nn.Sequential(
            nn.Linear(768, output_size),
            nn.Dropout(dp)
        )


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len1



def get_loader(data, data_dir, mode, task:str = 'Binary', batch_size:int = 16, seq_length:int = 768, words:bool = True, shuffle:bool = False):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(data, data_dir, mode, task.lower(), words, seq_length)
    
    print(mode)
    data_len = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        """s = lambda x: len(x)#[0][0].shape[0]
        for b in batch:
            print(s(b))"""
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        #labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        #labels = torch.from_numpy(batch[0][1])
        labels = np.array([sample[0][6] for sample in batch])
        if task.lower() == 'binary':
            labels = torch.tensor(labels)
        else:
            labels = torch.tensor(labels).permute(1, 0, 2)
        
        """if words:
            sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        else:
            sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch])"""
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD).permute(1, 0)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch]).permute(1, 0, 2)
        acoustic = pad_sequence([torch.FloatTensor(sample[0][3]) for sample in batch]).permute(1, 0, 2)
        mask_audio = pad_sequence([torch.FloatTensor(sample[0][4]) for sample in batch]).permute(1, 0)
        mask_image = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch]).permute(1, 0)

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer
        bert_details = []
        for sample in batch:
            if data.lower() == 'comic':
                if words:
                    text = " ".join(sample[0][5])
                    encoded_bert_sent = bert_tokenizer.encode_plus(text, max_length = SENT_LEN + 2, add_special_tokens = True,
                                                                   padding = 'max_length', truncation = True)
                    bert_details.append(encoded_bert_sent)
                else:
                    bert_details.append(sample[0][5])
            else:
                #print('sent', len(sample[0][3]))
                text = " ".join(sample[0][5])
                encoded_bert_sent = bert_tokenizer_es.encode_plus(
                    text, max_length=SENT_LEN+2, add_special_tokens=True, padding='max_length', truncation = True)
                bert_details.append(encoded_bert_sent)

        # Bert things are batch_first
        if not words:
            bert_sentences = sentences #torch.Tensor([sample for sample in bert_details])
            #print('\nsentences', bert_sentences.shape)
            bert_sentence_types = torch.zeros(bert_sentences.size(), dtype = torch.long)
            #bert_sentence_att_mask = torch.where(bert_sentences != 0, 1, 0) #torch.ones(16, 300)
            bert_sentence_att_mask = torch.where(bert_sentences == 1, 1, 0) #masking(sentences.transpose(1, 0)) #torch.ones(16, 300)

        else:
            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
            #print('\nsentences', bert_sentences.shape)
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])


        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        hidden, _ = dataset.bert(input_ids = bert_sentences, attention_mask = bert_sentence_att_mask, token_type_ids = bert_sentence_types)[-2:]
        img_encoded, (hid, ct) = dataset.rnn_img(visual)
        audio_encoded, (hid_audio, ct_audio) = dataset.rnn_audio(acoustic)

        txt = torch.nan_to_num(dataset.mlp_text(hidden[-1]), nan = 0.5)
        audio = torch.nan_to_num(dataset.mlp_audio(audio_encoded), nan = 0.5)
        image = torch.nan_to_num(dataset.mlp_img(img_encoded), nan = 0.5)

        #return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, mask_audio, mask_image
        return txt, audio, image, labels


    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        collate_fn = collate_fn)

    return data_loader, data_len