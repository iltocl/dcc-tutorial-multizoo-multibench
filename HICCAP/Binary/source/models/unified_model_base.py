import sys


sys.path.append('/storage/home/ebaharlo/comic_mischief/Original_Code/Bert_Based_Model/')
import torch
from torch import nn
from torch.nn import functional as F
#from allennlp.modules.elmo import Elmo, batch_to_ids
from source.models.attention_ORG import *
from source import config as C
import pprint
import numpy as np
from transformers import BertTokenizer, BertModel
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import random
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pp = pprint.PrettyPrinter(indent=4).pprint

debug = False
bidirectional = True
class BertOutAttention(nn.Module):
    def __init__(self, size, ctx_dim=None):
        super().__init__()
        if size % 12 != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (size, 12))
        self.num_attention_heads = 12
        self.attention_head_size = int(size / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =size
        self.query = nn.Linear(size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class Bert_Model(nn.Module):

    def __init__(self):
        super(Bert_Model, self).__init__()

        self.rnn_units = 256
        # self.embedding_dim = 200
        # self.embedding_dim = 3072
        self.embedding_dim = 768

        att1 = BertOutAttention(self.embedding_dim)
        att2 = BertOutAttention(self.embedding_dim)
        att3 = BertOutAttention(self.embedding_dim)

        sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )
        sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )
        bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True)


        rnn = nn.LSTM(self.embedding_dim, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)

        attention_audio = Attention(768)
        attention_image = Attention(768)
        attention = Attention(768)
        
        self.features = nn.ModuleList([bert, rnn_img, rnn_audio, att1, att2, att3, sequential_audio, sequential_image, attention, attention_audio, attention_image])
       
        self.sequential = nn.Sequential(
            nn.Linear(self.rnn_units, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),

            nn.Linear(100, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20)
        )

        self.img_audio_text_linear = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

 

        self.seq_cls = nn.Sequential(
            nn.Linear(768    , self.rnn_units),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),
            
            #nn.Linear(100, 40),
            #nn.BatchNorm1d(40),
            #nn.Dropout(0.3),

            #nn.Linear(40, 20)
            
        )



        self.output1 = nn.Linear(20, 2)


    def forward(self, sentences,mask,image, image_mask, audio, audio_mask):
        
        
        
        hidden, _ = self.features[0](sentences)[-2:]
        
        rnn_img_encoded, (hid, ct) = self.features[1](image)
        rnn_audio_encoded, (hid_audio, ct_audio) = self.features[2](audio)
        
        extended_attention_mask= mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
      
        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
 
        output_text = self.features[3](hidden[-1], self.features[6](rnn_audio_encoded) ,extended_audio_attention_mask )
        output_text = self.features[3](output_text, self.features[7](rnn_img_encoded) ,extended_image_attention_mask )

        output_audio = self.features[4](self.features[6](rnn_audio_encoded), hidden[-1], extended_attention_mask)
        output_audio = self.features[4](output_audio, self.features[7](rnn_img_encoded) ,extended_image_attention_mask )        

        output_image = self.features[5](self.features[7](rnn_img_encoded), hidden[-1], extended_attention_mask)
        output_image = self.features[5](output_image, self.features[6](rnn_audio_encoded) ,extended_audio_attention_mask )

       
        mask = torch.tensor(np.array([1]*output_text.size()[1])).cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).cuda()

        # print("output_text:", output_text.shape)
        # print("output_audio:", output_audio.shape)
        # print("output_image:", output_image.shape)

        output_text, attention_weights = self.features[8](output_text, mask.float())
        output_audio,  attention_weights = self.features[9](output_audio, audio_mask.float())
        output_image,  attention_weights = self.features[10](output_image, image_mask.float())

        audio_text_image  = torch.cat([output_text,output_audio,output_image], dim=-1)
        #image_audio_text  = torch.cat([cls_token, image_audio], dim=-1)

        output = F.softmax(self.img_audio_text_linear(audio_text_image), -1)
        
        sequential_output = []
        return output, sequential_output


class Bert_Model_CL(nn.Module):

    def __init__(self):
        super(Bert_Model_CL, self).__init__()

        self.rnn_units = 256
        # self.embedding_dim = 200
        # self.embedding_dim = 3072
        self.embedding_dim = 768

        att1 = BertOutAttention(self.embedding_dim)
        att2 = BertOutAttention(self.embedding_dim)
        att3 = BertOutAttention(self.embedding_dim)

        sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )
        sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )
        bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True)


        rnn = nn.LSTM(self.embedding_dim, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)

        attention_audio = Attention(768)
        attention_image = Attention(768)
        attention = Attention(768)
        
        self.features = nn.ModuleList([bert, rnn_img, rnn_audio, att1, att2, att3, sequential_audio, sequential_image, attention, attention_audio, attention_image])
       
        self.g_CL = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 400)
        )

        


    def forward(self, sentences,mask,image, image_mask, audio, audio_mask):
        
        hidden, _ = self.features[0](sentences)[-2:]
        
        rnn_img_encoded, (hid, ct) = self.features[1](image)
        rnn_audio_encoded, (hid_audio, ct_audio) = self.features[2](audio)
        
        extended_attention_mask= mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
      
        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
 
        output_text = self.features[3](hidden[-1], self.features[6](rnn_audio_encoded) ,extended_audio_attention_mask )
        output_text = self.features[3](output_text, self.features[7](rnn_img_encoded) ,extended_image_attention_mask )

        output_audio = self.features[4](self.features[6](rnn_audio_encoded), hidden[-1], extended_attention_mask)
        output_audio = self.features[4](output_audio, self.features[7](rnn_img_encoded) ,extended_image_attention_mask )        

        output_image = self.features[5](self.features[7](rnn_img_encoded), hidden[-1], extended_attention_mask)
        output_image = self.features[5](output_image, self.features[6](rnn_audio_encoded) ,extended_audio_attention_mask )

       
        mask = torch.tensor(np.array([1]*output_text.size()[1])).cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).cuda()

        # print("output_text:", output_text.shape)
        # print("output_audio:", output_audio.shape)
        # print("output_image:", output_image.shape)

        output_text, attention_weights = self.features[8](output_text, mask.float())
        output_audio,  attention_weights = self.features[9](output_audio, audio_mask.float())
        output_image,  attention_weights = self.features[10](output_image, image_mask.float())

        output_text = self.g_CL(output_text)
        output_audio = self.g_CL(output_audio)
        output_image = self.g_CL(output_image)
        return output_text, output_audio, output_image

class Bert_Model_Pretraining(nn.Module):

    def __init__(self):
        super(Bert_Model_Pretraining, self).__init__()

        self.rnn_units = 256
        # self.embedding_dim = 200
        # self.embedding_dim = 3072
        self.embedding_dim = 768

        att1 = BertOutAttention(self.embedding_dim)
        att2 = BertOutAttention(self.embedding_dim)
        att3 = BertOutAttention(self.embedding_dim)

        sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )
        sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )
        bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True)


        rnn = nn.LSTM(self.embedding_dim, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)

        attention_audio = Attention(768)
        attention_image = Attention(768)
        attention = Attention(768)
        
        self.features = nn.ModuleList([bert, rnn_img, rnn_audio, att1, att2, att3, sequential_audio, sequential_image, attention, attention_audio, attention_image])
       
        self.TAM_linear = nn.Sequential(
            nn.Linear(768*2  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )


        self.ITM_linear = nn.Sequential(
            nn.Linear(768*2  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

        self.IAM_linear = nn.Sequential(
            nn.Linear(768*2  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )


    def forward(self, sentences,mask,image, image_mask, audio, audio_mask, mode = 'pre_train'):
        
        
        hidden, _ = self.features[0](sentences)[-2:]
        
        rnn_img_encoded, (hid, ct) = self.features[1](image)
        rnn_audio_encoded, (hid_audio, ct_audio) = self.features[2](audio)
        
        extended_attention_mask= mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
      
        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
 
        output_text = self.features[3](hidden[-1], self.features[6](rnn_audio_encoded) ,extended_audio_attention_mask )
        output_text = self.features[3](output_text, self.features[7](rnn_img_encoded) ,extended_image_attention_mask )

        output_audio = self.features[4](self.features[6](rnn_audio_encoded), hidden[-1], extended_attention_mask)
        output_audio = self.features[4](output_audio, self.features[7](rnn_img_encoded) ,extended_image_attention_mask )        

        output_image = self.features[5](self.features[7](rnn_img_encoded), hidden[-1], extended_attention_mask)
        output_image = self.features[5](output_image, self.features[6](rnn_audio_encoded) ,extended_audio_attention_mask )

       
        mask = torch.tensor(np.array([1]*output_text.size()[1])).cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).cuda()

        # print("output_text:", output_text.shape)
        # print("output_audio:", output_audio.shape)
        # print("output_image:", output_image.shape)

        output_text, attention_weights = self.features[8](output_text, mask.float())
        output_audio,  attention_weights = self.features[9](output_audio, audio_mask.float())
        output_image,  attention_weights = self.features[10](output_image, image_mask.float())

        batch_image = output_image.clone()
        batch_audio = output_audio.clone()
        batch_text = output_text.clone()

        #label Image-Text Matching
        label_ITM = torch.zeros((output_image.shape[0], 2), device=device)
        #label Image-Audio Matching
        label_IAM = torch.zeros((output_image.shape[0], 2), device=device)
        #label Text-Audio Matching
        label_TAM = torch.zeros((output_image.shape[0], 2), device=device)

        for i in range(output_image.shape[0]):
          sel_mod = torch.randint(0, 3, (1,))
          # print("sel_mod :", sel_mod)
          #creates one number out of 0 or 1 with prob p 0.4 for 0 and 0.6 for 1
          flag_replace = np.random.choice(np.arange(0, 2), p=[0.4, 0.6])
          # print("flag_replace", flag_replace)
          if flag_replace == 1:
            allowed_values = list(range(0, output_image.shape[0]))
            allowed_values.remove(i)
            random_value = random.choice(allowed_values)
            # print("random_value", random_value)

            if sel_mod == 0:
              batch_image[i] = output_image[random_value]
              label_ITM[i] = torch.Tensor([1,0])
              label_IAM[i] = torch.Tensor([1,0])
              label_TAM[i] = torch.Tensor([0,1])

            if sel_mod == 1:
              batch_audio[i] = output_audio[random_value]
              label_ITM[i] = torch.Tensor([0,1])
              label_IAM[i] = torch.Tensor([1,0])
              label_TAM[i] = torch.Tensor([1,0])

            if sel_mod == 2:
              batch_text[i] = output_text[random_value]
              label_ITM[i] = torch.Tensor([1,0])
              label_IAM[i] = torch.Tensor([0,1])
              label_TAM[i] = torch.Tensor([1,0])
          
          else:
              label_ITM[i] = torch.Tensor([0,1])
              label_IAM[i] = torch.Tensor([0,1])
              label_TAM[i] = torch.Tensor([0,1])

        TAM_input = torch.cat([batch_text,batch_audio], dim=-1)
        output_TAM = F.softmax(self.TAM_linear(TAM_input), -1)

        IAM_input = torch.cat([batch_image,batch_audio], dim=-1)
        output_IAM = F.softmax(self.IAM_linear(IAM_input), -1)

        ITM_input = torch.cat([batch_image,batch_text], dim=-1)
        output_ITM = F.softmax(self.ITM_linear(ITM_input), -1)

        
        batch_image = None
        batch_audio = None
        batch_text = None
        
        return label_TAM, label_IAM, label_ITM, output_TAM, output_IAM, output_ITM

class Gated_modual(nn.Module):
    def __init__(self, dim):
        super(Gated_modual, self).__init__()
        
        self.dim = dim
        self.u = nn.Linear(2*dim, dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.W = nn.Parameter(torch.rand(2*dim,dim), requires_grad=True)

    def forward(self,image,text):
        x = torch.cat([image, text], 1)
        
        h1 = self.tanh(image)
        h2 = self.tanh(text)

        x = self.u(x)
        z = self.sigmoid(x)
        
        return z * image+ (1 -z)*text

class audio_LSTM(nn.Module):

    def __init__(self):
        super(audio_LSTM, self).__init__()

        
        self.rnn = nn.LSTM(40, 256, num_layers=1, bidirectional=False)
        #self.rnn_pros = nn.LSTM(3, 256, num_layers=1, bidirectional=False)
        
        #self.sequential_feat = nn.Sequential(
        #    nn.Linear(385, 100),
        #    nn.ReLU(),
        #    nn.BatchNorm1d(100),
        #    nn.Dropout(0.1),

           

        #)

       
        self.sequential = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.1),

            nn.Linear(100, 2),
            nn.ReLU(),
            nn.BatchNorm1d(2),
            

        )
        self.attention = Attention(256)
        #self.attention_fet = Attention(256)

    def forward(self, audio,features, mask):

        #print (features.shape)
        #print (audio.shape)
        #print (features)
        #print (features.reshape( (features.shape[0],20, 3),axis = '0'))
        rnn_out, _ = self.rnn(audio)
        attention_applied, attention_weights = self.attention(rnn_out, mask.float())        

        #rnn_out_fet, _ = self.rnn_pros(features)
        #attention_applied_feature, attention_weights_fet = self.attention_fet(rnn_out_fet, mask.float())

        #densed_feat = self.sequential_feat(features)
        #cats  = torch.cat([attention_applied, attention_applied_feature], dim=-1)
        sequential_output = self.sequential(attention_applied)
        output = F.sigmoid(sequential_output)
        return output, attention_applied




class Text_audio_unified(nn.Module):

    def __init__(self):
        super(Text_audio_unified, self).__init__()


        self.rnnAudio = nn.LSTM(40, 256, num_layers=1, bidirectional=False)
        self.attentionAudio = Attention(256)

        self.rnn_units = 256
        self.embedding_dim = 768

        self.sequential_DeepMoji = nn.Sequential(
            nn.Linear(2304  , 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.4)

        )


        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True)

        self.rnnText = nn.LSTM(self.embedding_dim, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)

        self.attention = Attention(self.rnn_units)

        self.sequentialText = nn.Sequential(
            nn.Linear(self.rnn_units+256  , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )


        self.gated = Gated_modual(256) 

        self.sequential_gated = nn.Sequential(
            nn.Linear(256  , 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.2),

            nn.Linear(100  , 2),
        )
        

    def forward(self,text, audio, deepMoji, mask, mask_audio):


        rnn_audio_out, _ = self.rnnAudio(audio)
        attention_applied_audio, attention_weights_audio = self.attention(rnn_audio_out, mask_audio.float())

        emoji = deepMoji.view(deepMoji.size(0), -1)
        hidden, _ = self.bert(text)[-2:]
        sentences = hidden[-1]
        rnn_encoded_text, hid = self.rnnText(sentences)
        attention_applied, attention_weights = self.attention(rnn_encoded_text, mask.float())
        sequential_emoji = self.sequential_DeepMoji(emoji)
        final_representation = torch.cat([attention_applied.float(), sequential_emoji.float()], 1)
        textShrinked = self.sequentialText(final_representation)

        gt = self.gated(textShrinked,attention_applied_audio)
        out = self.sequential_gated (gt)
        
        output = F.sigmoid(out)        

        return output, textShrinked



