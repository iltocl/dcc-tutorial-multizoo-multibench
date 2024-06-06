import sys


sys.path.append('/home/ebaharlo/comic_mischief/Bert_Based_Model')
import torch
from torch import nn
from torch.nn import functional as F
#from allennlp.modules.elmo import Elmo, batch_to_ids
from source.models.attention import *
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

pp = pprint.PrettyPrinter(indent=4).pprint

debug = False
bidirectional = True

class Gated_modual(nn.Module):
    def __init__(self, dim):
        super(Gated_modual, self).__init__()

        self.dim = dim
        self.u = nn.Linear(2*dim, dim)
        #self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.W = nn.Parameter(torch.rand(2*dim,dim), requires_grad=True)

    def forward(self,image,text):
        x = torch.cat([image, text], 1)
        #print (x.shape)
        #h1 = self.tanh(image)
        #h2 = self.tanh(text)

        x = self.u(x)
        z = self.sigmoid(x)
        #print (z * image+ (1 -z)*text)
        #print ((z * image+ (1 -z)*text).shape) 
        return z * image+ (1 -z)*text


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

        self.att1 = BertOutAttention(self.embedding_dim)
        self.att2 = BertOutAttention(self.embedding_dim)
        self.att3 = BertOutAttention(self.embedding_dim)

        self.sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )

        self.sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )


        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True)


        self.gated = Gated_modual(768*2)

        # self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.rnn = nn.LSTM(self.embedding_dim, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        self.rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        self.rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)

        self.attention_audio = Attention(768)
        self.attention_image = Attention(768)
        self.attention = Attention(768)
        #self.attention_audio = Attention(256)
        #self.attention_image = Attention(256)
        #self.attention = Attention(256)

        self.sequential = nn.Sequential(
            nn.Linear(self.rnn_units, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),

            nn.Linear(100, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20)
        )

        self.img_audio_text_linear_mature = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

 
        self.img_audio_text_linear_gory = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

        self.img_audio_text_linear_slapstick = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

        self.img_audio_text_linear_sarcasm = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )





        self.audio_text = nn.Sequential(
            nn.Linear(768*2    , 768),
            #nn.BatchNorm1d(768),
            #nn.Dropout(0.3),
            
            #nn.Linear(100, 40),
            #nn.BatchNorm1d(40),
            #nn.Dropout(0.3),

            #nn.Linear(40, 20)
            
        )
        self.image_text = nn.Sequential(
            nn.Linear(768*2    , 768),
            #nn.BatchNorm1d(768),
            #nn.Dropout(0.3),

            #nn.Linear(100, 40),
            #nn.BatchNorm1d(40),
            #nn.Dropout(0.3),

            #nn.Linear(40, 20)

        )



        self.output1 = nn.Linear(20, 2)


    def forward(self, sentences,mask,image, image_mask, audio, audio_mask):
        
        hidden, _ = self.bert(sentences)[-2:]
        #cls_token = hidden[-1][:,0]
        #sentences = hidden[-1]
        #rnn_encoded, hid = self.rnn(sentences)
        #print (rnn_encoded,mask)
        #attention_applied, attention_weights = self.attention(rnn_encoded, mask.float())
        #print (attention_applied.shape)
        ##sequential_output = self.seq_cls(cls_token)
        ##sequential_output = self.sequential(attention_applied.float())
        #print (audio.shape,audio_mask.shape)
        rnn_img_encoded, (hid, ct) = self.rnn_img(image)
        #attention_applied_image, attention_weights = self.attention_image(rnn_img_encoded, image_mask.float())

        rnn_audio_encoded, (hid_audio, ct_audio) = self.rnn_audio(audio)
        #attention_applied_audio, attention_weights = self.attention_audio(rnn_audio_encoded, audio_mask.float())
        #print (hidden[-1].shape)
        #print (self.sequential_audio(rnn_audio_encoded).shape)
        #, self.sequential_audio(rnn_audio_encoded).shape)

        #audio_text  = torch.cat([attention_applied.float(),attention_applied_audio.float()], dim=-1)
        #print (attention_applied_audio.shape)
        #output = F.softmax(self.img_audio_text_linear(attention_applied.float()),-1)
        
         
        extended_attention_mask= mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
     
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
        
        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
        #print (hidden[-1].shape,self.sequential_audio(rnn_audio_encoded).shape)
        output_text = self.att1(hidden[-1], self.sequential_image(rnn_img_encoded) ,extended_image_attention_mask )
        output_text = self.att1(output_text, self.sequential_audio(rnn_audio_encoded) ,extended_audio_attention_mask )
        #print (self.sequential_image(rnn_img_encoded).shape)
        output_audio = self.att2(self.sequential_audio(rnn_audio_encoded), self.sequential_image(rnn_img_encoded), extended_image_attention_mask)
        output_audio = self.att2(output_audio, hidden[-1] ,extended_attention_mask )        

        output_image = self.att3(self.sequential_image(rnn_img_encoded), self.sequential_audio(rnn_audio_encoded), extended_audio_attention_mask)
        output_image = self.att3(output_image, hidden[-1] ,extended_attention_mask )
        #print (output_text.shape,output_image.shape,output_audio.shape)
       
        mask = torch.tensor(np.array([1]*output_text.size()[1])).cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).cuda()

        output_text, attention_weights = self.attention(output_text, mask.float())
        output_audio,  attention_weights = self.attention_audio(output_audio, audio_mask.float())
        output_image,  attention_weights = self.attention_image(output_image, image_mask.float())
        #print (output_text.shape,output_image.shape,output_audio.shape)
         
        #audio_text_vec  = torch.cat([output_text,output_audio], dim=-1)
        audio_text_image  = torch.cat([output_text,output_image,output_audio], dim=-1) 
        #audio_text_image = output_text
        #image_audio_text  = torch.cat([attention_applied.float(),attention_applied_audio.float()], dim=-1)
        #text_image = self.audio_text(text_image)
        #audio_text_image = torch.cat([text_image,output_audio], dim=-1)
        #audio_image_vec = self.image_text(image_text_vec)
        #audio_text_image = self.gated(audio_text_vec,image_text_vec)
        #print (audio_text_vec.shape, audio_image_vec.shape, audio_text_image.shape)
        #print (audio_text_image.shape) 
        #audio_text_image = self.gated(output_text,output_audio,output_image)
        output1 = F.softmax(self.img_audio_text_linear_mature(audio_text_image), -1)
        output2 = F.softmax(self.img_audio_text_linear_gory(audio_text_image), -1)
        output3 = F.softmax(self.img_audio_text_linear_slapstick(audio_text_image), -1)
        output4 = F.softmax(self.img_audio_text_linear_sarcasm(audio_text_image), -1)
        
        sequential_output = []
        #output = F.softmax(self.output1(sequential_output), -1)
        #print (attention_applied.shape)
        #print ([output1.shape,output2.shape,output3.shape,output4.shape])    
        return [output1,output2,output3,output4], sequential_output

"""
class GMU(nn.Module):
    def __init__(self, dim):
        super(GMU, self).__init__()

        self.dim = dim
        self.u = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.z = nn.Linear(dim, dim)
    
        self.sigmoid = nn.Sigmoid()
 
    def forward(self,image,text,audio):
      
        x1 = self.u(image)
        x2 = self.v(text)
        x3 = self.z(audio)

      
        z1 = self.sigmoid(x1)
        z2 = self.sigmoid(x2)
        z3 = self.sigmoid(x3)


        out = (z1)* image + (z2)* text + (z3)* audio
        return out
"""

### GMU ARNOLD ###
class GMU(nn.Module):
    def __init__(self, dim):
        super(GMU, self).__init__()

        self.dim = dim
        self.u = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.z = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, dim)
    
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
 
    def forward(self,image,text,audio):
      
        x1 = self.u(image)
        x2 = self.v(text)
        x3 = self.z(audio)
        print(f'valores de x_i {x1.size()}')

        h1 = self.tanh(x1)
        h2 = self.tanh(x2)
        h3 = self.tanh(x3)
        print(f'valores de h_i {h1.size()}')

        combined1 = torch.cat([x1, x2, x3], dim = 1)
        combined = self.w(combined1)
        print(f'valores de combined {combined.size()}')

        z1 = self.sigmoid(torch.matmul(combined, x1))
        z2 = self.sigmoid(torch.matmul(combined, x1))
        z3 = self.sigmoid(torch.matmul(combined, x1))
        print(f'valores de z_i {z1.size()}')

        #out = (z1)* image + (z2)* text + (z3)* audio
        out = torch.mul(z1, h1) + torch.mul(z2, h2) + torch.mul(z3, h3)
        print(f'valores de out {out.size()}')

        return out

class Bert_Model_GMU(nn.Module):

    def __init__(self):
        super(Bert_Model_GMU, self).__init__()

        self.rnn_units = 256
        # self.embedding_dim = 200
        # self.embedding_dim = 3072
        self.embedding_dim = 768

        self.att = BertOutAttention(self.embedding_dim)

        self.sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )

        self.sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units  , self.embedding_dim),
            #nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),

        )


        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True)




        # self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.rnn = nn.LSTM(self.embedding_dim, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        self.rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)
        self.rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=1, bidirectional=False, batch_first = True)

        #self.attention_audio = Attention(768)
        #self.attention_image = Attention(768)
        #self.attention = Attention(768)
        self.attention_audio = Attention(768)
        self.attention_image = Attention(768)
        self.attention = Attention(768)

        self.gated = GMU(768)
      

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
            nn.Linear(768  , 200),
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
        
        hidden, _ = self.bert(sentences)[-2:]
        #cls_token = hidden[-1][:,0]
        #sentences = hidden[-1]
        #rnn_encoded, hid = self.rnn(sentences)
        #print (rnn_encoded,mask)
        #attention_applied, attention_weights = self.attention(rnn_encoded, mask.float())
        #print (attention_applied.shape)
        ##sequential_output = self.seq_cls(cls_token)
        ##sequential_output = self.sequential(attention_applied.float())
        #print (audio.shape,audio_mask.shape)
        rnn_img_encoded, (hid, ct) = self.rnn_img(image)
        #attention_applied_image, attention_weights = self.attention_image(rnn_img_encoded, image_mask.float())

        rnn_audio_encoded, (hid_audio, ct_audio) = self.rnn_audio(audio)
        #attention_applied_audio, attention_weights = self.attention_audio(rnn_audio_encoded, audio_mask.float())
        #print (hidden[-1].shape)
        #print (self.sequential_audio(rnn_audio_encoded).shape)
        #, self.sequential_audio(rnn_audio_encoded).shape)

        #audio_text  = torch.cat([attention_applied.float(),attention_applied_audio.float()], dim=-1)
        #print (attention_applied_audio.shape)
        #output = F.softmax(self.img_audio_text_linear(audio_text),-1)
        
         
        extended_attention_mask= mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
     
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
      
        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
        #print (hidden[-1].shape,self.sequential_audio(rnn_audio_encoded).shape)
        output_text = self.att(hidden[-1], self.sequential_audio(rnn_audio_encoded),extended_audio_attention_mask)
        output_text = self.att(output_text, self.sequential_image(rnn_img_encoded) ,extended_image_attention_mask)
        #print (self.sequential_image(rnn_img_encoded).shape)

        output_audio = self.att(self.sequential_audio(rnn_audio_encoded), hidden[-1], extended_attention_mask)
        output_audio = self.att(output_audio, self.sequential_image(rnn_img_encoded) ,extended_image_attention_mask )        

        output_image = self.att(self.sequential_image(rnn_img_encoded), hidden[-1], extended_attention_mask)
        output_image = self.att(output_image, self.sequential_audio(rnn_audio_encoded) ,extended_audio_attention_mask )
        #print (output_text.shape,output_image.shape,output_audio.shape)
       
        mask = torch.tensor(np.array([1]*output_text.size()[1])).cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).cuda()

        output_text, attention_weights = self.attention(output_text, mask.float())
        output_audio,  attention_weights = self.attention_audio(output_audio, audio_mask.float())
        output_image,  attention_weights = self.attention_image(output_image, image_mask.float())
        #print (output_text.shape,output_image.shape,output_audio.shape)
        
        #audio_text_image  = torch.cat([output_text,output_audio,output_image], dim=-1)
        audio_text_image = self.gated(output_text,output_image,output_audio)
        #image_audio_text  = torch.cat([attention_applied.float(),attention_applied_audio.float(),attention_applied_image.float()], dim=-1)

        output = F.softmax(self.img_audio_text_linear(audio_text_image), -1)
        
        sequential_output = []
        #output = F.softmax(self.output1(sequential_output), -1)
        #print (attention_applied.shape)
        return output, sequential_output


"""
class Gated_modual(nn.Module):
    def __init__(self, dim):
        super(Gated_modual, self).__init__()
        
        self.dim = dim
        self.u = nn.Linear(dim, dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.W = nn.Parameter(torch.rand(2*dim,dim), requires_grad=True)

    def forward(self,image,text):
        x = torch.cat([image, text], 1)
               
        #h1 = self.tanh(image)
        #h2 = self.tanh(text)
      
        x = self.u(x)
        z = self.sigmoid(x)
        
        return z * image+ (1 -z)*text
"""
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



