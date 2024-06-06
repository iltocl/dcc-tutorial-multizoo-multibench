import os
import sys

import random
import torch
sys.path.append('/home/ebaharlo/comic_mischief/Bert_Based_Model/source')

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

from torch import nn
import time
import json
import numpy as np
import pandas as pd
from random import shuffle
from torch import optim
from torch.nn import functional as F
from experiments import utils as U
from experiments.TorchHelper import TorchHelper
from experiments.tf_logger import Logger
#from experiments.dataloader import *
from models.unified_model_multi_task import * #### ======================> LISTO
import config as C #### ======================> LISTO
from sklearn.metrics import f1_score
import warnings
from sklearn.metrics import confusion_matrix
from pytorch_pretrained_bert import BertAdam
from transformers import AdamW
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

torch_helper = TorchHelper()


loss_weights1 = torch.Tensor([1,3])


run_mode = 'run'
#run_mode = 'resume'
#run_mode = 'test_resume'

criterian = nn.CrossEntropyLoss()
# exp_mode =

start_epoch = 0
# plot_interval = 5
# limit_movies_for_rank_plot = 500
# test_data_limit = 100
batch_size = C.batch_size

max_epochs = 40
learning_rate = 0.00002
clip_grad = 0.5
weight_decay_val = 0
optimizer_type = 'adam'  # sgd

# if run_mode == 'test' or run_mode == 'test_resume':
#     max_epochs = 25
#     plot_interval = 1
#     batch_size = 8
# if 'resume' in run_mode:
#     start_epoch = 2

collect_attention = True
run_multitask = False

l2_regularize = True
l2_lambda = 0.1

# Learning rate scheduler
lr_schedule_active = False
reduce_on_plateau_lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau

# Creates the directory where the results, logs, and models will be dumped.
run_name = 'Text_Image_Audio_cross_Multitask_captionAddedi_sarcasm/'
run_name = 'Multi_Task'
description = ''

output_dir_path = '/home/ebaharlo/comic_mischief/Bert_Based_Model/results/'+ run_name + '/'
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)
#logger = Logger(output_dir_path + 'logs')

# Files to keep backup
backup_file_list = ['/home/ebaharlo/comic_mischief/Bert_Based_Model/source/experiments/train_combination.py',
                    '/home/ebaharlo/comic_mischief/Bert_Based_Model/source/models/unified_model_multi_task_org.py']

path_res_out = os.path.join(output_dir_path, 'res_ORG_'+run_name+'_'+str(learning_rate)+'.out')
f = open(path_res_out, "a")
f.write('-------------------------\n')
f.close()

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------
# Load Data using the data generator
root_data_path = '/home/ebaharlo/comic_mischief/Bert_Based_Model/data/'

# Load Partition Information

features_dict_train = json.load(open(C.training_features_bert_multi))
features_dict_val = json.load(open(C.val_features_bert_multi))
features_dict_test = json.load(open(C.test_features_bert_multi))

#features_dict_test.update(features_dict_test)
#features_dict_test.update(features_dict_val)
#features_dict_test.update(features_dict_train)


train_set = features_dict_train
print (len(train_set))
print('Train Loaded')

validation_set = features_dict_val
print (len(validation_set))
print('Validation Loaded')

#total_id_list = train_id_list +val_id_list+test_id_list
test_set = features_dict_test
print (len(test_set))
print('test Loaded')

#===========================================================================================================




# ------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def create_model():
    """
    Creates and returns the EmotionFlowModel.
    Moves to GPU if found any.
    :return:

    """

    model =  Bert_Model()
    if run_mode == 'resume' or run_mode == 'test_resume' or run_mode == 'test':
        model1 =  Bert_Model_CL()
        #model1 = Bert_Model_Pretraining()
        
        checkpoint_path = "/storage/home/ebaharlo/comic_mischief/Original_Code/Bert_Based_Model/results/Contrastive_Loss_Lava/"#Pretraining_Kinetics
        torch_helper.load_saved_model(model1, checkpoint_path + 'best_contrastive_loss.pth') #last_pretrain_All.pth
        print('model loaded')
    
    
        model.features = model1.features
    model.cuda()
    return model


def compute_l2_reg_val(model):
    if not l2_regularize:
        return 0.

    l2_reg = None

    for w in model.parameters():
        if l2_reg is None:
            l2_reg = w.norm(2)
        else:
            l2_reg = l2_reg + w.norm(2)

    return l2_lambda * l2_reg.item()


from torch.utils.data import TensorDataset, DataLoader
def masking(docs_ints, seq_length=500):

    # getting the correct rows x cols shape
    masks = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        #mask[i, :len(row)] = 1
        masks[i, -len(row):] = 1

    return masks

def mask_vector(max_size,arr):
    # print (arr,arr.shape)
    if arr.shape[0] > max_size:
       output = [1]*max_size
    else:
       len_zero_value = max_size -  arr.shape[0]
       output = [1]*arr.shape[0] + [0]*len_zero_value
    
    return np.array(output)

def pad_segment(feature, max_feature_len, pad_idx):
    S, D = feature.shape
    #print (S, D)
    try:
       pad_l =  max_feature_len - S
       # pad
       pad_segment = np.zeros((pad_l, D))
       feature = np.concatenate((feature,pad_segment), axis=0)
       #print (feature.shape)
    except:
       feature = feature[0:max_feature_len]
       #print (feature.shape)
    return feature

def pad_features(docs_ints, seq_length=500):

    # getting the correct rows x cols shape
    features = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features
# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, optimizer):
    """
    Trains the model using the optimizer for a single epoch.
    :param model: pytorch model
    :param optimizer:
    :return:
    """

    start_time = time.time()

    model.train()

    batch_idx = 1
    total_loss = 0
    batch_x = []
    batch_image, batch_mask_img = [],[]
    batch_audio, batch_mask_audio = [],[]
    batch_emo_deepMoji = []
    batch_mask = []
    batch_y, batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], [],[]
    batch_text = []
    train_imdb = []
    sh_train_set = train_set

    log_vars = nn.Parameter(torch.zeros((4)))
    #random.Random(2).shuffle(sh_train_set)

    for index, i in enumerate(sh_train_set):
        #list(np.int_(batch_x))
        mid = sh_train_set[i]['IMDBid']

        path = "/home/ebaharlo/comic_mischief/BMT/i3D_vecs/"
        #image_vec = np.load("./deepMoji_out/"+mid+".npy")
        a1 = np.load(path+mid+"_rgb.npy")
        a2 = np.load(path+mid+"_flow.npy")
        a = a1+a2
        masked_img = mask_vector(36,a)
        a = pad_segment(a, 36, 0)
        image_vec = a
        #masked_img = mask_vector(36,a)

        path = "/home/ebaharlo/comic_mischief/BMT/vgg_vecs/"
        try:
           audio_arr = np.load(path+mid+"_vggish.npy")
        except:
           audio_arr = np.array([128*[0.0]])
        masked_audio = mask_vector(63,audio_arr)
        #print (masked_audio)
        audio_vec = pad_segment(audio_arr, 63, 0)
        batch_audio.append(audio_vec)
        batch_mask_audio.append(masked_audio)

        train_imdb.append(mid)
        batch_x.append(np.array(sh_train_set[i]['indexes']))
        batch_mask_img.append(masked_img)
        batch_image.append(image_vec)
        batch_mature.append(sh_train_set[i]['mature'])
        batch_gory.append(sh_train_set[i]['gory'])
        batch_slapstick.append(sh_train_set[i]['slapstick'])
        batch_sarcasm.append(sh_train_set[i]['sarcasm'])

        if (len(batch_x) == batch_size or index == len(sh_train_set) - 1 ) and len(batch_x)>0:

            optimizer.zero_grad()

            mask = masking(batch_x)
            batch_x = pad_features(batch_x)
            batch_x = np.array(batch_x)
            batch_x = torch.tensor(batch_x).cuda()

            batch_image = np.array(batch_image)
            batch_image = torch.tensor(batch_image).cuda()

            batch_mask_img = np.array(batch_mask_img)
            batch_mask_img = torch.tensor(batch_mask_img).cuda()

            batch_audio = np.array(batch_audio)
            batch_audio = torch.tensor(batch_audio).cuda()

            batch_mask_audio = np.array(batch_mask_audio)
            batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

            #batch_emo_deepMoji = np.array(batch_emo_deepMoji)
            #batch_emo_deepMoji = torch.tensor(batch_emo_deepMoji).cuda()

            out, mid = model(batch_x, torch.tensor(mask).cuda(),batch_image.float(),batch_mask_img, batch_audio.float(),batch_mask_audio)


            mature_pred = out[0].cpu()
            gory_pred = out[1].cpu()
            slapstick_pred = out[2].cpu()
            sarcasm_pred = out[3].cpu()
            #print (mature_pred,gory_pred,slapstick_pred,sarcasm_pred)
            loss1 = compute_l2_reg_val(model) + F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature)) 
            loss2 = compute_l2_reg_val(model) + F.binary_cross_entropy(gory_pred, torch.Tensor(batch_gory))
            loss3 = compute_l2_reg_val(model) + F.binary_cross_entropy(slapstick_pred, torch.Tensor(batch_slapstick))
            loss4 = compute_l2_reg_val(model) + F.binary_cross_entropy(sarcasm_pred, torch.Tensor(batch_sarcasm))
            #loss2 = compute_l2_reg_val(model) + F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y))

            """
            precision1 = torch.exp(-log_vars[0])
            loss1 = precision1*loss1 + log_vars[0]

            precision2 = torch.exp(-log_vars[1])
            loss2 = precision2*loss2 + log_vars[1]

            precision3 = torch.exp(-log_vars[2])
            loss3 = precision3*loss3 + log_vars[2]

            precision4 = torch.exp(-log_vars[3])
            loss4 = precision4*loss4 + log_vars[3]

            """
            total_loss += loss1.item() + loss2.item() + loss3.item() + loss4.item()
            loss = loss1 + loss2 + loss3 + loss4
            #print (loss)
            loss.backward()


            optimizer.step()

            torch_helper.show_progress(batch_idx, np.ceil(len(sh_train_set) / batch_size), start_time,
                                       round(total_loss / (index + 1), 4))
            batch_idx += 1
            batch_x, batch_y,batch_image,batch_mask_img = [], [], [],[]
            batch_audio, batch_mask_audio = [],[]
            batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []



    return model

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        c = 0
        for j in range(len(y_true[i])):
            if y_true[i][j] == y_pred[i][j]:
               c += 1
        acc_list.append (c/4)
        #set_true = set( np.where(y_true[i])[0] )
        #set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        #tmp_a = None
        #if len(set_true) == 0 and len(set_pred) == 0:
        #    tmp_a = 1
        #else:
        #    tmp_a = len(set_true.intersection(set_pred))/\
        #            float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        #acc_list.append(tmp_a)
    return np.mean(acc_list)


# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, dataset):
    model.eval()

    total_loss = 0
    total_loss1, total_loss2, total_loss3 = 0, 0, 0

    batch_x, batch_y1,batch_image,batch_mask_img = [], [],[],[]
    batch_director = []
    batch_genre = []
    y1_true, y2_true, y3_true = [], [], []
    imdb_ids = []
    predictions = [[], [], []]
    id_to_vec = {}
    batch_audio, batch_mask_audio = [],[]
    #batch_audio, batch_mask_audio = [],[]
    batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []
    mature_true, gory_true, sarcasm_true, slapstick_true = [], [], [], []
    batch_text = []
    predictions_mature, predictions_gory, predictions_slapstick, predictions_sarcasm = [], [], [], []
    with torch.no_grad():
        for index,i in enumerate(dataset):
            mid = dataset[i]['IMDBid']
            if mid == "tt1723121.03":
               print ("=========", dataset[i]["words"])
            #emoji_vec = np.load("./deepMoji_out/"+mid+".npy")
            #mid = dataset[i]['IMDBid']
            imdb_ids.append(mid)
            batch_x.append(np.array(dataset[i]['indexes']))
            #batch_emo.append(np.array(dataset[i]['emo']))
            #batch_emo_deepMoji.append(emoji_vec)
            #batch_y1.append(dataset[i]['y'])
            #y1_true.append(C.label_to_idx[dataset[i]['label']])

            path = "/home/ebaharlo/comic_mischief/BMT/i3D_vecs/"
            #image_vec = np.load("./deepMoji_out/"+mid+".npy")
            a1 = np.load(path+mid+"_rgb.npy")
            a2 = np.load(path+mid+"_flow.npy")
            a = a1+a2
            masked_img = mask_vector(36,a)
            a = pad_segment(a, 36, 0)
            image_vec = a
            batch_image.append(image_vec)
            #masked_img = mask_vector(36,a)
            batch_mask_img .append(masked_img)


            path = "/home/ebaharlo/comic_mischief/BMT/vgg_vecs/"
            try:
                audio_arr = np.load(path+mid+"_vggish.npy")
            except:
                audio_arr = np.array([128*[0.0]])

            #audio_arr = np.load(path+mid+"_vggish.npy")
            masked_audio = mask_vector(63,audio_arr)
            #print (masked_audio)
            audio_vec = pad_segment(audio_arr, 63, 0)
            batch_audio.append(audio_vec)
            batch_mask_audio.append(masked_audio)

            batch_mature.append(dataset[i]['mature'])
            batch_gory.append(dataset[i]['gory'])
            batch_slapstick.append(dataset[i]['slapstick'])
            batch_sarcasm.append(dataset[i]['sarcasm'])

            mature_true.append(np.argmax(np.array(dataset[i]['mature'])))
            gory_true.append(np.argmax(np.array(dataset[i]['gory'])))
            sarcasm_true.append(np.argmax(np.array(dataset[i]['sarcasm'])))
            slapstick_true.append(np.argmax(np.array(dataset[i]['slapstick'])))


            if (len(batch_x) == batch_size or index == len(dataset) - 1) and len(batch_x)>0:

                mask = masking(batch_x)

                #print (mask)
                batch_x = pad_features(batch_x)
                batch_x = np.array(batch_x)
                batch_x = torch.tensor(batch_x).cuda()

                batch_image = np.array(batch_image)
                batch_image = torch.tensor(batch_image).cuda()

                batch_mask_img = np.array(batch_mask_img )
                batch_mask_img = torch.tensor(batch_mask_img ).cuda()

                batch_audio = np.array(batch_audio)
                batch_audio = torch.tensor(batch_audio).cuda()
 
                batch_mask_audio = np.array(batch_mask_audio)
                batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

                out, mid_level_out = model(batch_x, torch.tensor(mask).cuda(),batch_image.float(),batch_mask_img,batch_audio.float(),batch_mask_audio)

                #mature_pred = out[0].cpu()
                mature_pred = out[0].cpu()
                gory_pred = out[1].cpu()
                slapstick_pred = out[2].cpu()
                sarcasm_pred = out[3].cpu()

                #mid_level_out = mid_level_out.cpu()
                """ 
                for idx, item in enumerate(mid_level_out):
                        mid = imdb_ids[idx]
                        #print (y_pred1[idx].shape) 
                        np.save("text_features_NEWASR/fold5/"+mid+".npy",mid_level_out[idx])  
                        #np.save("late_features/text_features/fold5/"+mid+".npy",y_pred1[idx])               
                """

                predictions_mature.extend(list(torch.argmax(mature_pred, -1).numpy()))
                predictions_gory.extend(list(torch.argmax(gory_pred, -1).numpy()))
                predictions_slapstick.extend(list(torch.argmax(slapstick_pred, -1).numpy()))
                predictions_sarcasm.extend(list(torch.argmax(sarcasm_pred, -1).numpy()))

                loss2 = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature))
                # _, labels1 = torch.Tensor(batch_y1).max(dim=1)

                total_loss1 += loss2.item()

                batch_x, batch_y1,batch_image,batch_mask_img  = [], [], [],[]
                batch_director = []
                batch_genre = []
                batch_mask = []
                batch_text = []
                batch_similar = []
                batch_description = []
                imdb_ids = []
                batch_audio, batch_mask_audio = [],[]
                batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []

    #print (len(y1_true))
    #print (len(predictions[0]))
    """
    s1,s2,t1,t2 = 0,0,0,0

    for idx, j in enumerate(y1_true):
        if j == 0:
           t1 += 1
           if predictions[0][idx] == 0:
              s1 += 1
        else:
           t2 += 1
           if predictions[0][idx] == 1:
              s2 += 1
    print ("scores ===> ", s1/t1, s2/t2)   
    metric = ((s1/t1) + (s2/t2))/2.0
    """
    true_values = []
    preds = []
    from sklearn.metrics import hamming_loss
    for i in range(len(mature_true)):
         true_values.append([mature_true[i], gory_true[i], slapstick_true[i], sarcasm_true[i]])
         preds.append([predictions_mature[i],predictions_gory[i],predictions_slapstick[i],predictions_sarcasm[i]])
         #print ([mature_true[i], gory_true[i], slapstick_true[i], sarcasm_true[i]], [predictions_mature[i],predictions_gory[i],predictions_slapstick[i],predictions_sarcasm[i]])
    #print (true_values)
    #print (preds)
    print ("acc_score ",accuracy_score(true_values,preds))
    print ("Hamin_score",hamming_score(np.array(true_values),np.array( preds)))
    print("Hamming_loss:", hamming_loss(true_values, preds))
    print (hamming_loss(true_values, preds) + hamming_score(np.array(true_values),np.array( preds)))
    from sklearn.metrics import classification_report
    print (classification_report(true_values,preds))
    micro1 = f1_score(mature_true, predictions_mature, average='macro')
    micro2 = f1_score(gory_true, predictions_gory, average='macro')
    micro3 = f1_score(slapstick_true, predictions_slapstick, average='macro')
    micro4 = f1_score(sarcasm_true, predictions_sarcasm, average='macro')
    
    f = open(path_res_out, "a")
    
    print ("Mature")
    print (confusion_matrix(mature_true, predictions_mature))
    print('weighted', f1_score(mature_true, predictions_mature, average='weighted'))
    print('micro', f1_score(mature_true, predictions_mature, average='micro'))
    print('macro', f1_score(mature_true, predictions_mature, average='macro'))
    print ("============================")
      
    f.write ("Mature\n")
    f.write('weighted: %f\n' % f1_score(mature_true, predictions_mature, average='weighted'))
    f.write('micro: %f\n' % f1_score(mature_true, predictions_mature, average='micro'))
    f.write('macro: %f\n' % f1_score(mature_true, predictions_mature, average='macro'))
    f.write ("============================\n")
    
    print ("Gory")
    print (confusion_matrix(gory_true, predictions_gory))
    print('weighted', f1_score(gory_true, predictions_gory, average='weighted'))
    print('micro', f1_score(gory_true, predictions_gory, average='micro'))
    print('macro', f1_score(gory_true, predictions_gory, average='macro'))
    print ("=============================")

    f.write ("Gory\n")
    f.write('weighted: %f\n' % f1_score(gory_true, predictions_gory, average='weighted'))
    f.write('micro: %f\n' % f1_score(gory_true, predictions_gory, average='micro'))
    f.write('macro: %f\n' % f1_score(gory_true, predictions_gory, average='macro'))
    f.write ("============================\n")
    
    print ("Slapstick")
    print (confusion_matrix(slapstick_true, predictions_slapstick))
    print('weighted', f1_score(slapstick_true, predictions_slapstick, average='weighted'))
    print('micro', f1_score(slapstick_true, predictions_slapstick, average='micro'))
    print('macro', f1_score(slapstick_true, predictions_slapstick, average='macro'))
    print ("=============================")

    f.write ("Slapstick\n")
    f.write('weighted: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='weighted'))
    f.write('micro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='micro'))
    f.write('macro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='macro'))
    f.write ("============================\n")
   
    print ("Sarcasm")
    print (confusion_matrix(sarcasm_true, predictions_sarcasm))
    print('weighted', f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
    print('micro', f1_score(sarcasm_true, predictions_sarcasm, average='micro'))
    print('macro', f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
    print ("=============================")

    f.write ("Sarcasm\n")
    f.write('weighted: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
    f.write('micro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='micro'))
    f.write('macro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
    f.write ("============================\n")
  
    metric = (micro1 + micro2 + micro3 + micro4)/4
    print ("Metric:", metric)

    f.write('acc_score: %f\n' % accuracy_score(true_values,preds))
    f.write('Hamin_score: %f\n'% hamming_score(np.array(true_values),np.array( preds)))
    f.write('Hamming_loss: %f\n'% hamming_loss(true_values, preds))
    f.close()

    return predictions, \
           total_loss1 / len(dataset), \
           metric
    # attn_weights,\
    # total_loss2/len(dataset), \
    # total_loss3/len(dataset), \
    # total_loss/len(dataset), \

    # [micro_f1_1, micro_f1_2, micro_f1_3]


def training_loop():
    """

    :return:
    """
    model = create_model()
    """
    optimizer = AdamW(model.parameters(),
                  lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    total_steps = (
            500
            / batch_size
            
            * 50
        )
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
    optimizer = BertAdam(
    optimizer_grouped_parameters,
    lr=learning_rate,
    #warmup=0.1, 
    #t_total=total_steps,
    )
    """
    
    if optimizer_type == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay_val)
   
    lr_scheduler = reduce_on_plateau_lr_schdlr(optimizer, 'max', min_lr=1e-8, patience=2, factor=0.5)

    max_test_f1 = 0
    for epoch in range(start_epoch, max_epochs):
        print('[Epoch %d] / %d : %s' % (epoch + 1, max_epochs, run_name))
        f = open(path_res_out, "a")
        f.write('[Epoch %d] / %d : %s' % (epoch + 1, max_epochs, run_name))


        # train model
        """
        if epoch == 115:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        """
        model = train(model, optimizer)

        val_pred, val_loss1, val_f1 = evaluate(model, validation_set)
        #test_pred, test_loss1, test_f1 = evaluate(model, test_set)

        if max_test_f1 < val_f1:
            max_test_f1 = val_f1

        current_lr = 0
        for pg in optimizer.param_groups:
            current_lr = pg['lr']

        print('Validation Loss %.5f, Validation F1 %.5f' % (val_loss1, val_f1))
        #print('Test Loss %.5f, Test F1 %.5f, Max Test F1 %.5f' % (test_loss1, test_f1, max_test_f1))

        print('Learning Rate', current_lr)

        if lr_schedule_active:
            lr_scheduler.step(val_f1)

        is_best = torch_helper.checkpoint_model(model, optimizer, output_dir_path, val_f1, epoch + 1,
                                                'max')


        f.write('Validation Loss %.5f, Validation F1 %.5f\n' % (val_loss1, val_f1))
        #f.write('Test Loss %.5f, Test F1 %.5f\n' % (test_loss1, test_f1))
        f.write('Learning Rate: %f\n' % (current_lr))
        f.close()
        

        print()

        # -------------------------------------------------------------
        # Tensorboard Logging
        # -------------------------------------------------------------
        info = {#'training loss': train_loss1,
                'validation loss': val_loss1,
                # 'train_loss1' : train_loss1,
                # 'val_loss1'   : val_loss1,
                # 'train_loss2': train_loss1,
                # 'val_loss2': val_loss2,
                # 'train_loss3'  : train_loss1,
                # 'val_loss3'  : val_loss3,

                #'train_f1_1': train_f1,
                'val_f1_1': val_f1,
                # 'train_f1_2': train_f1[1],
                # 'val_f1_2'  : val_f1[1],
                # 'train_f1_3': train_f1[2],
                # 'val_f1_3'  : val_f1[2],

                'lr': current_lr
                }
        """
        for tag, value in info.items():
            logger.log_scalar(tag, value, epoch + 1)
   
        # Log values and gradients of the model parameters
        for tag, value in model.named_parameters():
            if value.grad is not None:
                tag = tag.replace('.', '/')

                if torch.cuda.is_available():
                    logger.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                    logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
                else:
                    logger.log_histogram(tag, value.data.numpy(), epoch + 1)
                    logger.log_histogram(tag + '/grad', value.grad.data.numpy(), epoch + 1)
        """

def test():
    model = create_model()

    val_pred, val_loss1, val_f1 = evaluate(model, test_set)
    print('Validation Loss %.5f, Validation F1 %.5f' % (val_loss1, val_f1))


if __name__ == '__main__':
    #"""i
    if run_mode != 'test':
        U.copy_files(backup_file_list, output_dir_path)
        with open(output_dir_path + 'description.txt', 'w') as f:
                f.write(description)
                f.close()

        training_loop()
    #"""
    else:
        test()

    """
    U.copy_files(backup_file_list, output_dir_path)
    with open(output_dir_path + 'description.txt', 'w') as f:
        f.write(description)
        f.close()

    training_loop()
    """
    #
    #test()

    # x, y = collect_train()
    # x1, y1 = collect_val()
    # x= np.array(x)
    # x1 = np.array(x1)
    # x = np.reshape(x,(x.shape[0],x.shape[1],1))
    # x1 = np.reshape(x1, (x1.shape[0], x1.shape[1], 1))
    # model = cnn(4)
    # print (x.shape)
    # # print (y.shape)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(np.array(x), np.array(y), validation_data=(np.array(x1), np.array(y1)), epochs=400)



    # from sklearn import svm
    #
    # clf = svm.SVC()
    # clf.fit(x, y)
    #
    # p = clf.predict(x1)
    # print (f1_score(y1,p, average='weighted'))
