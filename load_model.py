# Load the model in fairseq
from fairseq.data.encoders.fastbpe import fastBPE

import os
import time
import random
import argparse
import pickle
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn
from sklearn.model_selection import StratifiedKFold

# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from transformers.modeling_utils import * 
from transformers import *

# Khởi tạo argument
EPOCHS = 20
BATCH_SIZE = 6
ACCUMULATION_STEPS = 5
FOLD = 4
LR = 0.0001
LR_DC_STEP = 80 
LR_DC = 0.1
CUR_DIR = os.path.dirname(os.getcwd())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOLD = 4
CKPT_PATH2 = 'model_ckpt2'

if not os.path.exists(CKPT_PATH2):
    os.mkdir(CKPT_PATH2)

# Khởi tạo DataLoader
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X, y))

for fold, (train_idx, val_idx) in enumerate(splits):
    best_score = 0
    if fold != FOLD:
        continue
    print("Training for fold {}".format(fold))
    
    # Create dataset
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X[train_idx],dtype=torch.long), torch.tensor(y[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X[val_idx],dtype=torch.long), torch.tensor(y[val_idx],dtype=torch.long))

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo model:
    MODEL_LAST_CKPT = os.path.join(CKPT_PATH2, 'latest_checkpoint.pth.tar')
    if os.path.exists(MODEL_LAST_CKPT):
      print('Load checkpoint model!')
      phoBERT = torch.load(MODEL_LAST_CKPT)
    else:
      print('Load model pretrained!')
      # Load the model in fairseq
      from fairseq.models.roberta import RobertaModel
      from fairseq.data.encoders.fastbpe import fastBPE
      from fairseq.data import Dictionary

      phoBERT = RobertaModel.from_pretrained('PhoBERT_base_fairseq', checkpoint_file='model.pt')
      phoBERT.eval()  # disable dropout (or leave in train mode to finetune

      # # Load BPE
      # class BPE():
      #   bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'

      # args = BPE()
      # phoBERT_cls.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

      # Add header cho classification với số lượng classes = 10
      phoBERT.register_classification_head('new_task', num_classes=10)
      
    ## Load BPE
    print('Load BPE')
    class BPE():
      bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'

    args = BPE()
    phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT
    phoBERT.to(DEVICE)

    # Khởi tạo optimizer và scheduler, criteria
    print('Init Optimizer, scheduler, criteria')
    param_optimizer = list(phoBERT.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(EPOCHS*len(train_dataset)/BATCH_SIZE/ACCUMULATION_STEPS)
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # scheduler với linear warmup
    scheduler0 = get_constant_schedule(optimizer)  # scheduler với hằng số
    # optimizer = optim.Adam(phoBERT_cls.parameters(), LR)
    criteria = nn.NLLLoss()
    # scheduler = StepLR(optimizer, step_size = LR_DC_STEP, gamma = LR_DC)
    avg_loss = 0.
    avg_accuracy = 0.
    frozen = True
    for epoch in tqdm(range(EPOCHS)):
        # warm up tại epoch đầu tiên, sau epoch đầu sẽ phá băng các layers
        if epoch > 0 and frozen:
            for child in phoBERT.children():
                for param in child.parameters():
                    param.requires_grad = True
            frozen = False
            del scheduler0
            torch.cuda.empty_cache()
        # Train model on EPOCH
        print('Epoch: ', epoch)
        trainOnEpoch(train_loader=train_loader, model=phoBERT, optimizer=optimizer, epoch=epoch, num_epochs=EPOCHS, criteria=criteria, device=DEVICE, log_aggr=100)
        # scheduler.step(epoch = epoch)
        # Phá băng layers sau epoch đầu tiên
        if not frozen:
            scheduler.step()
        else:
            scheduler0.step()
        optimizer.zero_grad()
        # Validate on validation set
        acc, f1 = validate(valid_loader, phoBERT, device=DEVICE)
        print('Epoch {} validation: acc: {:.4f}, f1: {:.4f} \n'.format(epoch, acc, f1))

        # Store best model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': phoBERT.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # Save model checkpoint into 'latest_checkpoint.pth.tar'
        torch.save(ckpt_dict, MODEL_LAST_CKPT)
# Load BPE
class BPE():
  bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'

args = BPE()
phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

# Add header cho classification với số lượng classes = 10
phoBERT.register_classification_head('new_task', num_classes=10)
tokens = 'Học_sinh được nghỉ học bắt đầu từ tháng 3 do ảnh hưởng của dịch covid-19'
token_idxs = phoBERT.encode(tokens)
logprobs = phoBERT.predict('new_task', token_idxs)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
logprobs