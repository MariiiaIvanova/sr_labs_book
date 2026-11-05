#!/usr/bin/env python3
# run_training.py - Скрипт для обучения модели в фоновом режиме

import os
import sys
import numpy as np

# ВАЖНО: Исправление для numpy
np.float = float  # Временное решение для совместимости

# Добавляем пути
sys.path.append(os.path.realpath('..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Импорты из ваших модулей
from common import download_dataset, concatenate, extract_dataset, part_extract, download_protocol, split_musan
from exercises_blank import train_dataset_loader, test_dataset_loader, ResNet, MainModel, train_network, test_network
from ResNetBlocks import BasicBlock
from LossFunction import AAMSoftmaxLoss
from Optimizer import SGDOptimizer
from Scheduler import OneCycleLRScheduler
from load_save_pth import saveParameters, loadParameters

def main():
    # Путь к данным
    BASE_PATH = '/mnt/storage/work_dir/databases'
    
    # Гиперпараметры
    n_mels = 40
    log_input = True
    layers = [3, 4, 6, 3]
    activation = nn.ReLU
    num_filters = [32, 64, 128, 256]
    encoder_type = 'SP'
    nOut = 512
    margin = 0.35
    scale = 32.0
    
    max_frames_train = 200
    train_path = f'{BASE_PATH}/voxceleb1_dev/wav'
    batch_size_train = 128
    pin_memory = False
    num_workers_train = 5
    shuffle = True
    
    max_frames_val = 200 #1000
    val_path = f'{BASE_PATH}/voxceleb1_dev/wav'
    batch_size_val = 128
    num_workers_val = 5
    
    lr = 2.5
    weight_decay = 0
    val_interval = 5
    max_epoch = 40
    
    musan_path = f'{BASE_PATH}/musan_split'
    rir_path = f'{BASE_PATH}/RIRS_NOISES/simulated_rirs'
    
    # Генерация списков данных
    train_list = []
    val_list = []
    test_list = []
    
    with open(f'{BASE_PATH}/voxceleb1_test/iden_split.txt', 'r') as f:
        lines = f.readlines()
    
    black_list = os.listdir(f'{BASE_PATH}/voxceleb1_test')
    num_train_spk = []
    
    for line in lines:
        line = line.strip().split(' ')
        spk_id = line[1].split('/')[0]
        
        if not (spk_id in black_list):
            num_train_spk.append(spk_id)
        else:
            continue
        
        if line[0] == '1':
            train_list.append(' '.join([spk_id, line[1]]))
        elif line[0] == '2':
            val_list.append(' '.join([spk_id, line[1]]))
        elif line[0] == '3':
            test_list.append(' '.join([spk_id, line[1]]))
    
    num_train_spk = len(set(num_train_spk))
    
    # Инициализация модели
    model = ResNet(BasicBlock, layers=layers, activation=activation, 
                   num_filters=num_filters, nOut=nOut, encoder_type=encoder_type, 
                   n_mels=n_mels, log_input=log_input)
    trainfunc = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
    main_model = MainModel(model, trainfunc).cuda()
    
    # Инициализация загрузчиков данных
    train_dataset = train_dataset_loader(train_list=train_list, max_frames=max_frames_train, 
                                         train_path=train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, 
                             pin_memory=pin_memory, num_workers=num_workers_train, 
                             shuffle=shuffle)
    
    val_dataset = test_dataset_loader(test_list=val_list, max_frames=max_frames_val, 
                                      test_path=val_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, 
                           num_workers=num_workers_val)
    
    # Инициализация оптимизатора и планировщика
    optimizer = SGDOptimizer(main_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLRScheduler(optimizer, 
                                    pct_start=0.30, 
                                    cycle_momentum=False, 
                                    max_lr=lr, 
                                    div_factor=20, 
                                    final_div_factor=10000, 
                                    total_steps=max_epoch*len(train_loader))
    
    # Создаем папку для сохранения моделей
    os.makedirs('../data/lab3_models', exist_ok=True)
    
    # Обучение
    start_epoch = 0
    checkpoint_flag = False
    
    if checkpoint_flag:
        start_epoch = loadParameters(main_model, optimizer, scheduler, 
                                    path='../data/lab3_models/lab3_model_0004.pth')
        start_epoch = start_epoch + 1
    
    for num_epoch in range(start_epoch, max_epoch):
        train_loss, train_top1 = train_network(train_loader, main_model, optimizer, 
                                               scheduler, num_epoch, verbose=True)
        
        print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))
        
        if (num_epoch + 1) % val_interval == 0:
            _, val_top1 = test_network(val_loader, main_model)
            
            print("Epoch {:1.0f}, Accuracy (validation set) {:2.3f}%".format(num_epoch, val_top1))
            
            saveParameters(main_model, optimizer, scheduler, num_epoch, path='../data/lab3_models')

if __name__ == "__main__":
    main()