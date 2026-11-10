import os
import sys
import argparse
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
from ResNetBlocksMHC import BasicBlockMHC, mhc_summary                   # [mHC] блок с mHC вместо BasicBlock
from LossFunction import AAMSoftmaxLoss
from Optimizer import SGDOptimizer
from Scheduler import OneCycleLRScheduler
from load_save_pth import saveParameters, loadParameters


def train_network_accum(train_loader, main_model, optimizer, scheduler, num_epoch, accum_steps, verbose=False):
    # [mHC] Запасной вариант на случай нехватки памяти GPU: батч из DataLoader (128) делится на
    # accum_steps частей, градиенты накапливаются, шаг оптимизатора и планировщика делается один раз
    # на батч. Эффективный размер батча и число шагов OneCycle остаются как в базовом запуске.
    # Отличие от базы только в статистиках BatchNorm (считаются по части батча).
    # При accum_steps = 1 эта функция не используется (вызывается ваш train_network).

    assert scheduler[1] in ['epoch', 'iteration']

    main_model.train()

    loss    = 0
    top1    = 0
    counter = 0

    for data, data_label in train_loader:

        optimizer.zero_grad()

        batch_loss = 0
        batch_top1 = 0

        for data_part, label_part in zip(torch.chunk(data, accum_steps, dim=0), torch.chunk(data_label, accum_steps, dim=0)):

            data_part  = data_part.transpose(1, 0).cuda()
            label_part = label_part.cuda()

            nloss, prec1 = main_model(data_part, label_part)

            # Вес части пропорционален её размеру, чтобы градиент совпадал со средним по всему батчу
            weight = label_part.size(0)/data_label.size(0)
            (nloss*weight).backward()

            batch_loss += nloss.detach().cpu().item()*weight
            batch_top1 += prec1.detach().cpu().item()*weight

        optimizer.step()

        loss += batch_loss
        top1 += batch_top1
        counter += 1

        if verbose:
            print("Epoch {:1.0f}, Batch {:1.0f}, LR {:f} Loss {:f}, Accuracy {:2.3f}%".format(num_epoch, counter, optimizer.param_groups[0]['lr'], loss/counter, top1/counter))

        if scheduler[1] == 'iteration': scheduler[0].step()

    if scheduler[1] == 'epoch': scheduler[0].step()

    return (loss/counter, top1/counter)


def main():
    # [mHC] Аргументы командной строки: продолжение с чекпоинта и накопление градиента
    parser = argparse.ArgumentParser(description='Lab 3: ResNet + mHC (Jin et al., Interspeech 2026), training with augmentation')
    parser.add_argument('--resume', type=str, default='', help='путь к чекпоинту для продолжения обучения, например ../data/lab3_models_aug_mhc/lab3_model_0019.pth')
    parser.add_argument('--accum_steps', type=int, default=1, help='на сколько частей делить батч 128 при нехватке памяти (1 = как в базе)')
    args = parser.parse_args()

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
    num_workers_train = 4
    shuffle = True
    
    max_frames_val = 1000
    val_path = f'{BASE_PATH}/voxceleb1_dev/wav'
    batch_size_val = 128
    num_workers_val = 4
    
    lr = 2.5
    weight_decay = 0
    val_interval = 5
    max_epoch = 40
    
    musan_path = f'{BASE_PATH}/musan_split'
    rir_path = f'{BASE_PATH}/RIRS_NOISES/simulated_rirs'

    # [mHC] Отдельная папка для чекпоинтов, чтобы не перезаписать базовую модель
    models_path = '../data/lab3_models_aug_mhc'
    
    # Генерация списков данных
    train_list = []
    val_list = []
    test_list = []
    
    with open(f'{BASE_PATH}/voxceleb1_test/iden_split.txt', 'r') as f:
        lines = f.readlines()
    
    black_list = set(os.listdir(f'{BASE_PATH}/voxceleb1_test/wav'))
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
    model = ResNet(BasicBlockMHC, layers=layers, activation=activation,      # [mHC] BasicBlock -> BasicBlockMHC
                   num_filters=num_filters, nOut=nOut, encoder_type=encoder_type, 
                   n_mels=n_mels, log_input=log_input)
    trainfunc = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
    main_model = MainModel(model, trainfunc).cuda()

    # [mHC] Контроль: число параметров и начальные матрицы смешивания
    print('Number of model parameters: {:d}'.format(sum(p.numel() for p in model.parameters())))
    print('Number of mHC parameters:   {:d}'.format(sum(p.numel() for n, p in model.named_parameters() if n.endswith('mhc.theta'))))
    print(mhc_summary(model))
    
    # Инициализация загрузчиков данных С АУГМЕНТАЦИЕЙ
    train_dataset = train_dataset_loader(train_list=train_list, 
                                         max_frames=max_frames_train, 
                                         train_path=train_path,
                                         augment=True,           
                                         musan_path=musan_path,  
                                         rir_path=rir_path)      
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
    
    # Создаем папку для сохранения моделей (с аугментацией и mHC)
    os.makedirs(models_path, exist_ok=True)
    
    # Обучение
    start_epoch = 0
    
    if args.resume:
        start_epoch = loadParameters(main_model, optimizer, scheduler, path=args.resume)
        start_epoch = start_epoch + 1
        print('Resumed from {:s}, starting epoch {:d}'.format(args.resume, start_epoch))
    
    for num_epoch in range(start_epoch, max_epoch):
        if args.accum_steps == 1:
            train_loss, train_top1 = train_network(train_loader, main_model, optimizer, 
                                                   scheduler, num_epoch, verbose=True)
        else:
            train_loss, train_top1 = train_network_accum(train_loader, main_model, optimizer, 
                                                         scheduler, num_epoch, args.accum_steps, verbose=True)
        
        print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))
        
        if (num_epoch + 1) % val_interval == 0:
            _, val_top1 = test_network(val_loader, main_model)
            
            print("Epoch {:1.0f}, Accuracy (validation set) {:2.3f}%".format(num_epoch, val_top1))
            print(mhc_summary(model))
            
            saveParameters(main_model, optimizer, scheduler, num_epoch, path=models_path)

if __name__ == "__main__":
    main()