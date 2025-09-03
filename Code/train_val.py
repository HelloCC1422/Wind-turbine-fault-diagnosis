# -*- encoding: utf-8 -*-
'''
  @Author   : changchen
  @Date     : 2024/12/25 22:43
  @PROJECT  : Programb
  @File     : train_val.py    
  @Describe :
'''
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from Model import Transformer_F, Transformer_H
import time
import os
from tensorboardX import SummaryWriter

class Config():
    data_path = '../Data/farm_ai4i/P_data/'
    model_path = '../Data/farm_ai4i/model/model_ai4i'
    epochs = 50  # 迭代轮数
    model_name1 = 'Transformer_H'  # 模型名称
    model_name2 = 'Transformer_F'  # 模型名称
    save_path1 = '{}/{}.pth'.format(model_path, model_name1)  # 最优模型保存路径
    save_path2 = '{}/{}.pth'.format(model_path, model_name2)  # 最优模型保存路径

def datasets(data_path):
    # 加载数据并打乱
    df_train = pd.read_csv(data_path+'device_train.csv', encoding='gbk').sample(frac=1, random_state=10)
    df_test = pd.read_csv(data_path+'device_test.csv', encoding='gbk').sample(frac=1, random_state=10)

    test_label = df_test['fault_number'].values

    # 数据归一化
    scaler = StandardScaler()
    df_train.iloc[:, 1:] = scaler.fit_transform(df_train.iloc[:, 1:].values)
    df_test.iloc[:, 1:] = scaler.transform(df_test.iloc[:, 1:].values)

    # 训练集
    # 多分类
    df_train2 = df_train.loc[df_train['fault_number'] > 0]
    data_train2 = df_train2.iloc[:, 1:].values
    train_label2 = df_train2['fault_number'].values-1
    # print(np.unique(train_label2))

    # 二分类
    data_train = df_train.iloc[:, 1:].values
    train_label1 = df_train['fault_number'].values
    train_label1[train_label1 > 0] = 1
    # print(np.unique(train_label1))

    # 测试集
    # 多分类
    df_test2 = df_test.loc[df_test['fault_number'] > 0]
    data_test2 = df_test2.iloc[:, 1:].values
    test_label2 = df_test2['fault_number'].values-1
    # print(np.unique(test_label2))

    # 二分类
    data_test = df_test.iloc[:, 1:].values
    test_label1 = df_test['fault_number'].values
    test_label1[test_label1 > 0] = 1
    # print(np.unique(test_label1))
    # print(data_train2.shape)

    # 一级分类
    train_data1 = torch.utils.data.TensorDataset(torch.Tensor(data_train), torch.Tensor(train_label1))
    test_data1 = torch.utils.data.TensorDataset(torch.Tensor(data_test), torch.Tensor(test_label1))

    train_loader1 = torch.utils.data.DataLoader(train_data1, batch_size=1, shuffle=True)
    test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=True)

    # 二级分类
    train_data2 = torch.utils.data.TensorDataset(torch.Tensor(data_train2), torch.Tensor(train_label2))
    test_data2 = torch.utils.data.TensorDataset(torch.Tensor(data_test2), torch.Tensor(test_label2))

    train_loader2 = torch.utils.data.DataLoader(train_data2, batch_size=1, shuffle=True)
    test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=True)

    # 测试集
    test_data = df_test.iloc[:, 1:].values
    # print(np.unique(test_label))

    test_data = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    return train_loader1, test_loader1, train_loader2, test_loader2, test_loader

def train_val(train_loader, test_loader, model, criterion, optimizer, scheduler, model_path, save_path, epochs, phase):
    train_time = 0
    best_acc = 0
    best_epoch = 0
    best_loss = 1

    writer = SummaryWriter(log_dir=model_path+'/logs/'+time.strftime('%m-%d_%H.%M', time.localtime())+'_'+phase)

    # 故障分类
    for epoch in range(epochs):
        time_start = time.time()
        train_acc = 0.0
        train_loss = 0.0

        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for param in model.parameters():
            param.requires_grad = True
        train_num = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs = torch.sigmoid(outputs)

            loss = criterion(outputs, labels.long())
            pred1 = torch.tensor(outputs.argmax(dim=1), dtype=torch.int)

            # print('labels', int(labels.item()))
            # print(pred1.item())

            correct = torch.eq(pred1, labels).float().sum().item()

            loss_temp = loss.item() * inputs.size(0)
            train_loss += loss_temp
            train_acc += correct

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                info = 'Train_Epoch: [{}][{}/{}]\tLoss:{:.5f}'.format(epoch+1, batch_idx, train_num, loss.item())
                print(info)

        # logging the train and val information via each epoch
        train_loss = train_loss / train_num
        train_acc = train_acc / train_num

        train_time += time.time() - time_start

        # info = 'Train{}_Epoch: [{}/{}]\tLR:{}, Loss:{:.4f}, Acc: {:.4f}, Time {:.4f} sec'.format(
        #     phase, epoch+1, epochs, optimizer.param_groups[0]['lr'],
        #     train_loss, train_acc, train_time)
        # print(info)
        print('-'*80)

        # 验证模型
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        val_num = len(test_loader)

        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            pred2 = torch.tensor(outputs.argmax(dim=1), dtype=torch.int)

            correct = torch.eq(pred2, labels).float().sum().item()
            loss_temp = loss.item() * inputs.size(0)
            val_loss += loss_temp
            val_acc += correct

        # logging the train and val information via each epoch
        val_loss = val_loss / val_num
        val_acc = val_acc / val_num

        info = 'Val{}_Epoch: [{}/{}]\tLR:{}, Loss:{:.4f}, Acc: {:.4f}'.format(
            phase, epoch+1, epochs, optimizer.param_groups[0]['lr'], val_loss, val_acc)
        print(info)

        if epoch >= epochs // 3 and (val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)):
            best_acc = val_acc
            best_epoch = epoch
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print('Best val Acc: {:4f}\tBest val loss: {:4f}'.format(best_acc,best_loss))
            print("=> Save Best")

        writer.add_scalar("time/train", train_time, epoch)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("loss/valid", val_loss, epoch)
        writer.add_scalar("accuracy/valid", val_acc, epoch)
        writer.add_scalar("epoch/best_epoch", best_epoch, epoch)
        writer.add_scalar("accuracy/best_acc", best_acc, epoch)
        writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()
    writer.close()



if __name__ == "__main__":
    args = Config()
    warnings.filterwarnings("ignore")
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dict_model = [args.model_path, args.save_path1, args.save_path2]

    if not os.path.exists(dict_model[0]):
        os.mkdir(dict_model[0])

    train_loader1, test_loader1, train_loader2, test_loader2, test_loader = datasets(args.data_path)
    print("----------------------数据读取完成------------------------")

    # 初始化模型和优化器
    model1 = Transformer_H(input_size=6, num_classes=2).to(device)  # 一级分类模型Transformer
    model2 = Transformer_F(input_size=6, num_classes=5).to(device)  # 二级分类模型

    optimizer1 = optim.AdamW(model1.parameters(), lr=1e-3)
    optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=15, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=15, gamma=0.1)

    criterion1 = nn.CrossEntropyLoss()  # 交叉熵损失函数
    criterion2 = nn.CrossEntropyLoss()  # 交叉熵损失函数
    train_val(train_loader1, test_loader1, model1, criterion1, optimizer1, scheduler1, dict_model[0], dict_model[1], args.epochs, '1')
    train_val(train_loader2, test_loader2, model2, criterion2, optimizer2, scheduler2, dict_model[0], dict_model[2], args.epochs, '2')
    print("----------------------模型训练完成------------------------")

