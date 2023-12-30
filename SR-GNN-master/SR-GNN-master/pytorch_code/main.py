#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle # 파이썬에서 사용하는 딕셔너리, 리스트, 클래스 등의 자료형을 변환 없이 그대로 파일로 저장하고 이를 불러올 때 사용하는 모듈
## pickle은 객체 자체를 저장하고 불러옴 pickle.dump(객체, 파일) : 저장. pickle.load(파일)로 로딩
import time
from utils import build_graph, Data, split_validation # utils에 저장되어있는 함수들 불러오기
from model import *

parser = argparse.ArgumentParser()  # 인자 추가시키기 위해 객체 생성  
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
# 인자 추가할 때 --dataset과 같이 앞에 -를 하나 혹은 두개 붙여주면 옵션이라는 의미임. 즉 입력하지 않아도 되는 인자라는 뜻임 
## 하이픈의 수는 인자명이 한문자인 경우에는 1개, 2개 이상인 경우에는 두개를 붙임
### help는 이 인자를 설명해줌 
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# type에는 함수도 가능함. lambda와 map을 이용해서 type에 함수를 부여해주면 어떤 값을 넣었을때 원하는 모습으로 인자를 설정해줌 
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
# action을 사용하면 플래그를 지정해줄 수 있음 store_true 는 True, store_false 는 False
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')

opt = parser.parse_args()
# .parse_args() 실행시 add_argument에 추가한 인자 순서대로 실행됨.

print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb')) # rb: 바이너리 모드로 읽어옴
    if opt.validation: # 따로 지정안해주면 True임
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica': # 분석에 쓰이는 데이터셋 
        n_node = 43098  # 아이템 개수 
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node)) # trans_to_cuda : model 모듈에 정의되어있는 함수  SessionGraph: model모듈에 정의된 class

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data) # model 모듈에 정의되어 있는 함수
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
