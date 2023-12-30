
import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'C:/Users/ydb80/Desktop/GNN/추천_논문/SR_GNN_study/SR-GNN-master/SR-GNN-master/datasets/sample_train-item-views.csv'  

print("-- Starting @ %ss" % datetime.datetime.now())


with open(dataset, "r") as f: # mode='r' : 읽기용으로 파일을 엶
    if True :
        reader = csv.DictReader(f, delimiter=';')      

    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader: # reader가 딕셔너리 형태이므로 data는 reader에 존재하는 key가 됨; 이 for문 돌리는 이유
       
        # print(data)
        sessid = data['session_id'] # 해당 row의 sessionid 값 부여 
#         print ('sessid:', sessid)
        if curdate and not curid == sessid: # curid랑 세션아이디랑 같지 않고 curdate는 True이면 
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d')) # g
            sess_date[curid] = date # 세션 데이트 dict형태로 넣어줌
            # print('sess_date[curid]: ' , sess_date[curid])
#             print()
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe']) # timeframe 아이템 클릭 시간인듯? 
        # print('item:',item)

#         print()
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks: # sess_clicks에 sessionid가 존재하면 추가를 해주고 
            sess_clicks[sessid] += [item]
        else:  # 존재하지 않으면 새롭게 session id 를 넣어주라는 의미
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):  # sess_clicks는 dict이므로 list()해주면 dict의 key값들 출력. 즉 session id값 출력
            # print(i)
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            # operator.itemgetter : 다양한 기준으로 정렬하기 위해 사용함. sess_clicks를 정렬하는데, 1번째 key의 value로 정렬한다는 의미 
            ## 즉 , sesseion id가 같은 아이템들을 아이템 id 기준으로 정렬한다는 의미 
            sess_clicks[i] = [c[0] for c in sorted_clicks] # 정렬한 값 순서대로 sess_clicks의 해당 세션id의 value를 리스트 형태로 넣어줌
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())



# with open(dataset, "r") as f: # mode='r' : 읽기용으로 파일을 엶
#     if opt.dataset == 'yoochoose': # 데이터셋에 따라 구분자 달라짐
#         reader = csv.DictReader(f, delimiter=',')  # csv.DictReader : 파일의 각 row를 딕셔너리 형태로 읽어옴 
#     else:
#         reader = csv.DictReader(f, delimiter=';')
#     sess_clicks = {}
#     sess_date = {}
#     ctr = 0
#     curid = -1
#     curdate = None
#     for data in reader: # reader가 딕셔너리 형태이므로 data는 reader에 존재하는 key가 됨; 이 for문 돌리는 이유?
#         print(data)
#         sessid = data['session_id'] # 해당 row의 sessionid 값 부여 
#         print ('sessid:', sessid)
#         if curdate and not curid == sessid: # curid랑 세션아이디랑 같지 않고 curdate는 True이면 
#             date = ''
#             if opt.dataset == 'yoochoose':
#                 date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
#             else:
#                 date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
#             sess_date[curid] = date # 세션 데이트 dict형태로 넣어줌
#             print('sess_date[curid]: ' , sess_date[curid])
#             print()
#         curid = sessid
#         if opt.dataset == 'yoochoose':
#             item = data['item_id']
#         else:
#             item = data['item_id'], int(data['timeframe'])
#         print('item:',item)
#         print()
#         curdate = ''
#         if opt.dataset == 'yoochoose':
#             curdate = data['timestamp']
#         else:
#             curdate = data['eventdate']

#         if sessid in sess_clicks:
#             sess_clicks[sessid] += [item]
#         else:
#             sess_clicks[sessid] = [item]
#         ctr += 1
#     date = ''
#     if opt.dataset == 'yoochoose':
#         date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
#     else:
#         date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
#         for i in list(sess_clicks):
#             sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
#             sess_clicks[i] = [c[0] for c in sorted_clicks]
#     sess_date[curid] = date
# print("-- Reading data @ %ss" % datetime.datetime.now())
