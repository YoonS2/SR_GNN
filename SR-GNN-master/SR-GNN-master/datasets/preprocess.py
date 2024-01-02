
#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

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

# 데이터셋 지정 기본적으로 sample 데이터 사용 
## 다른 데이터셋 사용하려면 preprocess.py --dataset=diginetica 또는  preprocess.py --dataset=yoochoose 입력
dataset = 'sample_train-item-views.csv'  
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f: # mode='r' : 읽기용으로 파일을 엶
    if opt.dataset == 'yoochoose': # 데이터셋에 따라 구분자 달라짐
        reader = csv.DictReader(f, delimiter=',')  # csv.DictReader : 파일의 각 row를 딕셔너리 형태로 읽어옴 
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader: # reader가 딕셔너리 형태임. for문으로 row를 하나씩 dict형태로 불러옴 
        sessid = data['session_id'] # 해당 row의 sessionid key에 해당하는 값을 sessid에 부여 
        if curdate and not curid == sessid: # curid랑 세션아이디랑 같지 않고 curdate는 True이면 
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date # 세션 데이트 dict형태로 넣어줌
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe']) # timeframe 아이템 클릭 시간인듯? 
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:  # sess_clicks에 sessionid가 존재하면 추가를 해주고 
            sess_clicks[sessid] += [item]
        else:  # 존재하지 않으면 새롭게 session id 를 넣어주라는 의미
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):   # sess_clicks는 dict이므로 list()해주면 dict의 key값들 출력. 즉 session id값 출력
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            # operator.itemgetter : 다양한 기준으로 정렬하기 위해 사용함. sess_clicks를 정렬하는데, 1번째 key의 value로 정렬한다는 의미 
            ## 즉 , sesseion id가 같은 아이템들을 아이템 id 기준으로 정렬한다는 의미 
            sess_clicks[i] = [c[0] for c in sorted_clicks]  # 정렬한 값 순서대로 sess_clicks의 해당 세션id의 value를 리스트 형태로 넣어줌
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:  # 만약에 한 세션아이디에 클릭한 수가 1개밖에 없으면 해당 세션은 제외
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks: # list(sess_clicks)랑 똑같음. sess_clicks의 key가 s로 들어감
    seq = sess_clicks[s] # session id의 정렬한 아이템 리스트가 seq에 들어감 
    for iid in seq: # 전체 세션에서 이 아이템의 클릭수를 구하기 위함
        if iid in iid_counts: # 이 아이템이 iid_counts dict에 존재하면 1을 더해줌
            iid_counts[iid] += 1
        else: # 이 아이템이 iid_counts dict에 존재하지 않으면 1이라는 값을 부여해줌 (처음 집계 된거니까)
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1)) # 아이템들을 클릭된 횟수 순으로 정렬해줌

length = len(sess_clicks) # 전체 세션의 개수 
for s in list(sess_clicks): # s에 세션 아이디 넣어줌 
    curseq = sess_clicks[s] # s의 아이템들 넣어줌 
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq)) # curseq에 존재하는 아이템들중 클릭이 5번 이상 된 아이템들 필터
    if len(filseq) < 2: # 다섯번 이상 클릭된 아이템의 개수가 2개 미만이면 해당 세션 제외함
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq # 클릭이 5번 이상 된 아이템만 필터한것을 다시 해당 세션의 아이템으로 넣어줌 

### 세션 아이디 중 아이템이 1개 이하이면 해당 세션 제외. 아이템이 2개이상이여도 그 아이템들이 전체 세션에서 5번 이상 클릭된 아이템들만 사용함
#### 만약 5번 이상 클릭된 아이템이 한개뿐이면 그 세션 아이디도 제외됨 

# Split out test set based on dates
dates = list(sess_date.items())  # sess_date의 각 key와 value가 튜플 형태로 된 것을 리스트로 만들어줌 즉 [(key1, value1),(key2,value2)] 이런모양 
maxdate = dates[0][1] # 일단 maxdate에 값 넣어놓음

for _, date in dates: # maxdate값 구하기위한 for문 
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates) # splitdate를 기점으로 앞데이터는 트레이닝, 뒤 데이터는 테스트로 사용
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date  트레이닝과 테스트 데이터 각각 날짜 순으로 정렬해줌 
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess: # tra_sess:  [(session_id, timestamp), (), ]
        seq = sess_clicks[s]  # 트레이닝 데이터의 세션 아이디에 해당하는 아이템 리스트 
        outseq = []
        for i in seq: # 아이템 리스트의 해당 아이템이 
            if i in item_dict: # item_dict에 존재하면 
                outseq += [item_dict[i]] #outseq에 item_dict의 해당 아이템의 value값 (item_ctr) 넣어줌 
            else: # 아이템이 item_dict에 존재하지 않으면, 
                outseq += [item_ctr] # outseq에 item_ctr넣고 
                item_dict[i] = item_ctr #item_dict에 해당 아이템을 key로, item_ctr을 value로 넣음 
                item_ctr += 1 # item_ctr에 1을 더해줌
            # item_ctr은 결국 아이템이 출연(?)하는 순서대로 번호를 매겨준것임. 번호 1번부터. 
        if len(outseq) < 2:  # Doesn't occur # 아이템이 1개뿐이라는 말이 되므로.. 이런 애들은 미리 제외시켰음 앞전에 
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs  # session id, session_date, 아이템no


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():  # training set처럼 test set도 같은 방식으로 처리해줌. item이 training에는 존재하지 않으면 무시함(해당 아이템 제외)
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs # session id, session_date, 아이템no


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates): # iseqs는 아이템 번호 , idates는 해당 시퀀스의 date값 
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates): # iseqs는 시퀀스의 id. 해당 dataset에서 시퀀스에 순서값(id) 부여해줌 
        for i in range(1, len(seq)):
            tar = seq[-i] # 시퀀스의 맨마지막, 뒤에서 두번째.. 순으로 target으로 넣음 1~t까지의 시간이 있다면 t, t-1, t-2.. 번째에 해당하는 아이템no가 target이 됨
            labs += [tar] # 라벨 리스트에 추가 
            out_seqs += [seq[:-i]] # target을 학습시키기 위해 target 앞까지의 아이템no를 넣어줌 
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids # 학습데이터, seq날짜, 라벨, 시퀀스id(시퀀스 순서no )


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs) # training의 시퀀스 데이터에서 label앞까지의 데이터/ label 
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs: # tra_seqs: 아이템no 
    all += len(seq) # 전체 시퀀스에 존재하는 아이템 개수 
for seq in tes_seqs:
    all += len(seq) # 테스트 셋에 존재하는 아이템개수까지 모두 더해줌 
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0)) # 시퀀스당 평균 아이템 개수 보여줌 
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb')) # pickle.dump(객체, 파일) :객체를 파일로 저장
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))  ## pickle.load(파일)은 파일을 로딩한다는 의미
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64) # 1/4, 1/64 로 나눠주기 위함 
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
