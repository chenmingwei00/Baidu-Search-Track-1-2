

import json
def t_data_split():
    """
    首先对预测文本doc_text,按照最大文本512 进行分割，分别预测，最后答案进行合并
    """
    max_seq_len=505
    with open('./test_data/test_pre.json','r',encoding='utf-8') as train_file,\
            open('./test_data/test.json','w',encoding='utf-8') as pre_del_file:
        lines=train_file.readlines()
        id=0
        for line in lines:
            t_data=json.loads(line)
            query=t_data['query']
            doc_text=t_data['doc_text']
            if len(query)+len(doc_text)>max_seq_len:
                doc_len=max_seq_len-len(query)
                new_docs=[]
                for k in range(len(doc_text)):
                    if len(new_docs)<doc_len:
                        new_docs.append(doc_text[k])
                    else:
                        t_data['doc_text']=''.join(new_docs)
                        t_data['id'] = id
                        t_data_line = json.dumps(t_data, ensure_ascii=False)
                        pre_del_file.write(t_data_line + '\n')
                        new_docs=[]
                t_data['doc_text'] = ''.join(new_docs)
                t_data['id'] = id
                t_data_line = json.dumps(t_data, ensure_ascii=False)
                pre_del_file.write(t_data_line + '\n')
                new_docs = []

            else:
                t_data['id']=id
                t_data_line=json.dumps(t_data,ensure_ascii=False)
                pre_del_file.write(t_data_line+'\n')
            id+=1
from tqdm import tqdm
def train_data_split():
    """
    对训练数据query+doc_text长度的文本进行截断形成多个数据集
    """
    max_seq_len=505
    with open('./train_data/train.json','r',encoding='utf-8') as train_file,\
            open('./train_data/train_split.json','w',encoding='utf-8') as pre_del_file:
        lines = train_file.readlines()
        id = 0
        neg_samples=0
        new_lines=0
        for line in tqdm(lines):
            t_data = json.loads(line)
            query = t_data['query']
            doc_text = t_data['doc_text']
            answer_list = t_data['answer_list']
            answer_start_list=t_data['answer_start_list']
            org_answer=t_data['org_answer']

            doc_text=doc_text.replace('~',"#")
            #首先对答案进行扩充到与doc_text长度相同
            new_all_answer=[]
            new_answers_type=[]
            for start_indx,t_ans in zip(answer_start_list,answer_list):
                t_ans=t_ans.replace('~',"#")
                if len(new_all_answer)==0:
                    padd_ans=['~']*start_indx+list(t_ans)
                    padd_type=[0]*start_indx
                    padd_type+=[1]*len(list(t_ans))

                    new_answers_type.extend(padd_type)
                    new_all_answer.extend(padd_ans)
                elif len(new_all_answer)<=start_indx:#说后边答案与前边答案有距离
                    padd_len=start_indx-len(new_all_answer)
                    padd_ans=['~']*padd_len+list(t_ans)

                    padd_type = [0] * padd_len
                    padd_type += [1] * len(list(t_ans))

                    new_answers_type+=padd_type
                    new_all_answer+=padd_ans
            if len(new_all_answer)<len(list(doc_text)):
                new_all_answer+=['~']*(len(list(doc_text))-len(new_all_answer))
                new_answers_type+=[0]*(len(list(doc_text))-len(new_answers_type))

            assert len(new_all_answer)==len(doc_text)
            assert len(new_answers_type)==len(doc_text)

            if len(query) + len(doc_text) > max_seq_len:
                doc_len = max_seq_len - len(query)

                new_docs = []
                tmp_ans=[]
                tmp_ans_types=[]

                for k in range(len(doc_text)):

                    if len(new_docs) < doc_len:
                        new_docs.append(doc_text[k])
                        tmp_ans.append(new_all_answer[k])
                        tmp_ans_types.append(new_answers_type[k])
                    else:
                        new_traindata={}
                        new_answer_list = []
                        new_answer_start_list = []
                        new_org_answer = ''
                        first=True
                        tmp_constru_ans=''
                        for step,type_id in enumerate(tmp_ans_types):
                            if type_id==1 and first==True:
                                new_answer_start_list.append(step)
                                tmp_constru_ans+=tmp_ans[step]
                                first=False
                            elif type_id==1 and first==False:
                                tmp_constru_ans += tmp_ans[step]
                                first = False
                            elif type_id==0 and first==False:
                                new_answer_list.append(tmp_constru_ans)
                                tmp_constru_ans=''
                                first=True
                        if len(new_answer_list)==0:
                            new_org_answer='NoAnswer'
                            if neg_samples>30000:
                                continue
                            neg_samples+=1
                        else:
                            new_org_answer=org_answer
                        if len(new_docs)>50:
                            new_traindata['answer_list']=new_answer_list
                            new_traindata['answer_start_list']=new_answer_start_list
                            new_traindata['doc_text']=''.join(new_docs)
                            new_traindata['org_answer']=new_org_answer
                            new_traindata['query'] = t_data['query']
                            new_traindata['title'] = t_data['title']
                            new_traindata['url'] = t_data['url']

                            new_traindata['id'] = id
                            t_data_line = json.dumps(new_traindata, ensure_ascii=False)
                            pre_del_file.write(t_data_line + '\n')

                            new_docs = []
                            tmp_ans=[]
                            tmp_ans_types = []
                            new_lines+=1

                new_traindata = {}
                new_answer_list = []
                new_answer_start_list = []
                new_org_answer = ''
                first = True
                tmp_constru_ans = ''
                if sum(tmp_ans_types)>0 and len(new_docs)>50:
                    for step, type_id in enumerate(tmp_ans_types):
                        if type_id == 1 and first == True:
                            new_answer_start_list.append(step)
                            tmp_constru_ans += tmp_ans[step]
                            first = False
                        elif type_id == 1 and first == False:
                            tmp_constru_ans += tmp_ans[step]
                            first = False
                        elif type_id == 0 and first == False:
                            new_answer_list.append(tmp_constru_ans)
                            tmp_constru_ans = ''
                            first = True
                    if len(new_answer_list) == 0:
                        new_org_answer = 'NoAnswer'
                        if neg_samples > 30000:
                            continue
                        neg_samples += 1
                    else:
                        new_org_answer = org_answer
                    new_traindata['answer_list'] = new_answer_list
                    new_traindata['answer_start_list'] = new_answer_start_list
                    new_traindata['doc_text'] = ''.join(new_docs)
                    new_traindata['org_answer'] = new_org_answer
                    new_traindata['query'] = t_data['query']
                    new_traindata['title'] = t_data['title']
                    new_traindata['url'] = t_data['url']

                    new_traindata['id'] = id
                    t_data_line = json.dumps(new_traindata, ensure_ascii=False)
                    pre_del_file.write(t_data_line + '\n')

                    new_docs = []
                    tmp_ans = []
                    tmp_ans_types = []
                    new_lines += 1

            else:
                t_data['id'] = id
                t_data_line = json.dumps(t_data, ensure_ascii=False)
                pre_del_file.write(t_data_line + '\n')
                new_lines += 1

            id += 1
    print('finished.......................')
    print(new_lines)
import math
def predict_deal():
    with open('./test_data/4001.txt','r',encoding='utf-8') as pre_answer,\
        open('./test_data/test_new.json','r',encoding='utf-8') as ori_json,\
        open('./test_data/subtask1_test_pred.txt','w',encoding='utf-8') as final_pre:
        pre_lines=pre_answer.readlines()
        input_lines=ori_json.readlines()

        assert len(pre_lines)==len(input_lines)
        id2answer={}
        for t_pre,t_input in zip(pre_lines,input_lines):
            t_inputs=json.loads(t_input)
            if t_inputs['id'] not in id2answer:
                id2answer[t_inputs['id']]=[t_pre]
            else:
                id2answer[t_inputs['id']].append(t_pre)
        id2answer_list=sorted(id2answer.items(),key=lambda k:k[0])
        for id, ansers_t in id2answer_list:
            trues_answer = []
            true_answer_score = []
            no_answer = []
            no_answer_scores = []
            for t_ans in ansers_t:
                t_ans = t_ans.strip().split('\t')
                if t_ans[1] == 'NoAnswer':
                    no_answer_scores.append(float(t_ans[0]))
                    no_answer.append(t_ans[1])
                else:
                    trues_answer.append(t_ans[1])
                    true_answer_score.append(float(t_ans[0]))
            if len(trues_answer)<=0:
                score=sum(no_answer_scores)/len(no_answer_scores)
                line=str(score)+'\t'+'NoAnswer'+'\n'
            else:
                score=sum(true_answer_score)/len(true_answer_score)
                line=str(score)+'\t'+''.join(trues_answer)+'\n'
                line=line.replace('#','')
            final_pre.write(line)

import re
def seg_tail_split(str1,sep=r":|,|。|，|。|？|！|；|,|.|?|!|;|"): # 分隔符可为多样的正则表达式
    # 保留分割符号，置于句尾，比如标点符号
    try:
        wlist = re.split(sep,str1)
        seg_word = re.findall(sep,str1)
        seg_word.extend(" ") # 末尾插入一个空字符串，以保持长度和切割成分相同
        wlist = [ x+y for x,y in zip(wlist,seg_word) ] # 顺序可根据需求调换
        return wlist
    except:
        return [str1]
def combine_result():
    original_path='./test_data/submit_result/0.6700_subtask1_test_pred.txt'
    original_path2='./test_data/subtask1_test_pred.txt'

    with open(original_path,'r',encoding='utf-8') as pre1, \
            open(original_path2, 'r', encoding='utf-8') as pre2,\
            open('./subtask1_test_pred2.txt', 'w', encoding='utf-8') as conbine_pre:
        lines1=pre1.readlines()
        lines2=pre2.readlines()

        for t_l1,t_l2 in zip(lines1,lines2):
            scores1,answer1=t_l1.strip().split('\t')
            scores2,answer2=t_l2.strip().split('\t')

            if answer1=='NoAnswer' and answer2=='NoAnswer':
                conbine_pre.write(t_l1)
            elif answer1!='NoAnswer' and answer2=='NoAnswer':
                conbine_pre.write(t_l1)
            elif answer1=='NoAnswer' and answer2!='NoAnswer':
                conbine_pre.write(t_l2)
            else:
                rest_listr=[]
                answer2_list=seg_tail_split(answer2)
                for charlist in answer2_list:
                    common=0
                    for char in charlist:
                        if char in answer1:
                            common+=1
                    if common/len(charlist)<0.90:
                        rest_listr.append(charlist)
                if len(rest_listr)>0:
                    final_answer=answer1.strip()+''.join(rest_listr).strip()+'\n'
                    conbine_pre.write(scores1+'\t'+final_answer)
                else:
                    conbine_pre.write(t_l1)


def final_predata():
    test_data='./test_data/answer_nli_data.txt'
    pre_result='./test_data/4000.txt'

    with open(test_data,'r',encoding='utf-8') as test_files,\
        open(pre_result,'r',encoding='utf-8') as pre_files,\
        open('./subtask2_test.txt','w',encoding='utf-8') as final_pre:
        test_lines=test_files.readlines()
        pre_files_lines=pre_files.readlines()

        clusters={}
        queyr_current=''
        id_main=-1
        id2scoires={}
        for t_data,t_pre in zip(test_lines,pre_files_lines):

            similary,simlary_score= t_pre.strip().split('\t')

            t_data=t_data.strip().split('\t')
            query=t_data[0]

            if id_main==-1:
                id_main= t_data[1]
                if similary == '1':
                    id2scoires[t_data[3]]=simlary_score
                
            elif id_main==t_data[1]:
                if similary=='1':
                    id2scoires[t_data[3]]=simlary_score
            else:
                clusters[id_main]=id2scoires
                id_main = t_data[1]
                id2scoires={}

                for match_id,t_scoredict in clusters.items():
                    if id_main in t_scoredict:
                        id2scoires[match_id]=t_scoredict[id_main]
                if similary=='1':
                    id2scoires[t_data[3]]=simlary_score

            if queyr_current=='':
                queyr_current=query
            elif queyr_current==query:
                pass
            else:
                result=sorted(clusters.items(), key=lambda k: len(k[1]))[-5:]
                id_mainid=[ele[0] for ele in result]
                result=[ele[-1] for ele in result]
                final_=list(set(result[0].keys())|set(result[1].keys())| set(result[2].keys())|set(result[3].keys())|set(result[4].keys()))

                final_.extend(id_mainid)
                final_=','.join(list(set(final_)))
                result_line=queyr_current+'\t'+final_+'\n'
                final_pre.write(result_line)
                queyr_current=query
                clusters={}
        result = sorted(clusters.items(), key=lambda k: len(k[1]))[-5:]
        id_mainid = [ele[0] for ele in result]
        result = [ele[-1] for ele in result]
        final_ = list(set(result[0].keys())|set(result[1].keys())| set(result[2].keys())|set(result[3].keys())|set(result[4].keys()))
        final_.extend(id_mainid)
        final_ = ','.join(list(set(final_)))
        result_line = queyr_current + '\t' + final_ + '\n'
        final_pre.write(result_line)
        queyr_current = query
        clusters = {}
import copy

def longestCommonSubstr(word1: str, word2: str) -> int:

    m = len(word1)
    n = len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # dp[i][j]代表word1以i结尾,word2以j结尾，的最大公共子串的长度

    max_len = 0
    row = 0
    col = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if max_len < dp[i][j]:
                    max_len = dp[i][j]
                    row = i
                    col = j

    max_str = ""
    i = row
    j = col
    while i > 0 and j > 0:
        if dp[i][j] == 0:
            break
        i -= 1
        j -= 1
        max_str += word1[i]

    lcstr = max_str[::-1]
    # 回溯的得到的最长公共子串
    return max_len,lcstr
from itertools import combinations,permutations

def final_predata2():
    test_data = './test_data/answer_nli_data.txt'
    pre_result = './test_data/4000.txt'

    ori_test_data_path='./test_data/test_query_doc.json'

    with open(test_data, 'r', encoding='utf-8') as test_files, \
            open(pre_result, 'r', encoding='utf-8') as pre_files, \
            open(ori_test_data_path,'r',encoding='utf-8') as ori_data,\
            open('./subtask2_test_pred.txt', 'w', encoding='utf-8') as final_pre:

        query_docall=ori_data.readlines()
        test_lines = test_files.readlines()
        pre_files_lines = pre_files.readlines()
        query2docs={}
        for line in query_docall:
            t_line=json.loads(line)
            query=t_line['query']
            docs=t_line['docs']
            query2docs[query]=docs


        clusters = {}
        queyr_current = ''
        id_main = -1
        id2scoires = {}
        ids2anss={}
        for t_data, t_pre in zip(test_lines, pre_files_lines):

            similary, simlary_score = t_pre.strip().split('\t')

            t_data = t_data.strip().split('\t')
            query = t_data[0]

            if id_main == -1:
                id_main = t_data[1]
                if similary == '1':
                    id2scoires[t_data[3]] = simlary_score

            elif id_main == t_data[1]:
                if similary == '1':
                    id2scoires[t_data[3]] = simlary_score
            else:
                clusters[id_main] = id2scoires
                id_main = t_data[1]
                id2scoires = {}

                for match_id, t_scoredict in clusters.items():
                    if id_main in t_scoredict:
                        id2scoires[match_id] = t_scoredict[id_main]
                if similary == '1':
                    id2scoires[t_data[3]] = simlary_score

            if queyr_current == '':
                queyr_current = query
            elif queyr_current == query:
                ids2anss[t_data[1]] = t_data[2]
                ids2anss[t_data[3]] = t_data[4]
            else:
                result = sorted(clusters.items(), key=lambda k: len(k[1]))[-5:]
                id_mainid = [ele[0] for ele in result]
                result = [ele[-1] for ele in result]
                final_ = list(set(result[1].keys()) | set(result[2].keys()) | set(result[3].keys()) | set(
                        result[4].keys()))

                final_.extend(id_mainid)
                final_=list(set(final_))
                add_ids=[]
                all_doc=query2docs[queyr_current]

                c = list(combinations(final_, 2))
                comnnstrs=[]
                for indx,tid in c:
                    tid_ans = ids2anss[indx]
                    tid_ans2 = ids2anss[tid]

                    comnn, comnnstr = longestCommonSubstr(tid_ans, tid_ans2)
                    if len(comnnstr)>5:
                        comnnstrs.append(comnnstr)


                for t_doc in all_doc:
                    support=0
                    if t_doc['doc_id'] not in final_:
                        t_doctext=t_doc['doc_text']
                        for t_common in comnnstrs:

                            comnn,comnnstr=longestCommonSubstr(t_common,t_doctext)
                            if comnn/len(t_common) >0.90:
                                support+=1
                    if support>8:
                        add_ids.append(t_doc['doc_id'])
                final_.extend(add_ids)
                final_=list(set(final_))

                final_=','.join(final_)
                result_line = queyr_current + '\t' + final_ + '\n'
                final_pre.write(result_line)
                queyr_current = query
                clusters = {}

        result = sorted(clusters.items(), key=lambda k: len(k[1]))[-5:]
        id_mainid = [ele[0] for ele in result]
        result = [ele[-1] for ele in result]
        final_ = list(set(result[1].keys()) | set(result[2].keys()) | set(result[3].keys()) | set(
                result[4].keys()))

        final_.extend(id_mainid)
        final_ = list(set(final_))

        add_ids = []
        all_doc = query2docs[queyr_current]

        c = list(combinations(final_, 2))
        comnnstrs = []
        for indx, tid in c:
            tid_ans = ids2anss[indx]
            tid_ans2 = ids2anss[tid]

            comnn, comnnstr = longestCommonSubstr(tid_ans, tid_ans2)
            if len(comnnstr) > 6:
                comnnstrs.append(comnnstr)

        for t_doc in all_doc:
            support = 0
            if t_doc['doc_id'] not in final_:
                t_doctext = t_doc['doc_text']
                for t_common in comnnstrs:

                    comnn, comnnstr = longestCommonSubstr(t_common, t_doctext)
                    if comnn / len(t_common) > 0.90:
                        support += 1
            if support > 8:
                add_ids.append(t_doc['doc_id'])
        final_.extend(add_ids)
        final_ = list(set(final_))

        final_ = ','.join(final_)
        result_line = queyr_current + '\t' + final_ + '\n'
        final_pre.write(result_line)
        queyr_current = query
        clusters = {}

def constaruct_second_data():
    query_ids_data = './train_data/train_label.tsv'
    pre_result = './train_data/train_query_doc.json'

    dev_label_path= './dev_data/dev_label.tsv'
    dev_querydoc_path= './dev_data/dev_query_doc.json'

    new_train='./train_data/train_recall.json'
    with open(query_ids_data, 'r', encoding='utf-8') as test_files, \
            open(new_train, 'w', encoding='utf-8') as new_data_files, \
            open(dev_querydoc_path, 'r', encoding='utf-8') as dev_doc,\
            open(dev_label_path, 'r', encoding='utf-8') as dev_label_files,\
            open(pre_result, 'r', encoding='utf-8') as pre_files:
        query_ids=test_files.readlines()
        pre_result_line=pre_files.readlines()
        all_train_counts=0
        for label,input in zip(query_ids,pre_result_line):
            query,label_ids=label.strip().split('\t')
            label_ids=label_ids.split(',')
            t_data=json.loads(input)

            match_query=t_data['query']

            assert query==match_query
            all_train_counts+=len(t_data['docs'])
            for t_doc in t_data['docs']:
                t_id=t_doc['doc_id']
                if t_id in label_ids:
                    t_doc['label']=1
                else:
                    t_doc['label'] = 0

                t_doc['query']=query
                t_doc_line=json.dumps(t_doc,ensure_ascii=False)
                new_data_files.write(t_doc_line+'\n')

        query_ids = dev_label_files.readlines()
        pre_result_line = dev_doc.readlines()
        for label, input in zip(query_ids, pre_result_line):
            query, label_ids = label.strip().split('\t')
            label_ids = label_ids.split(',')
            t_data = json.loads(input)

            match_query = t_data['query']

            assert query == match_query
            all_train_counts += len(t_data['docs'])
            for t_doc in t_data['docs']:
                t_id = t_doc['doc_id']
                if t_id in label_ids:
                    t_doc['label'] = 1
                else:
                    t_doc['label'] = 0

                t_doc['query'] = query
                t_doc_line = json.dumps(t_doc, ensure_ascii=False)
                new_data_files.write(t_doc_line + '\n')

    print('finished')
    print(all_train_counts)

if __name__ == '__main__':
    constaruct_second_data()
    # train_data_split()

    # predict_deal()
    # combine_result()
    # final_predata2()
    # with open('./train_data/train_split.json','r',encoding='utf-8') as files_read:
    #     lines=files_read.readlines()
    #     print(len(lines))
    #
    #     no_answer=0
    #     for line in lines:
    #          data=json.loads(line)
    #          ansecout=data['answer_start_list']
    #          if len(ansecout)<=0:
    #              no_answer+=1
    #     print(no_answer)
