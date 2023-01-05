from itertools import combinations
from tqdm import tqdm
def t1tot2_test():
    """
    根据第一期的答案进行第二阶段的预测
    :return:
    """
    t1_path='./test_data/test.json'
    t1_answer='./test_data/subtask1_test_pred_combine_0.68987.txt'
    t2_path='./task2_data/data_task2/test_query_doc.json'

    with open(t1_path,'r',encoding='utf-8') as test1, \
        open(t1_answer, 'r', encoding='utf-8') as test1_ans,\
        open(t2_path,'r',encoding='utf-8') as test2:
        lines1=test1.readlines()
        lines_ans1=test1_ans.readlines()
        lines2=test2.readlines()

        id2ans={}
        query2doctext={}
        id=0
        for t1line,pre_ans in zip(lines1,lines_ans1):
            t1_data=json.loads(t1line)

            query=t1_data['query']
            doc_text=t1_data['doc_text']
            # tags = jieba.analyse.extract_tags(query, topK=100, withWeight=True)
            # for v, n in tags:
            #     # 权重是小数，为了凑整，乘了一万
            #     print(v + '\t' + str(int(n * 10000)))
            # max_str = longestCommonSubstr(doc_text.lower(), query.lower())
            # print(max_str)
            #
            # for k,v in t1_data.items():
            #     print(k,v)
            t1_data['id']=id
            if t1_data['query'] not in query2doctext:
                query2doctext[t1_data['query']]=[t1_data]
            else:
                query2doctext[t1_data['query']].append(t1_data)
            id2ans[id]=pre_ans
            id+=1
        new_test2=[]
        for line2 in lines2:
            t2_data=json.loads(line2)
            query=t2_data['query']

            t1_docs=query2doctext[query]
            t1_texts=[ele['doc_text'] for ele in t1_docs]
            docs=t2_data['docs']

            for t_doc in docs:
                t2_doctext=t_doc['doc_text']
                if t2_doctext in t1_texts: #与任务1query和doc_text相同的ans付给任务2
                    doc_indx=t1_texts.index(t2_doctext)
                    t1_macth=t1_docs[doc_indx]
                    t1_id=t1_macth['id']
                    ans=id2ans[t1_id]
                    ans=ans.strip().split('\t')[-1]
                    t_doc['ans']=ans
                else:
                    print('..................33333333...................')
            new_test2.append(t2_data)
        with open('./answer_nli_data.txt','w',encoding='utf-8') as test2pre,\
             open('./context_answer_nli_data.txt','w',encoding='utf-8') as context_pre:
            for t_doc in tqdm(new_test2):
                query=t_doc['query']

                remove_noans=[]

                docs=t_doc['docs']
                contexts= list(combinations(docs, 2))

                for context_pair in contexts:
                    DOC1=context_pair[0]
                    DOC2=context_pair[1]
                    doc_id1=DOC1['doc_id']
                    ans1=DOC1['doc_text']

                    doc_id2 = DOC2['doc_id']
                    ans2 = DOC2['doc_text']

                    test2_pre_line = '\t'.join([query, doc_id1, ans1, doc_id2, ans2])
                    context_pre.write(test2_pre_line + '\n')

                for ele in t_doc['docs']:
                    if ele['ans']!='NoAnswer':
                        remove_noans.append(ele)
                doc_pair = list(combinations(remove_noans, 2))

                for t_pair in doc_pair:
                    test2_pre={}
                    test2_pre['query']=query
                    doc1=t_pair[0]

                    doc2=t_pair[1]
                    ans1=doc1['ans']
                    doc_id1=doc1['doc_id']
                    ans2 = doc2['ans']
                    doc_id2=doc2['doc_id']

                    test2_pre['ans1']=ans1
                    test2_pre['doc_id1']=doc_id1
                    test2_pre['ans2']=ans2
                    test2_pre['doc_id2']=doc_id2
                    test2_pre_line='\t'.join([query,doc_id1,ans1,doc_id2,ans2])
                    test2pre.write(test2_pre_line+'\n')

                    # padd_context=(312-len(ans1)-len(ans2))/2
                    #
                    # while len(ans1)<


        print('1111')

        print('111')
import json
def test_data_split():
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




if __name__ == '__main__':
    # train_data_split()

    predict_deal()
    # combine_result()

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
