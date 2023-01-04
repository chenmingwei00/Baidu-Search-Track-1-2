import json
import copy
import sys
import numpy as np
from itertools import combinations,permutations
import random
# k-means 聚类算法
def createCent(dots,k=2):
    return random.sample(dots, k)
def min_distance(ptsInClust,dot_distance):
    """
    计算该类每一个作为类中心的距离总和，找到最小的距离作为该类中心点
    :param ptsInClust:
    :param dot_distance:
    :return:
    """
    min_distance=10000
    min_dot=''
    for t_dot in ptsInClust:
        distecn_info=dot_distance[t_dot]
        current_dis=0
        for k_dot in ptsInClust:
            if k_dot==t_dot:continue
            current_dis+=distecn_info[k_dot]
        if min_distance>current_dis:
            min_distance=current_dis
            min_dot=t_dot
    return min_dot
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

def new_sort_recall():
    recal_json = '/home/aistudio/work/ernie_dqa_task2_recall/applications/tasks/sequence_labeling/data/test_data/test_recall.json'
    recal_pre = '/home/aistudio/work/ernie_dqa_task2_recall/applications/tasks/sequence_labeling/output/4000.txt'

    with open(recal_json, 'r', encoding='utf-8') as reacll_test_files, \
            open(recal_pre, 'r', encoding='utf-8') as recall_pre_files:
        ercall_datas = reacll_test_files.readlines()
        recall_preda = recall_pre_files.readlines()
        assert len(ercall_datas) == len(recall_preda)
        query2answ = {}
        query2docs_ids={}
        for t_d, t_pre in zip(ercall_datas, recall_preda):
            t_d = json.loads(t_d)
            if t_d['query'] not in query2answ:
                query2answ.update({t_d['query']: {t_d['doc_id']: t_pre.split('\t')}})
            else:
                query2answ[t_d['query']][t_d['doc_id']] = t_pre.split('\t')
            if t_d['query'] not in query2docs_ids:
                query2docs_ids[t_d['query']]={t_d['doc_id']:t_d}
            else:
                query2docs_ids[t_d['query']][t_d['doc_id']]=t_d
    test_data = '/home/aistudio/work/ernie_dqa_task2/applications/tasks/sequence_labeling/data/test_data/answer_nli_data.txt'
    pre_result = '/home/aistudio/work/ernie_dqa_task2/applications/tasks/sequence_labeling/output/4000.txt'

    # test_data = './test_data/5153/answer_nli_data.txt'
    # pre_result = './test_data/5153/4000.txt'
    with open(test_data, 'r', encoding='utf-8') as test_files, \
            open(pre_result, 'r', encoding='utf-8') as pre_files, \
            open('./subtask2_test_pred.txt', 'w', encoding='utf-8') as final_pre:
        test_lines = test_files.readlines()
        pre_files_lines = pre_files.readlines()

        querys2dot_dis_dict = {}
        dot_dis_dict = {}
        cutrent_query = ''
        query2rightans={}

        ids2ans={}
        for id_doc, doc_smilar in zip(test_lines, pre_files_lines):
            t_doc_pairs = id_doc.strip().split('\t')
            id1 = t_doc_pairs[1]
            id2 = t_doc_pairs[3]


            if cutrent_query == '':
                cutrent_query = t_doc_pairs[0]
            elif cutrent_query != t_doc_pairs[0]:
                query2rightans[cutrent_query]=ids2ans

                querys2dot_dis_dict[cutrent_query] = dot_dis_dict
                cutrent_query = t_doc_pairs[0]
                dot_dis_dict = {}
                ids2ans={}
            ids2ans[id1] = t_doc_pairs[2]
            ids2ans[id2] = t_doc_pairs[4]
            doc_smilar_label, doc_smilar_label_distance = doc_smilar.strip().split('\t')
            if doc_smilar_label == '1':
                doc_smilar_label_distance = 1 - float(doc_smilar_label_distance)
            else:
                doc_smilar_label_distance = float(doc_smilar_label_distance)

            if id1 not in dot_dis_dict:
                dot_dis_dict.update({id1: {id2: doc_smilar_label_distance}})
            else:
                dot_dis_dict[id1][id2] = doc_smilar_label_distance

            if id2 not in dot_dis_dict:
                dot_dis_dict.update({id2: {id1: doc_smilar_label_distance}})
            else:
                dot_dis_dict[id2][id1] = doc_smilar_label_distance
        querys2dot_dis_dict[cutrent_query] = dot_dis_dict
        query2rightans[cutrent_query] = ids2ans

        # context_q2dis=context_sort_recall()

        thread=0.8
        for query,dot_dis_lists in querys2dot_dis_dict.items():
            # if '核酸检测复检多久出结果' in query or '苹果录屏多久'  in query\
            #         or '新捷达价位' in query or '苹果14上市时间已定价格' in query or '奔驰1.3t相当于多大排量	' in query:
            #     thread=0.9
            rigths_ans=[]
            dots=[]
            for cand_id,support_ids in dot_dis_lists.items():
                dots.append(cand_id)
                support=0
                for sp_id,dis in support_ids.items():
                    if dis<=thread:
                        support+=1
                if support>=1:
                    rigths_ans.append(cand_id)
            # contextdis=context_q2dis[query]
            # recall_ansids=[]
            # for recal_id,dis_list in contextdis.items():
            #     if recal_id not in rigths_ans:
            #         supp=0
            #         for spid,dis_recal in dis_list.items():
            #             if spid in rigths_ans and dis_recal<thread:
            #                 supp+=1
            #         if supp/len(rigths_ans)>0.90:
            #             recall_ansids.append(recal_id)
            # rigths_ans.extend(recall_ansids)

            # for k_id, value_label in query2answ[query].items():
            #     if k_id not in dots and value_label[0] == '1' and float(value_label[1])>0.90:  #
            #         rigths_ans.append(k_id)
            # final_ = ','.join(list(set(rigths_ans)))
            c_ans = query2rightans[query]
            all_docs=query2docs_ids[query]
            # rigths_ans_copy=copy.deepcopy(rigths_ans)
            c = list(combinations(rigths_ans, 2))
            common=[]
            for tid in c:
                anbs1=c_ans[tid[0]]
                anbs2=c_ans[tid[1]]
                conn_str=longestCommonSubstr(anbs1,anbs2)
                if len(conn_str)<10:continue
                for indx,ele in enumerate(common):
                    if ele in conn_str and len(ele)!=len(conn_str):
                        common[indx]=conn_str
                    elif conn_str in ele:
                        break
                    elif conn_str not in common:
                        common.append(conn_str)
                if len(common)==0:
                    common.append(conn_str)
            for k_id, value_label in query2answ[query].items():
                if value_label[0] == '1' and float(value_label[1])>0.9:  #
                    t_doc=all_docs[k_id]['doc_text']
                    su=0

                    for t_str in common:
                        if t_str in t_doc:
                            su+=1
                        else:
                            char_count=0
                            for ele in t_str:
                                if ele in t_doc:
                                    char_count+=1
                            if char_count/len(t_str)>0.4:
                                su+=1

                    if len(common)==0:
                        continue
                    if su/len(common)>0.3:
                        rigths_ans.append(k_id)
            rigths_ans=list(set(rigths_ans))
            line_result=query+'\t'+','.join(rigths_ans)+'\n'
            final_pre.write(line_result)
    print('finished vote submit files....................')

def kMeans(k=3, distMeans=None):
    recal_json = '/home/aistudio/work/ernie_dqa_task2_recall/applications/tasks/sequence_labeling/data/test_data/test_recall.json'
    recal_pre = '/home/aistudio/work/ernie_dqa_task2_recall/applications/tasks/sequence_labeling/output/4000.txt'

    with open(recal_json, 'r', encoding='utf-8') as reacll_test_files, \
            open(recal_pre, 'r', encoding='utf-8') as recall_pre_files:
        ercall_datas = reacll_test_files.readlines()
        recall_preda = recall_pre_files.readlines()
        assert len(ercall_datas) == len(recall_preda)
        query2answ = {}
        for t_d, t_pre in zip(ercall_datas, recall_preda):
            t_d = json.loads(t_d)
            if t_d['query'] not in query2answ:
                query2answ.update({t_d['query']: {t_d['doc_id']: t_pre.split('\t')}})
            else:
                query2answ[t_d['query']][t_d['doc_id']] = t_pre.split('\t')

    test_data = '/home/aistudio/work/ernie_dqa_task2/applications/tasks/sequence_labeling/data/test_data/answer_nli_data.txt'
    pre_result = '/home/aistudio/work/ernie_dqa_task2/applications/tasks/sequence_labeling/output/4000.txt'
    with open(test_data, 'r', encoding='utf-8') as test_files, \
            open(pre_result, 'r', encoding='utf-8') as pre_files, \
            open('./subtask2_test_pred.txt', 'w', encoding='utf-8') as final_pre:
        test_lines = test_files.readlines()
        pre_files_lines = pre_files.readlines()

        querys2dot_dis_dict={}
        dot_dis_dict={}
        cutrent_query=''
        for id_doc,doc_smilar in zip(test_lines,pre_files_lines):
            t_doc_pairs=id_doc.strip().split('\t')
            id1=t_doc_pairs[1]
            id2=t_doc_pairs[3]
            if cutrent_query=='':
                cutrent_query=t_doc_pairs[0]
            elif cutrent_query!=t_doc_pairs[0]:
                querys2dot_dis_dict[cutrent_query]=dot_dis_dict
                cutrent_query=t_doc_pairs[0]
                dot_dis_dict={}

            doc_smilar_label,doc_smilar_label_distance=doc_smilar.strip().split('\t')
            if doc_smilar_label=='1':
                doc_smilar_label_distance=1-float(doc_smilar_label_distance)
            else:
                doc_smilar_label_distance=float(doc_smilar_label_distance)

            if id1 not in dot_dis_dict:
                dot_dis_dict.update({id1:{id2:doc_smilar_label_distance}})
            else:
                dot_dis_dict[id1][id2]= doc_smilar_label_distance

            if id2 not in dot_dis_dict:
                dot_dis_dict.update({id2:{id1:doc_smilar_label_distance}})
            else:
                dot_dis_dict[id2][id1]= doc_smilar_label_distance
        querys2dot_dis_dict[cutrent_query] = dot_dis_dict
        cutrent_query = t_doc_pairs[0]
        for query,dot_distance in querys2dot_dis_dict.items():
            step=0
            center_list=[]
            center_distance=[]
            while step<20:
                m = len(dot_distance)
                dots=list(dot_distance.keys())
                counts = []
                clusterAssments = []
                centroids = []
                clusterAssment = np.array(np.zeros((m, 2)))  # 用于存放该样本属于哪类及质心距离
                # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
                centroid = createCent(list(dot_distance.keys()), k)
                clusterChanged = True  # 用来判断聚类是否已经收敛
                while clusterChanged:
                    clusterChanged = False
                    count = 0
                    for i in range(m):  # 把每一个数据点划分到离它最近的中心点
                        minDist = np.inf;
                        minIndex = -1;
                        for j in range(k):
                            if centroid[j]==dots[i]:
                                distJI=0.0
                            else:
                                try:
                                    distJI = dot_distance[centroid[j]][dots[i]]
                                except:
                                    print('111')
                            if distJI < minDist:
                                minDist = distJI;
                                minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
                        if clusterAssment[i, 0] != minIndex:
                            clusterChanged = True;
                            count += 1  # 如果分配发生变化，则需要继续迭代
                            # print(clusterAssment[i,0],'-->',minIndex)
                        clusterAssment[i, :] = minIndex, minDist ** 2  # 并将第i个数据点的分配情况存入字典

                    for cent in range(k):
                        dots_arr=np.array(dots)# 重新计算中心点
                        ptsInClust = dots_arr[clusterAssment[:, 0] == cent]  # 去第一列等于cent的所有列

                        centroid[cent] = min_distance(ptsInClust,dot_distance) # 算出这些数据的中心点

                    # 此处为坑
                    #         centroids.append(centroid)
                    #         clusterAssments.append(clusterAssment)
                    if clusterChanged == True:
                        centroids.append(copy.copy(centroid))
                        clusterAssments.append(copy.copy(clusterAssment))
                        counts.append(count)
                center_list.append(centroids[-1])
                center_distance.append(clusterAssments[-1])
                step+=1
            dot_dict2={}
            for ele in center_list:
                if ele[0] not in dot_dict2:
                    dot_dict2[ele[0]]=1
                else:
                    dot_dict2[ele[0]] += 1

                if ele[1] not in dot_dict2:
                    dot_dict2[ele[1]]=1
                else:
                    dot_dict2[ele[1]] += 1
            dot_score=sorted(dot_dict2.items(),key=lambda k:k[-1],reverse=True)
            dot_score=[ele[0] for ele in dot_score]
            centroid=dot_score[:k]
            clusterAssment = np.array(np.zeros((m, 2)))  # 用于存放该样本属于哪类及质心距离

            for i in range(m):  # 把每一个数据点划分到离它最近的中心点
                minDist = np.inf;
                minIndex = -1;
                for j in range(k):
                    if centroid[j] == dots[i]:
                        distJI = 0.0
                    else:
                        distJI = dot_distance[centroid[j]][dots[i]]
                    if distJI < minDist:
                        minDist = distJI;
                        minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True;
                    count += 1  # 如果分配发生变化，则需要继续迭代
                    # print(clusterAssment[i,0],'-->',minIndex)
                clusterAssment[i, :] = minIndex, minDist ** 2  # 并将第i个数据点的分配情况存入字典
            anser_final=[]
            for cent in range(k):
                dots_arr = np.array(dots)  # 重新计算中心点
                ptsInClust = dots_arr[clusterAssment[:, 0] == cent]  # 去第一列等于cent的所有列
                if len(ptsInClust)<=1:
                    continue
                anser_final.extend(list(ptsInClust))
            for k_id, value_label in query2answ[query].items():
                if k_id not in dots and value_label[0] == '1':  #
                    anser_final.append(k_id)
            final_ = ','.join(list(set(anser_final)))
            result_line = query + '\t' + final_ + '\n'

            final_pre.write(result_line)
    print('fininsehed k-mean ......................')
if __name__ == '__main__':
    if sys.argv[1]=='k-mean':
        kMeans(k=3)
    else:
        new_sort_recall()
    
