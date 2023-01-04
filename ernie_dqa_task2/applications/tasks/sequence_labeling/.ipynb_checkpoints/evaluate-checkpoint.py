from sklearn.metrics import f1_score
if __name__ == '__main__':
    """
    21 0.8179176644900257
    20 0.8117734592547846
    18 0.8246079040211137
    14 0.8335269727225394
    """
    predict_out='./data/dev_data/4000_14000.txt'
    references = []
    predictions_result = []
    predictions_score = []
    with open('./data/dev_data/dev_answer_nli_data.tsv') as f:
        for line in f:
            if line == '':
                continue
            ture_label=line.strip().split('\t')[-1]
            references.append(ture_label)

    with open(predict_out) as f:
        for line in f:
            if line == '':
                continue
            parts = line.strip().split('\t')[0]
            predictions_score.append(parts)

    f1new=f1_score(references, predictions_score, average='macro')
    print(f1new)
