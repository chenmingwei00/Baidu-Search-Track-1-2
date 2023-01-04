import json
from collections import Counter
import sys

def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    in_str = ''.join(in_str.split())
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'?', u'!', u';', u'$', u'#',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

def fast_f1(text1, text2):
    common_char = Counter(text1) & Counter(text2)
    len_seq1 = len(text1)
    len_seq2 = len(text2)
    len_common = sum(common_char.values())
    if len_common == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * len_common / len_seq2
    recall = 1.0 * len_common / len_seq1
    return precision, recall, (2.0 * precision * recall) / (precision + recall)

def calc_f1_score_singe_pairs(ref, pred):
    text_a_segs = _tokenize_chinese_chars(_normalize(ref))
    text_b_segs = _tokenize_chinese_chars(_normalize(pred))
    precision, recall, f1 = fast_f1(text_a_segs, text_b_segs)
    return precision, recall, f1
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

