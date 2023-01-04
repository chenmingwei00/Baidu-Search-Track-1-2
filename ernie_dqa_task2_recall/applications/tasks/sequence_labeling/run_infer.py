# -*- coding: utf-8 -*
"""import"""
import os
import sys
sys.path.append("../../../")
import numpy as np
from erniekit.common.register import RegisterSet
from erniekit.common import register
from erniekit.data.data_set import DataSet
import logging
from erniekit.utils import args
from erniekit.utils import params
from erniekit.utils import log



def dataset_reader_from_params(params_dict):
    """
    :param params_dict:
    :return:
    """
    dataset_reader = DataSet(params_dict)
    dataset_reader.build()

    return dataset_reader


def build_inference(params_dict, dataset_reader, parser_handler):
    """build trainer"""
    inference_name = params_dict.get("type", "CustomInference")
    inference_class = RegisterSet.inference.__getitem__(inference_name)
    inference = inference_class(params=params_dict, data_set_reader=dataset_reader, parser_handler=parser_handler)

    return inference


def parse_predict_result(predict_result, input_list, params_dict, seq_lens):
    """按需解析模型预测出来的结果
    :param predict_result: 模型预测出来的结果
    :param input_list: 样本明文数据，dict类型
    :param params_dict: 一些参数配置
    :return:list 希望保存到文件中的结果，output/predict_result.txt
    """
    # return_list = []
    start_probs = predict_result[0].copy_to_cpu()
    pre_similary = start_probs.argmax(axis=-1).tolist()
    pre_prob=start_probs.max(axis=-1)
    return pre_similary,pre_prob


if __name__ == "__main__":
    args = args.build_common_arguments()
    log.init_log("./log/test", level=logging.DEBUG)
    args.param_path='./examples/seqlab_ernie_fc_ch_infer.json'
    param_dict = params.from_file(args.param_path)
    _params = params.replace_none(param_dict)

    # 记得import一下注册的模块
    register.import_modules()
    register.import_new_module("inference", "custom_inference")
    
    dataset_reader_params_dict = _params.get("dataset_reader")
    dataset_reader = dataset_reader_from_params(dataset_reader_params_dict)

    inference_params_dict = _params.get("inference")

    inference = build_inference(inference_params_dict, dataset_reader, parse_predict_result)

    inference.inference_batch()

    logging.info("os exit.")
    os._exit(0)
