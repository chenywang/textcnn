# -*- coding:utf-8 -*-
# @Author : Michael-Wang

import os

PROJECT_PATH = os.path.abspath(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.pardir))


class Config(object):
    # 项目路径
    project_path = PROJECT_PATH + "/.."

    # 定制化字典
    dict_path = project_path + "/data/dict"
    user_dict_path = dict_path + "/user_dict.dict"
    user_tag_path = dict_path + "/user_tag.dict"
    stopwords_path = dict_path + "/user_stopwords.dict"

    # ltp模型路径（可以根据自己下载的数据地址来覆盖改写）
    ltp_model_path = project_path + '/data/ltp_data'
    seg_model_path = ltp_model_path + '/cws.model'
    pos_model_path = ltp_model_path + '/pos.model'
    par_model_path = ltp_model_path + '/parser.model'

    # 模型文件：word2vec, doc2vec, 情感分析模型
    model_path = project_path + "/data/model"
    wordvector_model_path = model_path + "/word2vec.model"
    word2vec_format_model_path = model_path + "/word2vec.bin"
    docvector_model_path = model_path + "/doc2vec.model"
    sentiment_model_path = model_path + "/sentiment.model"

    sentiment_test_output_path = project_path + "/data/output/sentiment_test.csv"

    # 标签聚类模型
    cluster_tag_model_path = project_path + "/data/model/cluster_tag.model"

    # 数据路径
    input_data_path = project_path + "/data/review_10.csv"
    dsr_review_path = project_path + "/data/review_dsr.csv"
    dsr_review_100_path = project_path + "/data/review_dsr_100.csv"
    dsr_review_1000_path = project_path + "/data/review_dsr_1000.csv"

    segged_path = project_path + "/data/output/review.segged"
    tag_path = project_path + "/data/output/tags.csv"
    structured_tag_path = project_path + "/data/output/structured_tags.csv"
    tag_count_path = project_path + "/data/output/tag_count.csv"
    structured_tag_counts_path = project_path + "/data/output/structured_tag_counts.csv"

    # structured_tag_counts_path = project_path + "/data/output/100_structured_tag_counts.csv"
    tag_counts_segged_path = project_path + "/data/tag_counts.segged"
    tag_data_path = project_path + "/data/tag_data.npy"
    sentiment_input_path = project_path + "/data/output/tag_counts_oneday.csv"
    sentiment_validation_output_path = project_path + "/data/output/sentiment_validation.csv"
    sentiment_output_path = project_path + "/data/output/sentiment.csv"
    tag_sentiment_model_path = project_path + "/data/model/tag_sentiment_lr.model"
    tag_sentiment_result = project_path + "/data/output/tag_sentiment_predict.csv"
    predicted_result_path = project_path + "/data/output/predicted_result.csv"
    expand_predicted_result_path = project_path + "/data/output/expand_predicted_result.csv"

    delta_hit_output_path = '/user/penhuolong/output/delta_hit_result'
    delta_miss_output_path = '/user/penhuolong/output/delta_miss_result'
    tags = './tags/expand_predicted_result_250.csv'

    # spark测试路径
    spark_test_out = project_path + "/data/output/spark_out"
    spark_test_segged = project_path + "/data/spark_review.segged"

    # 聚类数据路径
    cluster_path = project_path + "/data/output/structured_cluster.csv"

    # 调试相关
    DEBUG_LOG_SWITCH = False  # 是否通过print_debug_log打印调试信息
    TAG_DELIM = ""  # tag组件之间分隔符
    INPUT_DELIM = "\t"  # 输入文件的分隔符

    # 标签提取策略
    NEGATIVE_ADVS = ['不']  # 负面副词
    ADV_BLACKLIST = ['但是', '而且']
    TEST_SMALL_DATA = False

    # 标签聚类策略
    MIN_TAG_OCCURANCE = 3
    CLUSTER_MODE = "all"

    # spark流程相关配置
    use_local_input = False
