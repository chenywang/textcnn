# -*- coding: utf-8 -*-
# @Author: disheng
import os

PROJECT_PATH = os.path.abspath(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.pardir))


class DefaultConfig(object):
    """
    default config, 由git托管，一般情况下不用修改
    可以在config目录下新建configs.py, 更新自己的本地配置，会覆盖此处的配置
    未被覆盖的配置将使用此处的配置值
    """
    # 项目路径
    project_path = PROJECT_PATH

    # 切词词典
    user_dict_path = project_path + "/data/dict/user_dict.dict"
    # 词性标注词典
    user_tag_path = project_path + "/data/dict/user_tag.dict"
    # 停用词词典
    stopwords_path = project_path + "/data/dict/user_stopwords.dict"

    # ltp模型路径（可以根据自己下载的数据地址来覆盖改写）
    ltp_model_path = project_path + '/data/ltp_data'
    seg_model_path = project_path + '/data/ltp_data/cws.model'
    pos_model_path = project_path + '/data/ltp_data/pos.model'
    par_model_path = project_path + '/data/ltp_data/parser.model'

    # word2vec
    wordvector_model_path = project_path + "/data/model/word2vec.model"
    word2vec_format_model_path = project_path + "/data/model/word2vec.bin"
    # doc2vec
    docvector_model_path = project_path + "/data/model/doc2vec.model"
    # 情感分析模型
    sentiment_model_path = project_path + "/data/model/sentiment.model"
    # 健康风险评论分析模型
    risk_model_path = project_path + "/data/model/risk.model"
    # 标签聚类模型
    cluster_tag_model_path = project_path + "/data/model/cluster_tag.model"

    # 数据路径
    input_data_path = project_path + "/data/review.csv"
    segged_path = project_path + "/data/review.segged"
    dsr_review_path = project_path + "/data/review_dsr.csv"
    new_dsr_review_path = project_path + "/data/new_review_dsr.csv"
    tag_path = project_path + "/data/output/tags.csv"
    structured_tag_path = project_path + "/data/output/structured_tags.csv"
    tag_counts_path = project_path + "/data/output/tag_counts.csv"
    structured_tag_counts_path = project_path + "/data/output/structured_tag_counts.csv"
    tag_counts_segged_path = project_path + "/data/tag_counts.segged"
    tag_data_path = project_path + "/data/tag_data.npy"
    sentiment_input_path = project_path + "/data/output/tag_counts_oneday.csv"
    sentiment_test_output_path = project_path + "/data/output/sentiment_test.csv"
    sentiment_output_path = project_path + "/data/output/sentiment.csv"
    risk_test_output_path = project_path + "/data/output/risk_test.csv"
    tag_sentiment_model_path = project_path + "/data/model.tag_sentiment_lr.model"

    # spark测试路径
    spark_test_out = project_path + "/data/output/spark_out"
    spark_test_segged = project_path + "/data/spark_review.segged"

    # 聚类数据路径
    cluster_path = project_path + "/data/output/clusters.csv"
    cluster_tag_filter_path = project_path + "/data/output/clusters_tag_filter.csv"
    cluster_tag_path = project_path + "/data/output/cluster_tags.csv"
    cluster_filter_path = project_path + "/data/output/clusters_filter.csv"
    cluster_filter_result_path = project_path + "/data/output/clusters_filter_result.csv"

    # eval路径
    eval_out_root = project_path + "/data/output/eval"
    risky_comments_path = project_path + "/data/risky_comments.csv"

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
