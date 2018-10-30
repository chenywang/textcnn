#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: disheng


import gensim
import numpy as np

from config import config
from config.configs import PROJECT_PATH


class Word2VecService(object):
    """
    word2vec服务
    """

    def __init__(self):
        self.config = {}
        self.load_config()
        self.load_model()

    def load_config(self):
        """
        读取配置
        :return:
        """
        pass

    def load_model(self):
        """
        加载模型
        :return:
        """
        # 加载word2vec模型
        self.model = gensim.models.Word2Vec.load(PROJECT_PATH + '/model/word2vec.model')


    def release_model(self):
        """
        释放模型
        :return:
        """
        pass

    def get_word_vector(self, word):
        if not isinstance(word, unicode):
            word = word.decode('utf-8')
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            return np.random.random(self.model.vector_size)

    def vector_analysis(self, sentence_list):
        '''
        向量分析，获得指定句子分词后词语的向量
        '''
        for sentence in sentence_list:
            words = self.segmentor.segment(sentence)
            for word in words:
                word = word.decode('utf-8')
                print word
                if word in self.model.wv:
                    print self.model.wv[word]
                else:
                    print 'Null'

    def tag_similarity_analysis(self, tag):
        """
        分析标签相似度
        :param tag:
        :return:
        """
        tag0 = tag[0].decode('utf-8')
        tag1 = tag[1].decode('utf-8')
        print self.model.wv.similarity(tag0, tag1)
        print self.cosine_distance(self.get_word_vector(tag1), self.get_word_vector(tag0))


    @staticmethod
    def cosine_distance(vec1, vec2):
        up_part = np.dot(vec1, vec2)
        down_part = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if down_part == 0.0:
            return 2.0
        else:
            return 1.0 - up_part / down_part


if __name__ == '__main__':
    # word_list = pd.read_csv(config.word2vec_test_path)
    # print word_list
    word2vec_service = Word2VecService()
    print "the vector of 王辰垚", word2vec_service.get_word_vector("王辰垚")
    print "the vector of 萝卜", word2vec_service.get_word_vector("萝卜")
    # word2vec_service.instance_analysis(word_list)
    # word2vec_service.vector_analysis(word_list)
    # word2vec_service.tag_similarity_analysis(['价钱', '价格'])
    word2vec_service.tag_similarity_analysis(['好', '不错'])
    # word2vec_service.tag_similarity_analysis(["垃圾袋好","毯子好"])
    word2vec_service.release_model()
