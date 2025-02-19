import clip
import numpy as np
import torch
from typing import List
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))
from gpt import GPTAPI

def check_hypernymy(word1, word2, pos='n'):
    synsets1 = wn.synsets(word1, pos=pos)
    synsets2 = wn.synsets(word2, pos=pos)

    for syn1 in synsets1:
        for syn2 in synsets2:
            # 检查 word1 是否是 word2 的下位词
            if syn2 in syn1.lowest_common_hypernyms(syn2):
                if syn1 in syn2.hyponyms():
                    return f"'{word1}' 是 '{word2}' 的下位词"
                elif syn2 in syn1.hyponyms():
                    return f"'{word2}' 是 '{word1}' 的下位词"
            # 检查间接关系（递归遍历）
            if syn1.path_similarity(syn2) is not None:
                if syn1 in syn2.hyponyms():
                    return f"'{word1}' 是 '{word2}' 的下位词"
                elif syn2 in syn1.hyponyms():
                    return f"'{word2}' 是 '{word1}' 的下位词"
    return "未找到明确关系"

class HierCluster:
    def __init__(self):
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.gptapi = GPTAPI()

    def __clip_embedding(self, keywords: List[str]):
        model, _ = clip.load('ViT-B/32', self.device)

        # Tokenize word
        tokenized_texts = torch.cat([clip.tokenize(f"{c}") for c in keywords]).to(self.device)

        # Clip text embedding
        text_embeddings = model.encode_text(tokenized_texts).cpu().detach().numpy()

        return text_embeddings
        print(text_embeddings)

    def __embedding_keywords(self, keywords: List[str], embedding: str = 'clip'):
        match embedding:
            case 'clip':
                return self.__clip_embedding(keywords)
            case _:
                raise ValueError('Not implemented')
    
    def clustering_in_test(self, keywords: List[str], embedding: str = 'clip'):
        '''
        TODO: Delete after the correct one finshed
        '''
        return np.ones(len(keywords))

    def __has_hypernym_relation(self, word1, word2):
        system_sentence = 'Given two words, determine if there is a hyponym-hypernym relationship between them. A hyponym is a more specific term, and a hypernym is a more general term. Answer with "yes" if there is a hyponym-hypernym relationship, otherwise answer with "no".'

        message = [
            {"role": "system", "content": system_sentence},
            {"role": "user", "content": f"{word1}, {word2}"}
        ]

        res = self.gptapi.call_gpt(message)
        return res

    def __has_hypernym_relation_in_cluster(self, cluster1, cluster2):
        for i in cluster1:
            for j in cluster2:
                if self.hypernym_matrix[i][j]:
                    return True

        return False
    
    def __init_hypernym_matrix(self, keywords):
        # matrix = np.zeros((len(keywords), len(keywords)), dtype=int)

        # for i in range(len(keywords)):
        #     for j in range(i+1, len(keywords)):
        #         is_hypernym = int(self.__has_hypernym_relation(keywords[i], keywords[j]) == 'yes')
        #         matrix[i][j] = is_hypernym
        #         matrix[j][i] = is_hypernym

        hypernym_matrix = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]

        # hypernym_matrix = [
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0],
        #     [1, 0, 1, 1, 0, 0, 0, 0, 0],
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
        # ]

        return hypernym_matrix

    def __calculate_distance(self, cluster1, cluster2):
        distance_sum = 0
        count = 0

        for i in cluster1:
            for j in cluster2:
                distance_sum += self.distance_matrix[i, j]
                count += 1
        
        return distance_sum / count


    def fit(self, keywords: List[str], embedding: str = 'clip'):
        # Get embeddings
        embeddings = self.__embedding_keywords(keywords, embedding)

        # Init cluster
        clusters = [[i] for i in range(len(keywords))]

        # Init distance matrix
        self.distance_matrix = pairwise_distances(embeddings)

        # Init hypernym_relation matrix
        self.hypernym_matrix = self.__init_hypernym_matrix(keywords)

        while len(clusters) > 1:
            # Find the closest two cluster
            min_distance = np.inf
            closest_pair = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self.__calculate_distance(clusters[i], clusters[j])

                    if distance < min_distance:
                        if self.__has_hypernym_relation_in_cluster(clusters[i], clusters[j]):
                            min_distance = distance
                            closest_pair = (i, j)

            # Not able to merge
            if closest_pair is None:
                break

            # Megre two cluster
            i, j = closest_pair
            new_cluster = clusters[i] + clusters[j]
            clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in [i, j]]
            clusters.append(new_cluster)
        
        for cluster in clusters:
            print([keywords[idx] for idx in cluster])
            
if __name__ == '__main__':
    test_keywords_1 = ['beach', 'seagull', 'boat', 'water', 'lake', 'sunset', 'sky']
    test_keywords_2 = ['bamboo forest', 'forest', 'woods', 'rainforest', 'garden', 'tree', 'tree stump', 'balcony', 'prey']
    
    hc = HierCluster()

    hc.fit(test_keywords_1)