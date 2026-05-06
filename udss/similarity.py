"""相似度函数 - Similarity Functions

Sim(R, a) = cosine(embed(R), embed(a))
"""

from typing import List, Optional
import numpy as np


class SimilarityFunction:
    """相似度函数

    计算搜索结果与动作的相似度
    """

    def __init__(self):
        pass

    def compute(self, search_result_embedding: np.ndarray, action_embedding: np.ndarray) -> float:
        """计算余弦相似度

        Sim(R, a) = cosine(embed(R), embed(a))

        Args:
            search_result_embedding: 搜索结果嵌入向量
            action_embedding: 动作嵌入向量

        Returns:
            相似度分数 [0, 1]
        """
        norm1 = np.linalg.norm(search_result_embedding)
        norm2 = np.linalg.norm(action_embedding)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(search_result_embedding, action_embedding) / (norm1 * norm2)
        # 映射到 [0, 1]
        return (cosine_sim + 1.0) / 2.0

    def compute_batch(self, search_embeddings: List[np.ndarray], action_embedding: np.ndarray) -> List[float]:
        """批量计算相似度

        Args:
            search_embeddings: 搜索结果嵌入列表
            action_embedding: 动作嵌入

        Returns:
            相似度分数列表
        """
        return [self.compute(emb, action_embedding) for emb in search_embeddings]


class WeightedSimilarityFunction(SimilarityFunction):
    """加权相似度函数

    考虑搜索结果的多维特征（相关性、时效性、权威性）
    """

    def __init__(self, relevance_weight: float = 0.5, freshness_weight: float = 0.3, authority_weight: float = 0.2):
        super().__init__()
        self.relevance_weight = relevance_weight
        self.freshness_weight = freshness_weight
        self.authority_weight = authority_weight

    def compute_with_features(self,
                              search_result_embedding: np.ndarray,
                              action_embedding: np.ndarray,
                              relevance: float,
                              freshness: float,
                              authority: float) -> float:
        """计算带特征的相似度

        Args:
            search_result_embedding: 搜索结果嵌入
            action_embedding: 动作嵌入
            relevance: 相关性 [0, 1]
            freshness: 时效性 [0, 1]
            authority: 权威性 [0, 1]

        Returns:
            加权相似度分数
        """
        base_similarity = self.compute(search_result_embedding, action_embedding)

        # 加权
        feature_weight = (
            self.relevance_weight * relevance +
            self.freshness_weight * freshness +
            self.authority_weight * authority
        )

        return base_similarity * (0.5 + 0.5 * feature_weight)


class EmbeddingBasedSimilarity:
    """基于嵌入的相似度计算

    使用动作的value_vector作为嵌入
    """

    def __init__(self, use_normalized: bool = True):
        self.use_normalized = use_normalized

    def get_action_embedding(self, action_value_vector: np.ndarray) -> np.ndarray:
        """获取动作嵌入

        Args:
            action_value_vector: 动作的价值向量 (10维)

        Returns:
            动作嵌入向量
        """
        if self.use_normalized:
            norm = np.linalg.norm(action_value_vector)
            if norm > 0:
                return action_value_vector / norm
        return action_value_vector

    def get_search_embedding(self, relevance: float, freshness: float, authority: float) -> np.ndarray:
        """从搜索特征生成嵌入

        Args:
            relevance: 相关性
            freshness: 时效性
            authority: 权威性

        Returns:
            伪嵌入向量 (10维)
        """
        # 使用特征生成一个固定的嵌入向量
        embedding = np.array([
            relevance,
            freshness,
            authority,
            (relevance + freshness) / 2,
            (freshness + authority) / 2,
            (authority + relevance) / 2,
            relevance * freshness,
            freshness * authority,
            authority * relevance,
            (relevance + freshness + authority) / 3
        ], dtype=np.float32)

        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


def create_similarity_function() -> SimilarityFunction:
    """工厂函数：创建相似度函数"""
    return SimilarityFunction()


def create_weighted_similarity_function(relevance_weight: float = 0.5,
                                        freshness_weight: float = 0.3,
                                        authority_weight: float = 0.2) -> WeightedSimilarityFunction:
    """工厂函数：创建加权相似度函数"""
    return WeightedSimilarityFunction(relevance_weight, freshness_weight, authority_weight)