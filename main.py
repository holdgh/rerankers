from typing import List

from rerankers.models import ColBERTRanker, FlashRankRanker, LLMLayerWiseRanker
from rerankers.models.transformer_ranker import TransformerRanker


def transformer_ranker_rank(query: str, docs: List[str]) -> str:
    ranker = TransformerRanker(r"E:\aiModel\maxkbModel\rerank\bge-reranker-v2-m3")
    results = ranker.rank(query=query, docs=docs)
    print(results)
    return results.__str__()


def colbert_ranker_rank(query: str, docs: List[str]) -> str:
    ranker = ColBERTRanker(r"E:\aiModel\maxkbModel\rerank\bge-reranker-v2-m3")
    results = ranker.rank(query=query, docs=docs)
    print(results)
    return results.__str__()


def flash_ranker_rank(query: str, docs: List[str]) -> str:
    ranker = FlashRankRanker("ms-marco-TinyBERT-L-2-v2")
    results = ranker.rank(query=query, docs=docs)
    print(results)
    return results.__str__()


def llm_layerwise_ranker_rank(query: str, docs: List[str]) -> str:
    ranker = LLMLayerWiseRanker(r"E:\aiModel\maxkbModel\rerank\bge-reranker-v2-m3")
    results = ranker.rank(query=query, docs=docs)
    print(results)
    return results.__str__()


if __name__ == '__main__':
    query = "乱世佳人绝对是一部杰作"
    docs = [
        "早上好，我想飞上天是一首杰作",
        "乱世佳人是一部叙事拙劣的作品",
        "乱世佳人是一部经典作品",
        "红粉佳人是一部普通的文学作品",
        "吃饭两年半睡觉时长的练习生绝对是一个非凡的人",
        "现在的NBA球风偏软现象绝对是肖华的杰作"
    ]
    llm_layerwise_ranker_rank(query, docs)
    # colbert_ranker_rank(query, docs)
    # flash_ranker_rank(query, docs)
    # transformer_ranker_rank(query, docs)
