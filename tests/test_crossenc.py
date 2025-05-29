from unittest.mock import patch
import torch
from rerankers import Reranker
from rerankers.models import ColBERTRanker, FlashRankRanker
from rerankers.models.transformer_ranker import TransformerRanker
from rerankers.results import Result, RankedResults
from rerankers.documents import Document

@patch("rerankers.models.transformer_ranker.TransformerRanker.rank")  # 这里会将mock_rank赋予TransformerRanker.rank方法的返回值
def test_transformer_ranker_rank(mock_rank):
    query = "Gone with the wind is an absolute masterpiece"  # 乱世佳人绝对是一部杰作
    docs = [
        "Gone with the wind is a masterclass in bad storytelling.",  # 乱世佳人是一部叙事拙劣的作品
        "Gone with the wind is an all-time classic",  # 乱世佳人是一部经典作品
    ]
    expected_results = RankedResults(
        results=[
            Result(
                document=Document(
                    doc_id=1, text="Gone with the wind is an all-time classic"
                ),
                score=1.6181640625,
                rank=1,
            ),
            Result(
                document=Document(
                    doc_id=0,
                    text="Gone with the wind is a masterclass in bad storytelling.",
                ),
                score=0.88427734375,
                rank=2,
            ),
        ],
        query=query,
        has_scores=True,
    )
    mock_rank.return_value = expected_results
    # ranker = TransformerRanker("mixedbread-ai/mxbai-rerank-xsmall-v1")
    ranker = TransformerRanker(r"E:\aiModel\maxkbModel\rerank\bge-reranker-v2-m3")
    results = ranker.rank(query=query, docs=docs)
    assert results == expected_results


@patch("rerankers.models.colbert_ranker.ColBERTRanker.rank")
def test_colbert_ranker_rank(mock_rank):
    query = "Gone with the wind is an absolute masterpiece"  # 乱世佳人绝对是一部杰作
    docs = [
        "Gone with the wind is a masterclass in bad storytelling.",  # 乱世佳人是一部叙事拙劣的作品
        "Gone with the wind is an all-time classic",  # 乱世佳人是一部经典作品
    ]
    expected_results = RankedResults(
        results=[
            Result(
                document=Document(
                    doc_id=1, text="Gone with the wind is an all-time classic"
                ),
                score=1.6181640625,
                rank=1,
            ),
            Result(
                document=Document(
                    doc_id=0,
                    text="Gone with the wind is a masterclass in bad storytelling.",
                ),
                score=0.88427734375,
                rank=2,
            ),
        ],
        query=query,
        has_scores=True,
    )
    mock_rank.return_value = expected_results
    ranker = ColBERTRanker(r"E:\aiModel\maxkbModel\rerank\bge-reranker-v2-m3")
    results = ranker.rank(query=query, docs=docs)
    assert results == expected_results


def test_flash_ranker_rank():
    query = "Gone with the wind is an absolute masterpiece"  # 乱世佳人绝对是一部杰作
    docs = [
        "Gone with the wind is a masterclass in bad storytelling.",  # 乱世佳人是一部叙事拙劣的作品
        "Gone with the wind is an all-time classic",  # 乱世佳人是一部经典作品
    ]
    # 注意flashrank支持特定模型

    """
    flashrank支持的模型列表如下：
    default_model = "ms-marco-TinyBERT-L-2-v2"
    model_file_map = {
        "ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",  # 有相关性，但对语义相似性效果一般
        "ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",  # 有相关性，但对语义相似性效果一般
        "ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx",  # 有相关性，但对语义相似性效果一般
        "rank-T5-flan": "flashrank-rankt5_Q.onnx",  # 相关性效果较差，且语义相似性效果较差
        "ce-esci-MiniLM-L12-v2": "flashrank-ce-esci-MiniLM-L12-v2_Q.onnx",  # 相关性一般，但语义相似性效果较上述模型稍好
        "rank_zephyr_7b_v1_full": "rank_zephyr_7b_v1_full.Q4_K_M.gguf",  # 模型文件较大，暂未下载
        "miniReranker_arabic_v1": "miniReranker_arabic_v1.onnx"  # 效果最差
    }
    """
    ranker = FlashRankRanker("miniReranker_arabic_v1")
    results = ranker.rank(query=query, docs=docs)
    print(results)