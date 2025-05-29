from rerankers.models.ranker import BaseRanker

from flashrank import Ranker, RerankRequest


from typing import Union, List, Optional, Tuple
from rerankers.utils import vprint, prep_docs
from rerankers.results import RankedResults, Result
from rerankers.documents import Document


class FlashRankRanker(BaseRanker):  # flashrank 使用轻量级神经网络模型对初步检索结果进行重新排序，将最相关的结果提升到顶部。
    def __init__(
        self,
        model_name_or_path: str = "ms-marco-TinyBERT-L-2-v2",  # flashrank依赖特定的模型
        verbose: int = 1,
        # cache_dir: str = "./.flashrank_cache",  # 模型文件缓存路径，此处替换为本地路径
        cache_dir: str = r"E:\aiModel\maxkbModel\rerank",  # 模型文件缓存路径，此处替换为本地路径
        **kwargs
    ):
        self.verbose = verbose
        vprint(
            f"Loading model FlashRank model {model_name_or_path}...", verbose=verbose
        )
        """
        flashrank支持的模型列表如下：
        default_model = "ms-marco-TinyBERT-L-2-v2"
        model_file_map = {
            "ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",
            "ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",
            "ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx",
            "rank-T5-flan": "flashrank-rankt5_Q.onnx",
            "ce-esci-MiniLM-L12-v2": "flashrank-ce-esci-MiniLM-L12-v2_Q.onnx",
            "rank_zephyr_7b_v1_full": "rank_zephyr_7b_v1_full.Q4_K_M.gguf",
            "miniReranker_arabic_v1": "miniReranker_arabic_v1.onnx"
        }  # 模型名称与相应模型文件的映射关系
        模型下载地址：'https://huggingface.co/prithivida/flashrank/resolve/main/{模型名称}.zip'
        由于https://huggingface.co访问不通，可替换为https://hf-mirror.com，也即'https://hf-mirror.com/prithivida/flashrank/resolve/main/{模型名称}.zip'
        """
        self.model = Ranker(model_name=model_name_or_path, cache_dir=cache_dir)
        self.ranking_type = "pointwise"

    def tokenize(self, inputs: Union[str, List[str], List[Tuple[str, str]]]):
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)  # 预处理文档【索引、元数据】
        passages = [
            {"id": doc_idx, "text": doc.text} for doc_idx, doc in enumerate(docs)
        ]  # 将文档列表转化为字典【含有索引和文档内容】列表
        # 将query和文档列表组装到既定提示词中，基于flashrank模型推理得到排序结果
        rerank_request = RerankRequest(query=query, passages=passages)
        flashrank_results = self.model.rerank(rerank_request)

        ranked_results = [
            Result(
                document=docs[result["id"]],  # Returns reranked documents.
                score=result["score"],
                rank=idx + 1,
            )
            for idx, result in enumerate(flashrank_results)
        ]
        
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def score(self, query: str, doc: str) -> float:
        rerank_request = RerankRequest(
            query=query, passages=[{"id": "temp_id", "text": doc}]
        )
        flashrank_result = self.model.rerank(rerank_request)
        score = flashrank_result[0]["score"]
        return score
