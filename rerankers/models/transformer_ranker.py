from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document


import torch
from typing import Union, List, Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from rerankers.utils import (
    vprint,
    get_device,
    get_dtype,
    prep_docs,
)
from rerankers.results import RankedResults, Result


class TransformerRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 16,
        verbose: int = 1,
        **kwargs,
    ):
        self.verbose = verbose
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        # monoBERT 就是最原始的BERT用法，将查询与段落按照BERT的方式拼接，计算他们之间的相关性分数，论文将这种方式称为 Pointwise re-ranker。阶段 H 1 H_{1} H1 基于 monoBERT 生成的相关性分数，输出 top- k 1 k_{1} k1 个候选 R 1 R_{1} R1
        """
        机器学习中有三种排序模型：
        pointwise: 直接预测每个文档和问题的相关分数。一定程度上缺少了文档之间的排序关系。
        pairwise: 将排序问题转化为对两两文档的比较【类似于llm-sort的思想】。
        listwise: 直接学习文档之间的排序关系。使用每个文档的top-1概率分布作为排序列表，并使用交叉熵损失来优化。
        科普：top-1准确率与top-5准确率
            背景：分类任务中，假设样本有50个类别，通过模型可以得到对应样本属于这50个类别的概率值
        top-1准确率：概率值排名第一的类别为样本的真实类别。对应有top-1错误率。
        top-5准确率：概率值排名在前五的类别中包含样本的真实类别。对应有top-5错误率。
        """
        self.is_monobert = "monobert" in model_name_or_path.lower()
        model_kwargs = kwargs.get("model_kwargs", {})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            **model_kwargs,
        ).to(self.device)
        vprint(f"Loaded model {model_name_or_path}", self.verbose)
        vprint(f"Using device {self.device}.", self.verbose)
        vprint(f"Using dtype {self.dtype}.", self.verbose)
        self.model.eval()
        tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        self.ranking_type = "pointwise"  # 逐个文档处理，计算与问题的相关得分，然后排序
        self.batch_size = batch_size

    def tokenize(self, inputs: Union[str, List[str], List[Tuple[str, str]]]):
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

    @torch.inference_mode()
    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: Optional[int] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)
        inputs = [(query, doc.text) for doc in docs]

        # Override self.batch_size if explicitely set
        if batch_size is None:
            batch_size = self.batch_size
        batched_inputs = [
            inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)
        ]
        scores = []
        for batch in batched_inputs:
            tokenized_inputs = self.tokenize(batch)
            batch_scores = self.model(**tokenized_inputs).logits.squeeze()
            if self.dtype != torch.float32:
                batch_scores = batch_scores.float()
            batch_scores = batch_scores.detach().cpu().numpy().tolist()
            if isinstance(batch_scores, float):  # Handling the case of single score
                scores.append(batch_scores)
            else:
                scores.extend(batch_scores)
        if self.is_monobert: scores = [x[1] - x[0] for x in scores]
        if len(scores) == 1:
            return RankedResults(results=[Result(document=docs[0], score=scores[0])], query=query, has_scores=True)
        else:
            ranked_results = [
                Result(document=doc, score=score, rank=idx + 1)
                for idx, (doc, score) in enumerate(
                    sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                )
            ]
            return RankedResults(results=ranked_results, query=query, has_scores=True)

    @torch.inference_mode()
    def score(self, query: str, doc: str) -> float:
        inputs = self.tokenize((query, doc))
        outputs = self.model(**inputs)
        score = outputs.logits.squeeze().detach().cpu().numpy().astype(float)
        return score
