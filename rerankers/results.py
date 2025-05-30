import math
from typing import List, Optional, Union

from rerankers.documents import Document


class Result:
    def __init__(self, document: Document, score: Optional[float] = None, rank: Optional[int] = None):
        self.document = document
        self.score = score
        self.rank = rank

        if rank is None and score is None:
            raise ValueError("Either score or rank must be provided.")

    def __getattr__(self, item):
        if hasattr(self.document, item):
            return getattr(self.document, item)
        elif item in ["document", "score", "rank"]:
            return getattr(self, item)
        elif item in self.document.attributes:
            return getattr(self.document, item)
        elif item in self.document.metadata:
            return self.document.metadata[item]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __repr__(self) -> str:
        fields = {
            "document": self.document,
            "score": self.score,
            "rank": self.rank,
        }
        field_str = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({field_str})"


class RankedResults:
    def __init__(self, results: List[Result], query: str, has_scores: bool = False, rank_method_name: str = '你猜'):
        self.results = results
        self.query = query
        self.has_scores = has_scores
        self.rank_method_name = rank_method_name
        self.sigmoid_score()  # 将相关性分值通过sigmoid函数映射到(0,1)

    def __iter__(self):
        """Allows iteration over the results list."""
        return iter(self.results)

    def __getitem__(self, index):
        """Allows indexing to access results directly."""
        return self.results[index]

    def results_count(self) -> int:
        """Returns the total number of results."""
        return len(self.results)

    def __str__(self):  # 重写当前类实例的打印结果，便于直观打印
        str_res = 'rank算法名称：' + self.rank_method_name + '\nquery信息：' + self.query + '\n排序结果：\n'
        for result in self.results[:-1]:
            str_res = str_res + '文档内容：' + result.document.text + '，排名：' + str(result.rank) + '，相关性得分：' + str(
                result.score) + '\n'
        str_res = str_res + '文档内容：' + self.results[-1].document.text + '，排名：' + str(
            self.results[-1].rank) + '，相关性得分：' + str(self.results[-1].score)
        return str_res

    def sigmoid_score(self):
        def sigmoid_func(x):
            return 1 / (1 + math.e ** (-1 * x))
        for res in self.results:
            res.score = sigmoid_func(res.score)

    def top_k(self, k: int) -> List[Result]:
        """Returns the top k results based on the score, if available, or rank."""
        if self.has_scores:
            return sorted(
                self.results,
                key=lambda x: x.score if x.score is not None else float("-inf"),
                reverse=True,
            )[:k]
        else:
            return sorted(
                self.results,
                key=lambda x: x.rank if x.rank is not None else float("inf"),
            )[:k]

    def get_score_by_docid(self, doc_id: Union[int, str]) -> Optional[float]:
        """Fetches the score of a result by its doc_id using a more efficient approach."""
        result = next((r for r in self.results if r.document.doc_id == doc_id), None)
        return result.score if result else None

    def get_result_by_docid(self, doc_id: Union[int, str]) -> Optional[Result]:
        """Fetches a result by its doc_id using a more efficient approach."""
        result = next((r for r in self.results if r.document.doc_id == doc_id), None)
        return result if result else None

    def __repr__(self) -> str:
        fields = {
            "results": self.results,
            "query": self.query,
            "has_scores": self.has_scores,
        }
        field_str = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({field_str})"
