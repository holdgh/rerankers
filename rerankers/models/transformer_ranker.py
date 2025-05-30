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


class TransformerRanker(BaseRanker):  # 朴素意义的rerank模型推理
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

    def tokenize(self, inputs: Union[str, List[str], List[Tuple[str, str]]]):  # 对输入文本元组列表进行分词编码
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
        docs = prep_docs(docs, doc_ids, metadata)  # 文档预处理【索引及元数据】
        inputs = [(query, doc.text) for doc in docs]  # 将query与文档内容两两配对，以元组形式作为输入。
        # 输入列表得批量化截取处理，得到批量化输入列表【每个元素为一个批次的输入列表】
        # Override self.batch_size if explicitely set
        if batch_size is None:
            batch_size = self.batch_size
        batched_inputs = [
            inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)
        ]
        scores = []
        for batch in batched_inputs:  # 批量处理
            tokenized_inputs = self.tokenize(batch)  # 对当前批次的输入列表进行分词编码
            """
            首先，`model(**tokenized_inputs)`这部分。在PyTorch或TensorFlow中，通常预训练模型接受输入的方式是将分词后的结果作为关键字参数传递。比如，输入的`tokenized_inputs`可能包含`input_ids`、`attention_mask`等键。使用`**`操作符会将字典中的键值对展开为关键字参数，例如`model(input_ids=..., attention_mask=...)`。这样模型就能接收到正确的输入参数。
            
            接下来，`.logits`应该是指模型输出的logits。在分类任务中，模型的最后一层通常是全连接层，输出未经归一化的分数，称为logits。这些logits之后通常会被softmax函数处理，得到概率分布。因此，`.logits`属性就是模型输出的原始分数。
            
            然后，`.squeeze()`是一个PyTorch或NumPy中的方法，用于去除张量中维度大小为1的维度。比如，如果logits的形状是`(batch_size, 1, num_classes)`，使用`squeeze()`之后会变成`(batch_size, num_classes)`。这是为了消除不必要的单一维度，方便后续处理。
            
            综合起来，`batch_scores`应该是模型对当前批次输入数据的每个样本在各个类别上的原始分数，且经过压缩处理后的结果.
            """
            batch_scores = self.model(**tokenized_inputs).logits.squeeze()  # 利用模型直接得到相关性得分，详情见D:\project\AI\rerankers\model-tokenized_inputs-logits-squeeze模型推理结果操作解析.md
            if self.dtype != torch.float32:
                batch_scores = batch_scores.float()
            """
            `batch_scores.detach().cpu().numpy().tolist()` 是一系列将 PyTorch 张量转换为 Python 列表的操作，每个步骤都有特定的目的。下面是详细解释：

            ### **操作步骤解析**
            
            1. **`.detach()`**：
               - **作用**：创建 `batch_scores` 张量的一个**副本**，该副本与原始计算图分离。
               - **目的**：防止梯度计算传播到该张量，节省内存并避免不必要的梯度计算。
               - **使用场景**：在模型评估（推理）阶段使用，此时不需要计算梯度。
            
            2. **`.cpu()`**：
               - **作用**：如果张量在 GPU 上（如 `cuda:0`），将其数据**转移到 CPU 内存**。
               - **目的**：GPU 张量无法直接转换为 NumPy 数组，需要先移动到 CPU。
               - **注意**：如果张量已在 CPU 上，此操作无实际效果。
            
            3. **`.numpy()`**：
               - **作用**：将 PyTorch 张量转换为 **NumPy 数组**。
               - **目的**：NumPy 数组更适合与科学计算库交互，且支持更丰富的操作。
               - **限制**：仅支持 CPU 上的张量转换（因此需要先执行 `.cpu()`）。
            
            4. **`.tolist()`**：
               - **作用**：将 NumPy 数组转换为 **Python 原生列表**。
               - **目的**：列表更易于序列化（如保存为 JSON）、传输（如 API 响应）或与纯 Python 代码交互。
            
            ### **完整流程图示**
            ```mermaid
            graph LR
                A[PyTorch Tensor<br>GPU/CPU] --> B[.detach()<br>分离计算图]
                B --> C[.cpu()<br>转移到CPU]
                C --> D[.numpy()<br>转为NumPy数组]
                D --> E[.tolist()<br>转为Python列表]
            ```
            
            ### **示例说明**
            假设 `batch_scores` 是一个形状为 `(2, 3)` 的张量：
            ```python
            import torch
            
            # 假设在GPU上的原始张量
            batch_scores = torch.tensor([[0.9, 0.1, -0.5], [1.2, 0.3, 0.8]], device='cuda:0', requires_grad=True)
            print("原始张量:", batch_scores)
            ```
            
            执行转换：
            ```python
            result = batch_scores.detach().cpu().numpy().tolist()
            print("转换结果:", result)
            print("类型:", type(result))
            ```
            
            **输出**：
            ```
            原始张量: tensor([[ 0.9000,  0.1000, -0.5000],
                             [ 1.2000,  0.3000,  0.8000]], device='cuda:0', grad_fn=<AddBackward0>)
                             
            转换结果: [[0.9, 0.1, -0.5], [1.2, 0.3, 0.8]]
            类型: <class 'list'>
            ```
            
            ### **典型应用场景**
            1. **模型部署**：
               ```python
               # 将预测结果通过API返回
               import json
               predictions = model(inputs).logits.detach().cpu().numpy().tolist()
               return json.dumps({"scores": predictions})
               ```
            
            2. **结果保存**：
               ```python
               # 将预测分数保存为JSON文件
               import json
               with open("predictions.json", "w") as f:
                   json.dump(batch_scores.detach().cpu().numpy().tolist(), f)
               ```
            
            3. **与其他库集成**：
               ```python
               # 使用scikit-learn计算指标
               from sklearn.metrics import accuracy_score
               predictions = batch_scores.argmax(-1).detach().cpu().numpy().tolist()
               accuracy = accuracy_score(labels, predictions)
               ```
            
            ### **注意事项**
            1. **性能影响**：
               - 频繁的 CPU-GPU 数据传输（`.cpu()`）会降低性能，应尽量减少。
               - 对于大批量数据，建议直接在 GPU 上完成计算后再转换。
            
            2. **内存管理**：
               - `.detach()` 不会释放原始张量，如需释放显存，可手动调用 `del batch_scores`。
            
            3. **替代方案**：
               - 如果只需要数值计算，可保留为 NumPy 数组（省略 `.tolist()`）。
               - 使用 `torch.no_grad()` 上下文管理器替代 `.detach()`：
                 ```python
                 with torch.no_grad():
                     batch_scores = model(inputs).logits
                     result = batch_scores.cpu().numpy().tolist()
                 ```
            
            ### **总结**
            - **核心目的**：将模型输出的预测分数从 PyTorch 张量转换为 Python 原生列表。
            - **关键步骤**：
              1. 分离计算图（`.detach()`）
              2. 转移到 CPU（`.cpu()`）
              3. 转为 NumPy 数组（`.numpy()`）
              4. 转为 Python 列表（`.tolist()`）
            - **适用场景**：模型部署、结果保存、与其他 Python 工具集成等需要脱离 PyTorch 环境的场景。
            """
            batch_scores = batch_scores.detach().cpu().numpy().tolist()  # 将torch张量转化为python列表
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
                    sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)  # 按照相关性得分从大到小排序
                )
            ]
            return RankedResults(results=ranked_results, query=query, has_scores=True, rank_method_name='transformer_ranker')

    @torch.inference_mode()
    def score(self, query: str, doc: str) -> float:
        inputs = self.tokenize((query, doc))
        outputs = self.model(**inputs)
        score = outputs.logits.squeeze().detach().cpu().numpy().astype(float)
        return score
