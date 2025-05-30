import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from typing import Union, List, Optional
from rerankers.utils import vprint, get_device, get_dtype, prep_docs
from rerankers.results import RankedResults, Result

PROMPTS = {
    "BAAI/bge-reranker-v2.5-gemma2-lightweight": "Given a query A and a passage B, determine whether the passage "
                                                 "contains an answer to the query by providing a prediction of either "
                                                 "'Yes' or 'No'.",
    "default": "Given a query A and a passage B, determine whether the passage contains an answer to the query by "
               "providing a prediction of either 'Yes' or 'No'.",
}  # 提示词设置【如果不是模型BAAI/bge-reranker-v2.5-gemma2-lightweight，则采用默认提示词】【从提示词中可以看出，模型推理是对一个query和一个文档内容的相关性进行评判】

DEFAULT_PARAMS = {
    "default": {},
    "BAAI/bge-multilingual-gemma2": {},
    "BAAI/bge-reranker-v2-gemma": {},
    "BAAI/bge-reranker-v2-minicpm-layerwise": {"cutoff_layers": [28]},
    "BAAI/bge-reranker-v2.5-gemma2-lightweight": {
        "cutoff_layers": [28],
        "compress_ratio": 2,
        "compress_layer": [24, 40],
    },
}


class LLMLayerWiseRanker(BaseRanker):
    def __init__(
            self,
            model_name_or_path: str = "BAAI/bge-reranker-v2.5-gemma2-lightweight",
            max_sequence_length: int = 512,
            dtype: Optional[Union[str, torch.dtype]] = None,
            device: Optional[Union[str, torch.device]] = None,
            batch_size: int = 16,
            verbose: int = 1,
            prompt: Optional[str] = None,
            cutoff_layers: Optional[List[int]] = None,
            compress_ratio: Optional[int] = None,
            compress_layer: Optional[List[int]] = None,
            **kwargs,
    ):
        self.verbose = verbose
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size

        vprint(
            f"Loading model {model_name_or_path}, this might take a while...",
            self.verbose,
        )
        vprint(f"Using device {self.device}.", self.verbose)
        vprint(f"Using dtype {self.dtype}.", self.verbose)
        tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
        tokenizer_trust_remote_code = tokenizer_kwargs.pop("trust_remote_code", True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=tokenizer_trust_remote_code,
            **tokenizer_kwargs,
        )
        self.max_sequence_length = max_sequence_length
        self.tokenizer.model_max_length = self.max_sequence_length
        self.tokenizer.padding_side = "right"
        model_kwargs = kwargs.get("model_kwargs", {})
        model_trust_remote_code = model_kwargs.pop("trust_remote_code", True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=model_trust_remote_code,
            torch_dtype=self.dtype,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

        # Create params dict based on specified values or defaults
        params = {}
        if cutoff_layers is not None:
            params["cutoff_layers"] = cutoff_layers
        if compress_ratio is not None:
            params["compress_ratio"] = compress_ratio
        if compress_layer is not None:
            params["compress_layer"] = compress_layer
        if not params:
            params = DEFAULT_PARAMS.get(model_name_or_path, DEFAULT_PARAMS["default"])
        self.params = params

        self.prompt = prompt
        if self.prompt is None:  # 初始化时，如果未传提示词参数，根据传入的模型名称【模型路径】设置提示词
            self.prompt = PROMPTS.get(model_name_or_path, PROMPTS["default"])

    def _get_inputs(self, pairs, max_sequence_length: int):  # 分词算法：提示词分词、换行符分词、query分词、文档内容分词；将query分词结果和文档内容分词结果进行融合；拼接分词【融合分词结果、换行符分词、提示词分词】组装分词结果【对于一对query和文档的元组】
        prompt = self.prompt
        sep = "\n"
        prompt_inputs = self.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]  # 对提示词进行分词编码，并获取编码结果的input_ids【分词编码结果包含input_ids【分词id向量】和attention_masks掩码标识向量】
        sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]  # 对换行符进行分词编码，并获取编码结果的input_ids
        inputs = []
        for query, passage in pairs:  # 提取元组中的query和文档内容
            query_inputs = self.tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_sequence_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = self.tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_sequence_length,
                truncation=True,
            )
            """
            在 Hugging Face Transformers 库中，self.tokenizer.prepare_for_model 是一个用于将分词结果转换为模型输入格式的方法。它主要完成以下关键任务：
            核心功能解析
            
                添加特殊标记（Special Tokens）：
            
                    在序列开头添加 [CLS] 或 <s>（取决于模型）
            
                    在序列结尾添加 [SEP] 或 </s>
            
                    在句子对任务中添加分隔标记
            
                生成注意力掩码（Attention Mask）：
            
                    创建二进制掩码，标识哪些是真实 token（1），哪些是填充 token（0）
            
                填充序列（Padding）：
            
                    将序列填充到指定长度（max_length）或批次中的最大长度
            
                    默认在右侧填充（可配置为左侧填充）
            
                生成 token 类型 ID（Token Type IDs）：
            
                    对于句子对任务（如问答），区分第一句和第二句（0 和 1）
            
                转换为张量格式：
            
                    将结果转换为 PyTorch/TensorFlow 张量（通过 return_tensors 参数指定）
            """
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs["input_ids"],
                sep_inputs + passage_inputs["input_ids"],
                truncation="only_second",
                max_length=max_sequence_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])  # 手动设定掩码为1，标识所有分词皆可用【非填充】
            inputs.append(item)

        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_sequence_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    @torch.inference_mode()
    def rank(
            self,
            query: str,
            docs: Union[str, List[str], Document, List[Document]],
            doc_ids: Optional[Union[List[str], List[int]]] = None,
            metadata: Optional[List[dict]] = None,
            batch_size: Optional[int] = None,
            max_sequence_length: Optional[int] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)  # 文档预处理【索引、元数据】
        pairs = [(query, doc.text) for doc in docs]  # query与文档内容配对【元组】
        # (query,文档内容)元组列表批量处理
        # Override self.batch_size if explicitly set
        if batch_size is None:
            batch_size = self.batch_size

        # Same for max_sequence_length
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length

        batched_pairs = [
            pairs[i: i + batch_size] for i in range(0, len(pairs), batch_size)
        ]
        scores = []  # 初始化得分列表

        for batch in batched_pairs:
            # 当前批次的文本内容分词编码
            inputs = self._get_inputs(batch, max_sequence_length=max_sequence_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # 模型推理
            outputs = self.model(**inputs, return_dict=True, **self.params)
            all_scores = [
                scores[:, -1]
                .view(
                    -1,
                )
                .float()
                for scores in outputs[0]
            ]
            batch_scores = all_scores[-1].cpu().numpy().tolist()

            scores.extend(batch_scores)  # 收集模型推理得到的评分结果【由此可见当前机制没有考虑文档列表中全局文档之间的影响，仅是计算一批收集一批】

        ranked_results = [
            Result(document=doc, score=score, rank=idx + 1)
            for idx, (doc, score) in enumerate(
                sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            )
        ]  # 封装rank结果
        return RankedResults(results=ranked_results, query=query, has_scores=True, rank_method_name='llm_layerwise_ranker')

    @torch.inference_mode()
    def score(self, query: str, doc: str) -> float:
        inputs = self._get_inputs(
            [(query, doc)], max_sequence_length=self.max_sequence_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, return_dict=True, **self.params)
        all_scores = [
            scores[:, -1]
            .view(
                -1,
            )
            .float()
            for scores in outputs[0]
        ]
        score = all_scores[-1].item()

        return score
