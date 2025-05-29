"""Code from HotchPotch's JQaRa repository: https://github.com/hotchpotch/JQaRA/blob/main/evaluator/reranker/colbert_reranker.py
Modifications include packaging into a BaseRanker, dynamic query/doc length and batch size handling."""

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer
from typing import List, Optional, Union
from math import ceil

from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from rerankers.results import RankedResults, Result
from rerankers.utils import vprint, get_device, get_dtype, prep_docs


def _insert_token(
    output: dict,
    insert_token_id: int,
    insert_position: int = 1,
    token_type_id: int = 0,
    attention_value: int = 1,
):  # 将insert_token_id【分情况创建新元素】插入到output键值对的值中相应insert_position位置处
    """
    Inserts a new token at a specified position into the sequences of a tokenized representation.

    This function takes a dictionary containing tokenized representations
    (e.g., 'input_ids', 'token_type_ids', 'attention_mask') as PyTorch tensors,
    and inserts a specified token into each sequence at the given position.
    This can be used to add special tokens or other modifications to tokenized inputs.

    Parameters:
    - output (dict): A dictionary containing the tokenized representations. Expected keys
                     are 'input_ids', 'token_type_ids', and 'attention_mask'. Each key
                     is associated with a PyTorch tensor.
    - insert_token_id (int): The token ID to be inserted into each sequence.
    - insert_position (int, optional): The position in the sequence where the new token
                                       should be inserted. Defaults to 1, which typically
                                       follows a special starting token like '[CLS]' or '[BOS]'.
    - token_type_id (int, optional): The token type ID to assign to the inserted token.
                                     Defaults to 0.
    - attention_value (int, optional): The attention mask value to assign to the inserted token.
                                        Defaults to 1.

    Returns:
    - updated_output (dict): A dictionary containing the updated tokenized representations,
                             with the new token inserted at the specified position in each sequence.
                             The structure and keys of the output dictionary are the same as the input.
    """
    updated_output = {}  # 初始化更新后的分词编码结果【更新前为入参output】
    for key in output:
        updated_tensor_list = []  # 初始化更新后的tensor列表
        for seqs in output[key]:
            if len(seqs.shape) == 1:
                seqs = seqs.unsqueeze(0)  # 在第0维上对seqs增加维度，例如：seqs原来为12，执行后变为(1,12)
            for seq in seqs:
                # 基于插入位置，将序列一分为二
                first_part = seq[:insert_position]
                second_part = seq[insert_position:]
                new_element = (
                    torch.tensor([insert_token_id])
                    if key == "input_ids"
                    else torch.tensor([token_type_id])
                )  # 基于output中的键构造tensor新元素【input_ids时，采用torch.tensor([insert_token_id])；其他情况采用torch.tensor([token_type_id])】
                # 对于output中的attention_mask，采用torch.tensor([attention_value]作为新元素
                if key == "attention_mask":
                    new_element = torch.tensor([attention_value])
                updated_seq = torch.cat((first_part, new_element, second_part), dim=0)  # 将新元素拼接在第一部分和第二部分之间
                updated_tensor_list.append(updated_seq)
        updated_output[key] = torch.stack(updated_tensor_list)  # 整合更新后的值到相应的键
    return updated_output


def _colbert_score(q_reps, p_reps, q_mask: torch.Tensor, p_mask: torch.Tensor):  # 计算查询与文档的向量化表示相关得分
    # calc max sim
    # base code from: https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/modeling.py

    # Assert that all q_reps are at least as long as the query length
    assert (
        q_reps.shape[1] >= q_mask.shape[1]
    ), f"q_reps should have at least {q_mask.shape[1]} tokens, but has {q_reps.shape[1]}"
    """
    爱因斯坦张量计算规则：torch.einsum("qin,pjn->qipj", q_reps, p_reps)
    1. 输入张量维度
    q_reps：形状为 (Q, I, N)，索引为 qin。
    p_reps：形状为 (P, J, N)，索引为 pjn。
    2. 操作解析
    求和索引 n：两个张量在最后一个维度 N 上进行点积（对应位置相乘后求和）。
    保留索引 q, i, p, j：不参与求和，直接展开为输出张量的四个维度。
    3. 输出张量
    形状：(Q, I, P, J)。
    每个元素的值： output[q,i,p,j] = sum{n从0到N-1}(q_reps[q,i,n]*p_reps[p,j,n])
    4. 示例
    假设：
    
    q_reps 形状为 (2, 3, 4)，表示：
    
    2 个查询组（Q=2）。
    每个组有 3 个查询项（I=3）。
    每个项的特征维度为 4（N=4）。
    p_reps 形状为 (5, 6, 4)，表示：
    
    5 个文档组（P=5）。
    每个组有 6 个文档项（J=6）。
    每个项的特征维度为 4（N=4）。
    输出 output 形状为 (2, 3, 5, 6)，含义如下：
    
    output[q, i, p, j]：第 q 个查询组中的第 i 个查询项与第 p 个文档组中的第 j 个文档项的特征点积。
    5. 数学意义
    该操作计算了 所有查询项与文档项之间的相似度矩阵，常用于以下场景：
    
    注意力机制：计算查询（Query）与键（Key）的相似度得分。
    推荐系统：用户特征与物品特征的交互得分。
    信息检索：查询与文档的匹配分数。
    """
    token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)  # 1*16*128与2*17*128进行张量计算，得到1*16*2*17
    token_scores = token_scores.masked_fill(p_mask.unsqueeze(0).unsqueeze(0) == 0, -1e4)  # p_mask.unsqueeze(0).unsqueeze(0)`这部分操作会将`p_mask`的维度扩展。`unsqueeze(0)`的作用是在指定位置增加一个维度，这里连续调用了两次`unsqueeze(0)`，所以原来的2×17会变成1×1×2×17，以使得维数1×1×2×17与token_scores的维数1×16×2×17对齐。当前语句的作用：对于p_mask中元素为0的位置(k,l)，对于所有的q和i，对应将token_scores中的(q,i,k,l)的元素替换为-10000。
    scores, _ = token_scores.max(-1)  # token_scores的维数为1*16*2*17，这里对最后一维取最大值，得到1*16*2
    scores = scores.sum(1) / q_mask.sum(-1, keepdim=True)  # 将scores沿着第二个维度求和，也即1*16*2中的16求和，得到1*2；将q_mask沿着最后一个维度求和，也即1*16中的16【由于q_mask表示掩码，这里求和相当于求query中真实的【非填充】词的个数】，得到1*1。最终得到结果scores为query相对于文档列表【这里是两个文档】的相关性得分
    return scores


class ColBERTModel(BertPreTrainedModel):  # ColBERT提出了一种新的基于后期交互范式的用于估计查询q和文档d的相关性。
    def __init__(self, config, verbose: int):
        super().__init__(config)
        self.bert = BertModel(config)
        self.verbose = verbose
        # TODO: Load from artifact.metadata
        if "small" in config._name_or_path:
            linear_dim = 96
        else:
            linear_dim = 128
        vprint(f"Linear Dim set to: {linear_dim} for downcasting", self.verbose)
        self.linear = nn.Linear(config.hidden_size, linear_dim, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always output hidden states
        )

        sequence_output = outputs[0]

        return self.linear(sequence_output)

    def _encode(self, texts: list[str], insert_token_id: int, is_query: bool = False):
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length - 1,  # for insert token
            truncation=True,
        )
        encoding = _insert_token(encoding, insert_token_id)  # type: ignore

        if is_query:
            mask_token_id = self.tokenizer.mask_token_id

            new_encodings = {"input_ids": [], "attention_mask": []}

            for i, input_ids in enumerate(encoding["input_ids"]):
                original_length = (
                    (input_ids != self.tokenizer.pad_token_id).sum().item()
                )

                # Calculate QLEN dynamically for each query
                if original_length % 32 <= 8:
                    QLEN = original_length + 8
                else:
                    QLEN = ceil(original_length / 32) * 32

                if original_length < QLEN:
                    pad_length = QLEN - original_length
                    padded_input_ids = input_ids.tolist() + [mask_token_id] * pad_length
                    padded_attention_mask = (
                        encoding["attention_mask"][i].tolist() + [0] * pad_length
                    )
                else:
                    padded_input_ids = input_ids[:QLEN].tolist()
                    padded_attention_mask = encoding["attention_mask"][i][
                        :QLEN
                    ].tolist()

                new_encodings["input_ids"].append(padded_input_ids)
                new_encodings["attention_mask"].append(padded_attention_mask)

            for key in new_encodings:
                new_encodings[key] = torch.tensor(
                    new_encodings[key], device=self.device
                )

            encoding = new_encodings

        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        return encoding

    def _query_encode(self, query: list[str]):
        return self._encode(query, self.query_token_id, is_query=True)

    def _document_encode(self, documents: list[str]):
        return self._encode(documents, self.document_token_id)

    def _to_embs(self, encoding) -> torch.Tensor:
        with torch.inference_mode():
            # embs = self.model(**encoding).last_hidden_state.squeeze(1)
            embs = self.model(**encoding)
        if self.normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_encoding = self._query_encode([query])
        documents_encoding = self._document_encode(documents)
        query_embeddings = self._to_embs(query_encoding)
        document_embeddings = self._to_embs(documents_encoding)
        scores = (
            _colbert_score(
                query_embeddings,
                document_embeddings,
                query_encoding["attention_mask"],
                documents_encoding["attention_mask"],
            )
            .cpu()
            .tolist()[0]
        )
        return scores


class ColBERTRanker(BaseRanker):  # 类似于基于embedding模型计算相似度得分。分别对query和文档列表进行分词编码embedding化，然后基于既定评分规则计算query与文档列表的相关性得分
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: int = 1,
        query_token: str = "[unused0]",  # 查询预留token
        document_token: str = "[unused1]",  # 文档内容预留token
        **kwargs,
    ):
        self.verbose = verbose
        self.device = get_device(device, self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size  # 对文本分词编码进行embedding处理时的批量参数
        vprint(
            f"Loading model {model_name}, this might take a while...",
            self.verbose,
        )
        tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model_kwargs = kwargs.get("model_kwargs", {})
        self.model = (
            ColBERTModel.from_pretrained(
                model_name, 
                verbose=self.verbose, 
                **model_kwargs
            )
            .to(self.device)
            .to(self.dtype)
        )  # 初始化时，此处要加载rerank模型，电脑资源不足时，导致程序运行较慢
        self.model.eval()
        self.query_max_length = 32  # Lower bound
        self.doc_max_length = (
            self.model.config.max_position_embeddings - 2
        )  # Upper bound文本编码上界
        self.query_token_id: int = self.tokenizer.convert_tokens_to_ids(query_token)  # type: ignore
        self.document_token_id: int = self.tokenizer.convert_tokens_to_ids(
            document_token
        )  # type: ignore
        self.normalize = True

    def rank(
        self,
        query: str,
        docs: Union[Document, str, List[Document], List[str]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)  # 文档对象化处理，设置文档id和元数据

        scores = self._colbert_rank(query, [d.text for d in docs])  # 对query和文档列表先后做分词编码--rerank模型推理产生embedding向量【例如：1*16*128，2*17*128】--基于爱因斯坦张量计算相关性得分规则计算得分
        ranked_results = [
            Result(document=doc, score=score, rank=idx + 1)  # 封装rerank文档结果
            for idx, (doc, score) in enumerate(
                sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)  # 按照相关性得分降序排列
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def score(self, query: str, doc: str) -> float:
        scores = self._colbert_rank(query, [doc])
        return scores[0] if scores else 0.0

    @torch.inference_mode()
    def _colbert_rank(
        self,
        query: str,
        docs: List[str],
    ) -> List[float]:  # 计算查询与文档内容之间的相关性得分
        query_encoding = self._query_encode([query])  # 对query进行分词编码
        documents_encoding = self._document_encode(docs)  # 对文档进行分词编码
        # 对分词编码进行向量化
        query_embeddings = self._to_embs(query_encoding)  # 1*16*128的tensor
        document_embeddings = self._to_embs(documents_encoding)  # 2*17*128的tensor
        scores = (
            _colbert_score(
                query_embeddings,  # 1*16*128
                document_embeddings,  # 2*17*128
                query_encoding["attention_mask"],  # 1*16
                documents_encoding["attention_mask"],  # 2*17
            )  # 基于向量化表示和掩码向量计算相关得分
            .cpu()
            .tolist()[0]
        )
        return scores

    def _query_encode(self, query: list[str]):  # 对查询进行编码
        return self._encode(
            query, self.query_token_id, max_length=self.doc_max_length, is_query=True
        )

    def _document_encode(self, documents: list[str]):  # 先进行预编码，获取最大分词编码长度【确保在模型的self.model.config.max_position_embeddings - 2以内】，然后基于该最大分词编码长度进行正式的编码处理
        tokenized_doc_lengths = [
            len(
                self.tokenizer.encode(
                    doc, max_length=self.doc_max_length, truncation=True
                )
            )
            for doc in documents
        ]  # 对文档列表进行初始【max_length=self.doc_max_length】的分词编码，获取对应的分词编码长度列表
        max_length = max(tokenized_doc_lengths)  # 获取文档列表的分词编码最大长度
        # 基于最大长度进行32整数倍意义下的向上取整
        max_length = (
            ceil(max_length / 32) * 32
        )  # Round up to the nearest multiple of 32
        # 确保文档编码最大长度在模型的self.model.config.max_position_embeddings - 2以内
        max_length = max(
            max_length, self.query_max_length
        )  # Ensure not smaller than query_max_length
        max_length = int(
            min(max_length, self.doc_max_length)
        )  # Ensure not larger than doc_max_length
        return self._encode(documents, self.document_token_id, max_length)

    def _encode(
        self,
        texts: list[str],
        insert_token_id: int,
        max_length: int,
        is_query: bool = False,
    ):  # 对文本列表批量编码
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=max_length - 1,  # for insert token 输入token最大长度
            truncation=True,
        )
        # 基于预留token的分词id更新encoding
        encoding = _insert_token(encoding, insert_token_id)  # type: ignore
        # query文本的特殊处理【动态长度填充/截断编码】
        if is_query:  # 对于query，进行编码长度【比较的标准，动态设置，见QLEN】填充/截断处理
            mask_token_id = self.tokenizer.mask_token_id  # 获取当前分词器的mask_token分词id

            new_encodings = {"input_ids": [], "attention_mask": []}

            for i, input_ids in enumerate(encoding["input_ids"]):
                original_length = (
                    (input_ids != self.tokenizer.pad_token_id).sum().item()  # 统计input_ids中不等于当前分词器pad_token【填充标记，输入文本长度不够时，填充pad_token】分词id的个数
                )

                # Calculate QLEN dynamically for each query
                # 两种情况的QLEN长度相差最多在8左右
                if original_length % 16 <= 8:  # 取余16不足8时，加8
                    QLEN = original_length + 8
                else:  # 取余16超过8时，除以16再向上取整，然后乘以16
                    QLEN = ceil(original_length / 16) * 16

                if original_length < QLEN:  # input_ids中的分词id长度不足QLEN时
                    pad_length = QLEN - original_length  # 计算不足长度，也即待填充长度
                    padded_input_ids = input_ids.tolist() + [mask_token_id] * pad_length  # 用当前分词器的mask_token_id按照不足长度的个数追加到input_ids后面
                    padded_attention_mask = (
                        encoding["attention_mask"][i].tolist() + [0] * pad_length
                    )  # attention_mask处理，对于填充为设置0标记
                else:  # input_ids中的分词id长度超过QLEN时，截断处理【从前往后只保留QLEN长度的值】
                    padded_input_ids = input_ids[:QLEN].tolist()
                    padded_attention_mask = encoding["attention_mask"][i][
                        :QLEN
                    ].tolist()
                # 收集新编码结果
                new_encodings["input_ids"].append(padded_input_ids)
                new_encodings["attention_mask"].append(padded_attention_mask)
            # 编码结果tensor化处理
            for key in new_encodings:
                new_encodings[key] = torch.tensor(
                    new_encodings[key], device=self.device
                )
            # 采纳新编码结果
            encoding = new_encodings
        # 将编码结果写入CPU/gpu
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        return encoding

    def _to_embs(self, encoding) -> torch.Tensor:  # 将文本编码转化为embedding向量，用以后续按照相关评分规则计算评分
        with torch.inference_mode():
            batched_embs = []
            for i in range(0, encoding["input_ids"].size(0), self.batch_size):
                batch_encoding = {
                    key: val[i : i + self.batch_size] for key, val in encoding.items()
                }  # 获取当前批次的分词编码信息，形如：{'attention_mask':tensor([[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]]),'input_ids':tensor([[0,3,2016,86,678,70,32382,83,142,89931,31347,152667,2,250001,250001,250001]])}
                batch_embs = self.model(**batch_encoding)  # 利用rerank模型进行推理，得到1*16*128的tensor
                batched_embs.append(batch_embs)  # 收集embedding结果
            embs = torch.cat(batched_embs, dim=0)  # 拼接组装所有批次
        if self.normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs  # 最终结果为1*16*128，这里的16和分词编码信息的长度对应，不为固定值
