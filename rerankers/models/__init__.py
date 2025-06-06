AVAILABLE_RANKERS = {}  # 可用的rerank工具【有13个】

try:
    from rerankers.models.transformer_ranker import TransformerRanker

    AVAILABLE_RANKERS["TransformerRanker"] = TransformerRanker
except ImportError:
    pass
try:
    from rerankers.models.api_rankers import APIRanker

    AVAILABLE_RANKERS["APIRanker"] = APIRanker
except ImportError:
    pass
try:
    from rerankers.models.rankgpt_rankers import RankGPTRanker

    AVAILABLE_RANKERS["RankGPTRanker"] = RankGPTRanker
except ImportError:
    pass
try:
    from rerankers.models.t5ranker import T5Ranker

    AVAILABLE_RANKERS["T5Ranker"] = T5Ranker
except ImportError:
    pass

try:
    from rerankers.models.colbert_ranker import ColBERTRanker

    AVAILABLE_RANKERS["ColBERTRanker"] = ColBERTRanker
except ImportError:
    pass

try:
    from rerankers.models.flashrank_ranker import FlashRankRanker

    AVAILABLE_RANKERS["FlashRankRanker"] = FlashRankRanker
except ImportError:
    pass

try:
    from rerankers.models.rankllm_ranker import RankLLMRanker

    AVAILABLE_RANKERS["RankLLMRanker"] = RankLLMRanker
except ImportError:
    pass

try:
    from rerankers.models.llm_layerwise_ranker import LLMLayerWiseRanker

    AVAILABLE_RANKERS["LLMLayerWiseRanker"] = LLMLayerWiseRanker
except ImportError:
    pass

try:
    from rerankers.models.monovlm_ranker import MonoVLMRanker
    AVAILABLE_RANKERS["MonoVLMRanker"] = MonoVLMRanker
except ImportError:
    pass

try:
    from rerankers.models.llm_relevance_filter import LLMRelevanceFilter
    AVAILABLE_RANKERS["LLMRelevanceFilter"] = LLMRelevanceFilter
except ImportError:
    pass

try:
    from rerankers.models.upr import UPRRanker
    AVAILABLE_RANKERS["UPRRanker"] = UPRRanker
except ImportError:
    pass

try:
    from rerankers.models.mxbai_v2 import MxBaiV2Ranker
    AVAILABLE_RANKERS["MxBaiV2Ranker"] = MxBaiV2Ranker
except ImportError:
    pass

try:
    from rerankers.models.pylate_ranker import PyLateRanker

    AVAILABLE_RANKERS["PyLateRanker"] = PyLateRanker
except ImportError:
    pass
