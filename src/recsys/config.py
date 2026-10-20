"""项目级配置常量。"""

from __future__ import annotations

from pathlib import Path

# 数据切分策略。会写进 checkpoint，并在评测时校验「数据是否按同一套规则切分」。
# 具体做法见 recsys.data.movielens.split_interactions_per_user：
# 每个用户的最后两次交互依次作为验证集与测试集。
SPLIT_STRATEGY = "per_user_last_two_strict_timestamp_v2"

# 默认目录。CLI 参数默认值同样使用这两个路径。
DEFAULT_DATA_DIR = Path("data")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# 新入口默认使用 KuaiRec：它是最终实验的数据集。
# MovieLens 仍保留为快速上手的第二条数据线，已交付的 MovieLens checkpoint
# 会显式带上 dataset 字段，评测时不受这个默认值影响。
DEFAULT_DATASET = "kuairec"

# MovieLens 的正反馈阈值：评分 >= 4 记为「喜欢」。
DEFAULT_MOVIELENS_POSITIVE_THRESHOLD = 4

# KuaiRec 的正反馈阈值：watch_ratio >= 1.0（完整播放）记为「喜欢」。
# 定案依据见 artifacts/kuairec_data_stats.json 与 docs/04-kuairec/01。
# 这个值必须在训练之前定死：正样本集合决定用户/物品词表，
# 之后换阈值会让所有已交付的 checkpoint 和指标失效。
KUAIREC_WATCH_RATIO_THRESHOLD = 1.0

# 是否把同一个 (user, video) 的多次播放合并成一次。
# KuaiRec 的 big_matrix 有 17.8% 的重复行，其中一部分是同一次播放被记了多行，
# 另一部分是同一用户在不同时间重复播放同一个视频。
# 不合并会让评测目标出现在训练历史里，从而被 Top-K 的「已看过就不推荐」规则过滤掉。
KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS = True
