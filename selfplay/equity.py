import random
from collections import Counter
from itertools import combinations

from poker_domain.value_objects.hand import HandRank

# 学習データ用の確率分布を作る列順 (弱い役 -> 強い役)
HAND_RANK_ORDER: tuple[HandRank, ...] = (
    HandRank.HIGH_CARD,
    HandRank.ONE_PAIR,
    HandRank.TWO_PAIR,
    HandRank.THREE_OF_A_KIND,
    HandRank.STRAIGHT,
    HandRank.FLUSH,
    HandRank.FULL_HOUSE,
    HandRank.FOUR_OF_A_KIND,
    HandRank.STRAIGHT_FLUSH,
    HandRank.ROYAL_FLUSH,
)

_RANK_VALUE = {ch: i + 2 for i, ch in enumerate("23456789TJQKA")}
FULL_DECK: tuple[str, ...] = tuple(r + s for r in "23456789TJQKA" for s in "hdcs")


def _best_straight_high(desc_unique_ranks: list[int]) -> int | None:
    """降順・重複なしのランク列から、最も高いストレートの最上位ランクを返す(無ければNone)。
    ホイール(A-2-3-4-5)はAを1として末尾に加えて判定する"""
    values = desc_unique_ranks
    if 14 in values and 1 not in values:
        values = values + [1]
    for i in range(len(values) - 4):
        if values[i] - values[i + 4] == 4:
            return values[i]
    return None


def classify(cards: list[str]) -> HandRank:
    """
    7枚(や6枚以下でも可)のカード文字列 (例: "Ah","Td") から、
    最も強い5枚の組み合わせの役カテゴリだけを判定する。
    poker_domain.HandEvaluator と違い、全 C(n,5) 組み合わせを列挙しないため高速。
    タイブレーカーの算出は行わない(役カテゴリの確率分布計算にのみ使用するため不要)。
    """
    ranks = [_RANK_VALUE[c[0]] for c in cards]
    suits = [c[1] for c in cards]

    suit_counts = Counter(suits)
    flush_suit = next((s for s, cnt in suit_counts.items() if cnt >= 5), None)

    if flush_suit is not None:
        flush_ranks = sorted({r for r, s in zip(ranks, suits) if s == flush_suit}, reverse=True)
        sf_high = _best_straight_high(flush_ranks)
        if sf_high is not None:
            return HandRank.ROYAL_FLUSH if sf_high == 14 else HandRank.STRAIGHT_FLUSH

    rank_counts = Counter(ranks)
    by_count = sorted(rank_counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    pattern = tuple(c for _, c in by_count)

    if pattern[0] == 4:
        return HandRank.FOUR_OF_A_KIND

    if pattern[0] == 3 and len(pattern) > 1 and pattern[1] >= 2:
        return HandRank.FULL_HOUSE

    if flush_suit is not None:
        return HandRank.FLUSH

    straight_high = _best_straight_high(sorted(set(ranks), reverse=True))
    if straight_high is not None:
        return HandRank.STRAIGHT

    if pattern[0] == 3:
        return HandRank.THREE_OF_A_KIND

    if pattern[0] == 2 and pattern.count(2) >= 2:
        return HandRank.TWO_PAIR

    if pattern[0] == 2:
        return HandRank.ONE_PAIR

    return HandRank.HIGH_CARD


def has_draw(cards: list[str]) -> bool:
    """
    5枚(ホール2枚+フロップ3枚)のカードから、フラッシュドロー・ストレートドロー
    (オープンエンド/ガットショット問わず)のいずれかが存在するかを判定する。
    役自体が既にワンペア以上の場合はこの判定を使う必要はない(呼び出し側で分岐する)。
    """
    ranks = [_RANK_VALUE[c[0]] for c in cards]
    suits = [c[1] for c in cards]

    suit_counts = Counter(suits)
    if any(cnt == 4 for cnt in suit_counts.values()):
        return True

    unique_ranks = set(ranks)
    if 14 in unique_ranks:
        unique_ranks.add(1)  # ホイール方向のストレートドローも考慮
    for low in range(1, 11):
        window = set(range(low, low + 5))
        if len(window & unique_ranks) == 4:
            return True
    return False


def is_nuts(hole_cards: list[str], community_cards: list[str]) -> bool:
    """
    現時点(ターン/リバー)で、残りの見えていない全カードから作りうるどの2枚のホールカードと
    比べても自分の役が負けないか(=ナッツかどうか)を判定する。
    役カテゴリが低い(ツーペア未満)場合はナッツである可能性が実質無いため、
    重い全列挙を避けて早期にFalseを返す。
    """
    known = hole_cards + community_cards
    own_category = classify(known)
    if own_category < HandRank.TWO_PAIR:
        return False

    unseen = [c for c in FULL_DECK if c not in known]
    for other_hole in combinations(unseen, 2):
        other_category = classify(list(other_hole) + community_cards)
        if other_category > own_category:
            return False
    return True


def estimate_hand_rank_probabilities(
    hole_cards: tuple[str, ...],
    community_cards: tuple[str, ...],
    samples: int,
    rng: random.Random,
) -> tuple[dict[HandRank, float], int]:
    """
    現時点のホールカード+コミュニティカードから、リバーまでに完成する役カテゴリの
    確率分布を推定する。残り枚数が少ない(FLOP以降)場合は全パターンを厳密に列挙し、
    残り5枚(PRE_FLOP)の場合のみモンテカルロ・サンプリングで近似する。
    戻り値は (役カテゴリ -> 確率 の辞書, 使用したサンプル/組み合わせ数)
    """
    known = list(hole_cards) + list(community_cards)
    remaining_needed = 5 - len(community_cards)
    unseen = [c for c in FULL_DECK if c not in known]

    counts: dict[HandRank, int] = {r: 0 for r in HandRank}

    if remaining_needed == 0:
        counts[classify(known)] = 1
        total = 1
    elif remaining_needed <= 2:
        combos = list(combinations(unseen, remaining_needed))
        for combo in combos:
            counts[classify(known + list(combo))] += 1
        total = len(combos)
    else:
        for _ in range(samples):
            draw = rng.sample(unseen, remaining_needed)
            counts[classify(known + draw)] += 1
        total = samples

    probabilities = {r: counts[r] / total for r in HandRank}
    return probabilities, total
