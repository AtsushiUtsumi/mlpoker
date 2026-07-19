from poker_domain.value_objects.card import Card, Rank, Suit

_RANK_CHARS = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A",
}
_SUIT_CHARS = {
    Suit.HEARTS: "h", Suit.DIAMONDS: "d", Suit.CLUBS: "c", Suit.SPADES: "s",
}


_RANK_FROM_CHAR = {v: k for k, v in _RANK_CHARS.items()}
_SUIT_FROM_CHAR = {v: k for k, v in _SUIT_CHARS.items()}


def card_to_str(card: Card) -> str:
    """例: Card(ACE, SPADES) -> 'As'"""
    return _RANK_CHARS[card.rank] + _SUIT_CHARS[card.suit]


def str_to_card(s: str) -> Card:
    """例: 'As' -> Card(ACE, SPADES) (card_to_str の逆変換)"""
    return Card(rank=_RANK_FROM_CHAR[s[0]], suit=_SUIT_FROM_CHAR[s[1]])


def cards_to_list(cards) -> list[str]:
    return [card_to_str(c) for c in cards]
