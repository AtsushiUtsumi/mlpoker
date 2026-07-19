from poker_domain.value_objects.card import Card, Rank, Suit

_RANK_CHARS = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A",
}
_SUIT_CHARS = {
    Suit.HEARTS: "h", Suit.DIAMONDS: "d", Suit.CLUBS: "c", Suit.SPADES: "s",
}


def card_to_str(card: Card) -> str:
    """例: Card(ACE, SPADES) -> 'As'"""
    return _RANK_CHARS[card.rank] + _SUIT_CHARS[card.suit]


def cards_to_list(cards) -> list[str]:
    return [card_to_str(c) for c in cards]
