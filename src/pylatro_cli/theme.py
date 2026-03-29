"""Balatro color palette and theme constants."""

BALATRO_PALETTE = {
    # Background
    "bg_dark": "#1a1a2e",
    "bg_medium": "#16213e",
    "bg_light": "#0f3460",
    "bg_panel": "#1c1c3a",
    # Suits
    "hearts": "#e74c3c",
    "diamonds": "#3498db",
    "clubs": "#2ecc71",
    "spades": "#ecf0f1",
    # Buttons / actions
    "btn_play": "#2980b9",
    "btn_discard": "#c0392b",
    "btn_sort": "#7f8c8d",
    "btn_buy": "#27ae60",
    "btn_sell": "#e67e22",
    "btn_reroll": "#8e44ad",
    "btn_skip": "#95a5a6",
    "btn_disabled": "#4a4a6a",
    # Editions
    "edition_foil": "#4fc3f7",
    "edition_holo": "#ce93d8",
    "edition_polychrome": "#ffd54f",
    "edition_negative": "#b0bec5",
    # Seals
    "seal_red": "#e74c3c",
    "seal_blue": "#2196f3",
    "seal_gold": "#ffc107",
    "seal_purple": "#9c27b0",
    # Scoring
    "chips": "#42a5f5",
    "mult": "#ef5350",
    "gold": "#ffc107",
    # Text
    "text_primary": "#ecf0f1",
    "text_secondary": "#95a5a6",
    "text_muted": "#636e72",
    # Card
    "card_bg": "#2c2c54",
    "card_border": "#5c5c8a",
    "card_selected": "#f1c40f",
    "card_highlighted": "#3498db",
    "card_face_down": "#4a4a6a",
}

SUIT_COLORS = {
    "Hearts": BALATRO_PALETTE["hearts"],
    "Diamonds": BALATRO_PALETTE["diamonds"],
    "Clubs": BALATRO_PALETTE["clubs"],
    "Spades": BALATRO_PALETTE["spades"],
}

SUIT_SYMBOLS = {
    "Hearts": "\u2665",
    "Diamonds": "\u2666",
    "Clubs": "\u2663",
    "Spades": "\u2660",
}

EDITION_COLORS = {
    "foil": BALATRO_PALETTE["edition_foil"],
    "holo": BALATRO_PALETTE["edition_holo"],
    "polychrome": BALATRO_PALETTE["edition_polychrome"],
    "negative": BALATRO_PALETTE["edition_negative"],
}
