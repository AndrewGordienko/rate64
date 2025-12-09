# rate64/util/_policy/move_indexing.py

import chess

# Knight move offsets
KNIGHT_DIRS = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

# Sliding directions (Queen-like)
SLIDE_DIRS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

# Promotion bucket (single index for all underpromotions)
PROMOTION_INDEX = 4671  # last entry in policy vector


def move_to_index(move: chess.Move) -> int:
    """
    Map a chess.Move to an integer index in [0, 4671].

    Structure:
        0–511     knight moves (8 dirs × 64 from-sq)
        512–4095  sliding moves (8 dirs × 7 distances × 64 from-sq)
        4096–4670 reserved (if needed)
        4671      promotion bucket
    """

    from_sq = move.from_square
    to_sq = move.to_square

    fr = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)
    tr = chess.square_rank(to_sq)
    tf = chess.square_file(to_sq)

    dr = tr - fr
    df = tf - ff

    # ---------------------------------------------------
    # KNIGHT MOVES (first 512)
    # ---------------------------------------------------
    for dir_id, (rr, ff_) in enumerate(KNIGHT_DIRS):
        if dr == rr and df == ff_:
            return 64 * dir_id + from_sq

    # ---------------------------------------------------
    # SLIDING MOVES (Queen-like)
    # 8 dirs × 7 distances × 64 = 3584 entries
    # ---------------------------------------------------
    base = 512
    offset = 0

    for d_id, (rr, ff_) in enumerate(SLIDE_DIRS):
        for dist in range(1, 8):
            if dr == rr * dist and df == ff_ * dist:
                offset = (d_id * 7 + (dist - 1)) * 64
                return base + offset + from_sq

    # ---------------------------------------------------
    # PROMOTIONS BUCKET
    # ---------------------------------------------------
    if move.promotion is not None:
        return PROMOTION_INDEX

    # ---------------------------------------------------
    # If we cannot encode the move
    # ---------------------------------------------------
    raise ValueError(f"Cannot encode move {move.uci()} → no matching index.")
