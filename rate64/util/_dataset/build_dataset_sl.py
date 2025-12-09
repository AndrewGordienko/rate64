import chess
import chess.pgn
import numpy as np
import torch

from rate64.util._dataset.data_pipeline import encode_board

RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

# -------------------------------------------------------------------
# Move indexing (AlphaZero-style 4672 policy vector)
# -------------------------------------------------------------------

# Directions for sliding moves (queen-like)
DIRS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

def move_to_index(move: chess.Move) -> int:
    """
    Map a legal move to an index in a 4672-dimensional vector.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    fr = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)
    tr = chess.square_rank(to_sq)
    tf = chess.square_file(to_sq)

    dr = tr - fr
    df = tf - ff

    # ---------------------------------------------------------------
    # 1) KNIGHT MOVES (first 512 indices)
    # ---------------------------------------------------------------
    knight_moves = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]

    for dir_id, (rr, ff_) in enumerate(knight_moves):
        if dr == rr and df == ff_:
            return 64 * dir_id + from_sq

    # ---------------------------------------------------------------
    # 2) SLIDING MOVES (Queen, Rook, Bishop)
    # 8 directions × 7 steps × 64 squares = 3584
    # ---------------------------------------------------------------
    base = 512
    for d_id, (rr, ff_) in enumerate(DIRS):
        for dist in range(1, 8):
            if dr == rr * dist and df == ff_ * dist:
                offset = (d_id * 7 + (dist - 1)) * 64
                return base + offset + from_sq

    # ---------------------------------------------------------------
    # 3) Underpromotions (simple handling)
    # ---------------------------------------------------------------
    # For now: map all promotions to a single bucket near the end
    promo_index = 4671
    if move.promotion is not None:
        return promo_index

    raise ValueError(f"Move {move.uci()} cannot be encoded!")
    

# -------------------------------------------------------------------
# Dataset builder
# -------------------------------------------------------------------

def build_dataset(pgn_path="lichess_elite_2020-06.pgn",
                  max_games=2000,
                  max_positions=200_000):

    X = []
    Pi = []
    V = []

    game_count = 0
    pos_count = 0

    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            if game_count >= max_games:
                break

            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_count += 1
            result = game.headers.get("Result")
            if result not in RESULT_MAP:
                continue

            value = RESULT_MAP[result]
            board = game.board()

            for move in game.mainline_moves():
                if pos_count >= max_positions:
                    break

                # Encode board
                x = encode_board(board)

                # Policy target (expert move)
                pi = np.zeros(4672, dtype=np.float32)
                try:
                    idx = move_to_index(move)
                    pi[idx] = 1.0
                except Exception as e:
                    # Skip moves we cannot encode
                    board.push(move)
                    continue

                X.append(x)
                Pi.append(pi)
                V.append(value)

                pos_count += 1
                board.push(move)

            if pos_count >= max_positions:
                break

    # Convert to tensors
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Pi = torch.tensor(np.array(Pi), dtype=torch.float32)
    V = torch.tensor(np.array(V), dtype=torch.float32).view(-1, 1)

    print("Saving dataset_sl.pt ...")
    torch.save({"inputs": X, "policy": Pi, "value": V}, "dataset_sl.pt")

    print("Done.")
    print("Games:", game_count)
    print("Positions:", pos_count)

if __name__ == "__main__":
    print("Building supervised-learning dataset (AlphaZero style)...")
    build_dataset(
        pgn_path="lichess_elite_2020-06.pgn",
        max_games=2000,
        max_positions=200_000
    )
