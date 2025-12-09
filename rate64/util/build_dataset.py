import chess.pgn
import numpy as np
import torch
from data_pipeline import encode_board

RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

pgn_path = "lichess_elite_2020-06.pgn"

# ---------------------------
# LIMITS
# ---------------------------
MAX_GAMES = 2000
MAX_POSITIONS = 200_000
# ---------------------------

inputs = []
labels = []

game_count = 0
pos_count = 0

print("Starting dataset build...")

with open(pgn_path, "r", encoding="utf-8") as f:
    while True:
        if MAX_GAMES is not None and game_count >= MAX_GAMES:
            print("Reached MAX_GAMES limit.")
            break

        game = chess.pgn.read_game(f)
        if game is None:
            break

        game_count += 1

        if game_count % 100 == 0:
            print(f"Processed {game_count} games, {pos_count} positions...")

        result = game.headers.get("Result")
        if result not in RESULT_MAP:
            continue

        y = RESULT_MAP[result]
        board = game.board()

        for move in game.mainline_moves():
            if MAX_POSITIONS is not None and pos_count >= MAX_POSITIONS:
                print("Reached MAX_POSITIONS limit.")
                break

            x = encode_board(board)
            inputs.append(x)
            labels.append(y)
            pos_count += 1

            if pos_count % 100_000 == 0:
                print(f"Encoded {pos_count} positions so far...")

            board.push(move)

        if MAX_POSITIONS is not None and pos_count >= MAX_POSITIONS:
            break

print("Converting to tensors...")
inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
labels = torch.tensor(np.array(labels), dtype=torch.float32).view(-1, 1)

print("Saving dataset.pt ...")
torch.save({"inputs": inputs, "labels": labels}, "dataset.pt")

print("Done.")
print(f"Total games processed: {game_count}")
print(f"Total positions encoded: {pos_count}")
