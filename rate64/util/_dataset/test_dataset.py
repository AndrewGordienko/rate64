# test_dataset.py

import torch
import chess
import numpy as np
import random

DATASET_PATH = "dataset.pt"

print("Loading dataset...")
data = torch.load(DATASET_PATH)

inputs = data["inputs"]     # shape: [N, 782]
labels = data["labels"]     # shape: [N, 1]

print("Dataset loaded.")
print("Inputs shape:", inputs.shape)
print("Labels shape:", labels.shape)

# Show first sample
idx = random.randint(0, len(inputs) - 1)

x = inputs[idx]
y = labels[idx].item()

print("\nFirst sample:")
print("Vector length:", len(x))
print("Label (game result):", y)

# ------------------------------------------------------------
# OPTIONAL: decode the board back into a chess.Board position
# ------------------------------------------------------------

def decode_board(vec):
    board = chess.Board.empty()

    # First 768 values = 12 planes × 64 squares
    planes = vec[:768].reshape(12, 8, 8)

    piece_types = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING,
    ]

    # 0–5 white planes, 6–11 black planes
    for color in [chess.WHITE, chess.BLACK]:
        for i, ptype in enumerate(piece_types):
            plane_idx = i if color == chess.WHITE else i + 6
            plane = planes[plane_idx]

            for r in range(8):
                for c in range(8):
                    if plane[r, c] == 1.0:
                        square = chess.square(c, 7 - r)
                        board.set_piece_at(square, chess.Piece(ptype, color))

    # Side to move
    stm = vec[768]
    board.turn = chess.WHITE if stm > 0.5 else chess.BLACK

    # Castling rights
    castling_bits = vec[769:773]
    board.castling_rights = 0

    if castling_bits[0] > 0.5:
        board.castling_rights |= chess.BB_H1  # white kingside
    if castling_bits[1] > 0.5:
        board.castling_rights |= chess.BB_A1  # white queenside
    if castling_bits[2] > 0.5:
        board.castling_rights |= chess.BB_H8  # black kingside
    if castling_bits[3] > 0.5:
        board.castling_rights |= chess.BB_A8  # black queenside

    # En passant square
    ep_vec = vec[773:782]
    ep_file = int(np.argmax(ep_vec))

    if ep_file == 8:
        board.ep_square = None
    else:
        # EP rank depends on side to move (inverse of previous move)
        rank = 5 if board.turn == chess.BLACK else 2
        board.ep_square = chess.square(ep_file, rank)

    return board


# Try decoding the first vector
print("\nDecoded board position:")
try:
    board_decoded = decode_board(x.numpy())
    print(board_decoded)
    print("")
    print(board_decoded.unicode())
except Exception as e:
    print("Decode failed:", e)
