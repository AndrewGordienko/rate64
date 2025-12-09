# data_pipeline.py for encoding the board
import chess
import numpy as np

def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_order = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING,
    ]

    for color in [chess.WHITE, chess.BLACK]:
        for i, ptype in enumerate(piece_order):
            plane_idx = i if color == chess.WHITE else i + 6
            for square in board.pieces(ptype, color):
                r = 7 - chess.square_rank(square)
                c = chess.square_file(square)
                planes[plane_idx, r, c] = 1.0

    features = planes.reshape(-1)

    stm = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)

    castling = np.array([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ], dtype=np.float32)

    ep_vec = np.zeros(9, dtype=np.float32)
    if board.ep_square is None:
        ep_vec[-1] = 1.0
    else:
        ep_vec[chess.square_file(board.ep_square)] = 1.0

    return np.concatenate([features, stm, castling, ep_vec])
