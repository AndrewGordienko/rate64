import pygame
import chess
import torch
import numpy as np

from rate64.core.model import ValueNet
from rate64.util._dataset.data_pipeline import encode_board
from rate64.core._mcts.mcts import mcts_search   # <-- ADDED

# ============================================
# SETTINGS
# ============================================

SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
WINDOW_SIZE = (BOARD_SIZE, BOARD_SIZE)

PIECE_IMAGES = {}

FILENAME_MAP = {
    # White
    "P": "wp.png",
    "N": "wn.png",
    "B": "wb.png",
    "R": "wr.png",
    "Q": "wq.png",
    "K": "wk.png",

    # Black
    "p": "p.png",
    "n": "n.png",
    "b": "b.png",
    "r": "r.png",
    "q": "q.png",
    "k": "k.png",
}

# ============================================
# MODEL LOADING
# ============================================

def load_value_net(path="value_net.pt"):
    model = ValueNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def evaluate_position(model, board):
    x = encode_board(board)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(x).item()

# ============================================
# ENGINE MOVE (MCTS)
# ============================================

def engine_best_move(model, board):
    """
    Choose a move using Monte Carlo Tree Search guided by the neural value network.
    """
    return mcts_search(board, model, simulations=200)

# ============================================
# GUI: BOARD + PIECES
# ============================================

def load_piece_images():
    import os
    base = os.path.dirname(os.path.abspath(__file__))  # rate64/util/_visualizer/
    piece_dir = os.path.join(base, "..", "..", "pieces")
    piece_dir = os.path.abspath(piece_dir)

    for symbol, filename in FILENAME_MAP.items():
        path = os.path.join(piece_dir, filename)
        print("Loading:", path)  # debug

        img = pygame.image.load(path)
        img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        PIECE_IMAGES[symbol] = img


def draw_board(screen, board):
    colors = [(240, 217, 181), (181, 136, 99)]

    for rank in range(8):
        for file in range(8):
            square_color = colors[(rank + file) % 2]

            rect = pygame.Rect(
                file * SQUARE_SIZE,
                (7 - rank) * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE,
            )

            pygame.draw.rect(screen, square_color, rect)

    # Draw pieces
    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE

        screen.blit(PIECE_IMAGES[symbol], (x, y))

# ============================================
# HELPER
# ============================================

def mouse_square(pos):
    x, y = pos
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    return chess.square(file, rank)

# ============================================
# MAIN GAME LOOP
# ============================================

def play_game(play_as_white=True):
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Rate64 Engine Play")

    load_piece_images()
    model = load_value_net()

    board = chess.Board()
    selected_square = None
    running = True

    # Engine plays first if user is Black
    if not play_as_white:
        move = engine_best_move(model, board)
        board.push(move)

    while running:
        draw_board(screen, board)
        pygame.display.flip()

        if board.is_game_over():
            print("Game finished:", board.result())
            pygame.time.wait(3000)
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse input only on user's turn
            if event.type == pygame.MOUSEBUTTONDOWN:
                if board.turn == (chess.WHITE if play_as_white else chess.BLACK):
                    sq = mouse_square(event.pos)

                    if selected_square is None:
                        selected_square = sq
                    else:
                        move = chess.Move(selected_square, sq)

                        if move in board.legal_moves:
                            board.push(move)

                            if not board.is_game_over():
                                engine_move = engine_best_move(model, board)
                                board.push(engine_move)

                        selected_square = None

    pygame.quit()

# ============================================
# ENTRY POINT
# ============================================

def main():
    print("Choose side:")
    print("1 = Play as WHITE")
    print("2 = Play as BLACK")

    side = input("> ").strip()

    if side == "1":
        play_game(play_as_white=True)
    else:
        play_game(play_as_white=False)

if __name__ == "__main__":
    main()
