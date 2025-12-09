import pygame
import chess
import torch

# Load the AlphaZero dual-head model
from rate64.core.model import AlphaZeroNet

# MCTS using policy + value heads
from rate64.core._mcts.mcts import mcts_search

# ============================================
# VISUAL SETTINGS
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

def load_engine_model(path="policy_value_net.pt"):
    """
    Load the AlphaZero-style neural network.
    """
    model = AlphaZeroNet()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================
# ENGINE MOVE (MCTS—200 searches)
# ============================================

def engine_best_move(model, board):
    """
    Compute the engine move using full AlphaZero MCTS.
    200 simulations per move.
    """
    return mcts_search(board, model, simulations=200)


# ============================================
# GRAPHICS
# ============================================

def load_piece_images():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    piece_dir = os.path.join(base, "..", "..", "pieces")
    piece_dir = os.path.abspath(piece_dir)

    for symbol, filename in FILENAME_MAP.items():
        path = os.path.join(piece_dir, filename)
        print("Loading:", path)
        img = pygame.image.load(path)
        img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        PIECE_IMAGES[symbol] = img


def draw_board(screen, board):
    light = (240, 217, 181)
    dark = (181, 136, 99)

    for rank in range(8):
        for file in range(8):
            color = light if (rank + file) % 2 == 0 else dark
            rect = pygame.Rect(
                file * SQUARE_SIZE,
                (7 - rank) * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE,
            )
            pygame.draw.rect(screen, color, rect)

    # Draw pieces
    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        screen.blit(PIECE_IMAGES[symbol], (x, y))


# ============================================
# HELPERS
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
    pygame.display.set_caption("Rate64 AlphaZero Engine")

    load_piece_images()
    model = load_engine_model()

    board = chess.Board()
    selected_square = None
    running = True

    # Engine moves immediately if user plays Black
    if not play_as_white:
        engine_move = engine_best_move(model, board)
        board.push(engine_move)

    while running:
        draw_board(screen, board)
        pygame.display.flip()

        if board.is_game_over():
            print("Game Over:", board.result())
            pygame.time.wait(2000)
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle human move
            if event.type == pygame.MOUSEBUTTONDOWN:
                if board.turn == (chess.WHITE if play_as_white else chess.BLACK):

                    sq = mouse_square(event.pos)

                    if selected_square is None:
                        selected_square = sq
                    else:
                        move = chess.Move(selected_square, sq)

                        if move in board.legal_moves:
                            board.push(move)

                            # Engine responds using full MCTS
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

    play_game(play_as_white=(side == "1"))


if __name__ == "__main__":
    main()
