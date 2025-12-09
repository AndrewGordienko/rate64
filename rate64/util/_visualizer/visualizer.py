import pygame
import chess
import chess.pgn

# --- CONFIG ---
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
WINDOW_WIDTH = BOARD_SIZE + 200
WINDOW_HEIGHT = BOARD_SIZE

PIECE_IMAGES = {}

# Your filenames
FILENAME_MAP = {
    "P": "wp.png",
    "N": "wn.png",
    "B": "wb.png",
    "R": "wr.png",
    "Q": "wq.png",
    "K": "wk.png",

    "p": "p.png",
    "n": "n.png",
    "b": "b.png",
    "r": "r.png",
    "q": "q.png",
    "k": "k.png",
}


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
    # Draw squares
    for rank in range(8):
        for file in range(8):
            rect = pygame.Rect(file * SQUARE_SIZE,
                               (7 - rank) * SQUARE_SIZE,
                               SQUARE_SIZE, SQUARE_SIZE)

            color = (238, 238, 210) if (file + rank) % 2 == 0 else (118, 150, 86)
            pygame.draw.rect(screen, color, rect)

    # Draw pieces
    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE

        screen.blit(PIECE_IMAGES[symbol], (x, y))


def load_pgn(path="selfplay.pgn"):
    with open(path) as f:
        game = chess.pgn.read_game(f)

    moves = []
    node = game

    while node.variations:
        node = node.variations[0]
        moves.append(node.move)

    return game, moves


def draw_ui(screen, font, move_index, total_moves):
    pygame.draw.rect(screen, (30, 30, 30), (BOARD_SIZE, 0, 200, WINDOW_HEIGHT))

    text = font.render(f"Move: {move_index}/{total_moves}", True, (255, 255, 255))
    screen.blit(text, (BOARD_SIZE + 20, 20))

    # Buttons
    pygame.draw.rect(screen, (70, 70, 200), (BOARD_SIZE + 20, 80, 160, 50))
    pygame.draw.rect(screen, (70, 200, 70), (BOARD_SIZE + 20, 150, 160, 50))

    prev_text = font.render("Previous", True, (255, 255, 255))
    next_text = font.render("Next", True, (255, 255, 255))

    screen.blit(prev_text, (BOARD_SIZE + 45, 95))
    screen.blit(next_text, (BOARD_SIZE + 60, 165))


def in_button(mouse, x, y, w, h):
    return x <= mouse[0] <= x + w and y <= mouse[1] <= y + h


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Rate64 Self-Play Viewer")

    font = pygame.font.SysFont("Arial", 22)

    load_piece_images()

    game, moves = load_pgn("selfplay.pgn")
    board = game.board()
    move_index = 0

    running = True
    while running:
        screen.fill((0, 0, 0))
        draw_board(screen, board)
        draw_ui(screen, font, move_index, len(moves))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse clicks on UI buttons
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                # Previous button
                if in_button((mx, my), BOARD_SIZE + 20, 80, 160, 50):
                    if move_index > 0:
                        move_index -= 1
                        board.pop()

                # Next button
                if in_button((mx, my), BOARD_SIZE + 20, 150, 160, 50):
                    if move_index < len(moves):
                        board.push(moves[move_index])
                        move_index += 1

    pygame.quit()


if __name__ == "__main__":
    main()
