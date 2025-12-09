import chess
import chess.pgn
import torch
import numpy as np

from rate64.core.model import ValueNet
from rate64.util.data_pipeline import encode_board


def load_value_net(path="value_net.pt"):
    model = ValueNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def evaluate_position(model, board):
    x = encode_board(board)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        value = model(x).item()
    return value


def choose_move(model, board):
    legal_moves = list(board.legal_moves)

    best_move = None
    best_score = None

    for move in legal_moves:
        board.push(move)
        score = evaluate_position(model, board)
        board.pop()

        # White wants max score, black wants min score
        if best_move is None:
            best_move = move
            best_score = score
        else:
            if board.turn == chess.BLACK:
                if score < best_score:
                    best_move = move
                    best_score = score
            else:
                if score > best_score:
                    best_move = move
                    best_score = score

    return best_move, best_score


def print_board_with_eval(board, model):
    print(board)
    v = evaluate_position(model, board)
    print(f"Eval: {v:+.3f}\n")
    return v


def self_play(model, max_moves=200):
    board = chess.Board()

    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play"
    game.headers["White"] = "ValueNet"
    game.headers["Black"] = "ValueNet"
    node = game

    print("Starting self-play game...\n")

    for move_number in range(1, max_moves + 1):
        if board.is_game_over():
            break

        print(f"Move {move_number} — {'White' if board.turn else 'Black'} to move")
        current_eval = print_board_with_eval(board, model)

        move, move_eval = choose_move(model, board)

        print(f"Chosen move: {move}  (eval after move = {move_eval:+.3f})")
        print("-" * 40)

        board.push(move)
        node = node.add_variation(move)

    print("\nFinal position:")
    print(board)
    print("\nGame result:", board.result())

    game.headers["Result"] = board.result()

    print("\nPGN:\n")
    print(game)



def main():
    model = load_value_net("value_net.pt")
    self_play(model)


if __name__ == "__main__":
    main()
