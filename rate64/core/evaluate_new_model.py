# evaluate_new_model.py

import chess
import torch
from rate64.core.model import AlphaZeroNet
from rate64.core._mcts.mcts import mcts_search


def play_match_game(model_old, model_new, sims=200, new_as_white=True):
    """
    Plays ONE full game of:
        new_model vs old_model
    Returns +1 if new wins, -1 if old wins, 0 for draw.
    """

    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # White moves
            if new_as_white:
                move = mcts_search(board, model_new, simulations=sims)
            else:
                move = mcts_search(board, model_old, simulations=sims)
        else:
            # Black moves
            if new_as_white:
                move = mcts_search(board, model_old, simulations=sims)
            else:
                move = mcts_search(board, model_new, simulations=sims)

        board.push(move)

    result = board.result()
    if result == "1-0":
        print("white won")
        return 1 if new_as_white else -1
    elif result == "0-1":
        print("black won")
        return -1 if new_as_white else 1
    else:
        return 0


def evaluate_models(old_path="policy_value_net.pt",
                    new_path="policy_value_net_new.pt",
                    games=40,
                    sims=200):

    print("Loading models...")
    model_old = AlphaZeroNet()
    model_old.load_state_dict(torch.load(old_path, map_location="cpu"))
    model_old.eval()

    model_new = AlphaZeroNet()
    model_new.load_state_dict(torch.load(new_path, map_location="cpu"))
    model_new.eval()

    print("Beginning evaluation match...")
    half = games // 2
    score = 0

    # First half: new model is White
    print(" - Playing games as WHITE...")
    for _ in range(half):
        score += play_match_game(model_old, model_new, sims, new_as_white=True)

    # Second half: new model is Black
    print(" - Playing games as BLACK...")
    for _ in range(half):
        score += play_match_game(model_old, model_new, sims, new_as_white=False)

    # Convert score into win rate
    # score counts: new win = +1, draw = 0, loss = -1
    win_rate = (score + games) / (2 * games)

    print(f"Final Score: {score} (from {-games} to +{games})")
    print(f"Win Rate (new vs old): {win_rate * 100:.2f}%")

    if win_rate >= 0.55:
        print("ACCEPTED: New model is better. Replace old model.")
        return True
    else:
        print("REJECTED: New model not strong enough.")
        return False

