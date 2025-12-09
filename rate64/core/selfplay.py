import torch
import chess
import numpy as np
from tqdm import tqdm

from rate64.util._dataset.data_pipeline import encode_board
from rate64.core._mcts.mcts import mcts_search
from rate64.core.model import AlphaZeroNet   # your dual-head net

POLICY_DIM = 4672


def play_one_game(model, simulations=200):
    """
    Plays a single self-play game using MCTS guided by `model`.
    Returns game history as (state, pi, z) tuples.
    """

    board = chess.Board()
    history = []   # [(x, pi_vector), ...]
    moves_made = []

    # --------------------------------------
    # PLAY UNTIL GAME OVER
    # --------------------------------------
    while not board.is_game_over():
        # --- Run MCTS to get a move ---
        move, pi_vector = mcts_search(board, model, simulations, return_pi=True)

        # Store training sample BEFORE playing move
        x = encode_board(board)
        history.append((x, pi_vector, board.turn))  # NOTE: store side-to-move for later reward flipping

        board.push(move)
        moves_made.append(move)

    # --------------------------------------
    # GAME RESULT
    # --------------------------------------
    result = board.result()
    if result == "1-0":
        z_final = 1.0
    elif result == "0-1":
        z_final = -1.0
    else:
        z_final = 0.0

    # --------------------------------------
    # ASSIGN REWARD FOR EACH MOVE
    # --------------------------------------
    samples = []
    for x, pi, turn in history:
        # If turn was black-to-move, flip value
        z = z_final if turn == chess.WHITE else -z_final
        samples.append((x, pi, z))

    return samples


def selfplay_many_games(model_path="alphazero.pt",
                        output_path="selfplay_buffer.pt",
                        games=50,
                        simulations=200):
    """
    Runs N self-play games, collects all training samples, saves to file.
    """

    print("Loading model:", model_path)
    model = AlphaZeroNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_X = []
    all_PI = []
    all_Z = []

    for _ in tqdm(range(games), desc="Self-play games"):
        samples = play_one_game(model, simulations=simulations)
        for x, pi, z in samples:
            all_X.append(x)
            all_PI.append(pi)
            all_Z.append(z)

    all_X = torch.tensor(np.array(all_X), dtype=torch.float32)
    all_PI = torch.tensor(np.array(all_PI), dtype=torch.float32)
    all_Z = torch.tensor(np.array(all_Z), dtype=torch.float32).view(-1, 1)

    print(f"Total positions collected: {len(all_X)}")

    torch.save({
        "inputs": all_X,
        "policy": all_PI,
        "value": all_Z
    }, output_path)

    print("Saved:", output_path)
