import math
import random

import chess
import torch
import numpy as np

from rate64.util._dataset.data_pipeline import encode_board
from rate64.util._policy.move_indexing import move_to_index   # <-- REQUIRED
from rate64.core.model import POLICY_DIM                      # 4672


# ============================================================
# MCTS NODE
# ============================================================

class MCTSNode:
    def __init__(self, board: chess.Board, priors=None):
        """
        priors: dict(move -> prior probability)
        """
        self.board = board
        self.children = {}

        self.N = {}   # visit counts
        self.W = {}   # total value
        self.Q = {}   # mean value
        self.P = {}   # policy prior

        legal = list(board.legal_moves)

        for move in legal:
            self.N[move] = 0
            self.W[move] = 0.0
            self.Q[move] = 0.0

            if priors is None:
                self.P[move] = 1.0 / len(legal)
            else:
                self.P[move] = priors.get(move, 0.0)


# ============================================================
# NETWORK EVALUATION
# ============================================================

def evaluate_with_network(board, model):
    """
    Returns:
        priors: dict(move -> probability)
        value: float in [-1,1]
    """
    x = encode_board(board)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = model(x)

    value = value.item()

    # softmax to probabilities
    probs = torch.softmax(policy_logits, dim=1).squeeze(0)

    # Map moves -> probability using move_to_index()
    priors = {}
    for move in board.legal_moves:
        try:
            idx = move_to_index(move)
            priors[move] = probs[idx].item()
        except Exception:
            priors[move] = 0.0

    # Normalize priors
    total = sum(priors.values())
    if total > 0:
        for m in priors:
            priors[m] /= total
    else:
        legal = list(board.legal_moves)
        for m in legal:
            priors[m] = 1.0 / len(legal)

    return priors, value


# ============================================================
# UCB SELECTION
# ============================================================

def select_child(node, c_puct=1.4):
    best_score = -1e9
    best_move = None

    total_N = sum(node.N[m] for m in node.N) + 1

    for move in node.N:
        Q = node.Q[move]
        U = c_puct * node.P[move] * math.sqrt(total_N) / (1 + node.N[move])
        score = Q + U

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# ============================================================
# MCTS SEARCH (returns move AND π vector)
# ============================================================

def mcts_search(board, model, simulations=200, return_pi=False):
    """
    AlphaZero-style MCTS.
    Returns:
        (best_move)                     if return_pi=False
        (best_move, pi_vector_4672)     if return_pi=True
    """

    # ---- INITIAL EVALUATION ----
    priors, _ = evaluate_with_network(board, model)
    root = MCTSNode(board, priors=priors)

    # ---- DIRICHLET NOISE ----
    legal_moves = list(root.P.keys())
    noise = np.random.dirichlet([0.3] * len(legal_moves))
    for m, n in zip(legal_moves, noise):
        root.P[m] = 0.75 * root.P[m] + 0.25 * n

    # ---- SIMULATIONS ----
    for _ in range(simulations):
        node = root
        b = board.copy()
        path = []

        # --- SELECTION ---
        while node.children:
            move = select_child(node)
            path.append((node, move))
            b.push(move)
            node = node.children[move]

        # --- EXPANSION + EVALUATION ---
        if not b.is_game_over():
            priors, value = evaluate_with_network(b, model)
            new_node = MCTSNode(b, priors=priors)

            if path:
                parent, move = path[-1]
                parent.children[move] = new_node
            else:
                root = new_node

            node = new_node
        else:
            # Terminal value
            result = b.result()
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            else:
                value = 0.0

        # Value always from perspective of side to move
        if not b.turn:
            value = -value

        # --- BACKPROPAGATE ---
        for parent, move in reversed(path):
            parent.N[move] += 1
            parent.W[move] += value
            parent.Q[move] = parent.W[move] / parent.N[move]
            value = -value

    # ============================================================
    # BEST MOVE SELECTION
    # ============================================================

    # pick by highest visit count
    max_visits = max(root.N[m] for m in root.N)
    candidates = [m for m in root.N if root.N[m] == max_visits]
    best_move = random.choice(candidates)

    # ============================================================
    # CONSTRUCT POLICY VECTOR π(s)
    # ============================================================

    if not return_pi:
        return best_move

    pi = np.zeros(POLICY_DIM, dtype=np.float32)

    total_visits = sum(root.N[m] for m in root.N)

    for move in root.N:
        try:
            idx = move_to_index(move)
            pi[idx] = root.N[move] / total_visits
        except Exception:
            pass  # ignore moves that can't be encoded

    return best_move, pi
