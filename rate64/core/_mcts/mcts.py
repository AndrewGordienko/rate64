import math
import random

import chess
import torch
import numpy as np

from rate64.util._dataset.data_pipeline import encode_board


class MCTSNode:
    def __init__(self, board: chess.Board):
        """
        Simple AlphaZero-style node:
        - children: move -> MCTSNode
        - N, W, Q, P indexed by move
        """
        self.board = board
        self.children = {}

        self.N = {}  # visit counts
        self.W = {}  # total value
        self.Q = {}  # mean value
        self.P = {}  # prior (uniform for now)

        legal_moves = list(board.legal_moves)
        if legal_moves:
            prior = 1.0 / len(legal_moves)
            for move in legal_moves:
                self.N[move] = 0
                self.W[move] = 0.0
                self.Q[move] = 0.0
                self.P[move] = prior


def evaluate_leaf(board: chess.Board, model) -> float:
    """
    Evaluate a leaf position with the value network.
    Returns value from White's perspective in [-1, 1].
    """
    x = encode_board(board)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        v = model(x).item()
    return v


def select_child(node: MCTSNode, board: chess.Board, c_puct: float = 1.4) -> chess.Move | None:
    """
    UCB-style child selection with only moves that are legal
    in the *current* board. If no legal moves match the node's
    move set, returns None.
    """
    best_score = -1e9
    best_move = None

    # Only consider moves that are actually pseudo-legal in this board
    candidate_moves = [m for m in node.N.keys() if board.is_pseudo_legal(m)]
    if not candidate_moves:
        return None

    total_N = sum(node.N[m] for m in candidate_moves)

    for move in candidate_moves:
        Q = node.Q[move]
        U = c_puct * node.P[move] * math.sqrt(total_N + 1.0) / (1.0 + node.N[move])
        score = Q + U

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def mcts_search(board: chess.Board, model, simulations: int = 100) -> chess.Move:
    """
    Run MCTS from the given board using the value network.
    Returns the move with highest visit count at the root.
    """

    root = MCTSNode(board)

    # If no legal moves, just return None
    if not root.N:
        return None

    # ----------------------------
    # Dirichlet noise at the root
    # ----------------------------
    moves_root = list(root.N.keys())
    alpha = 0.3
    epsilon = 0.25
    noise = np.random.dirichlet([alpha] * len(moves_root))

    for m, n in zip(moves_root, noise):
        root.P[m] = (1.0 - epsilon) * root.P[m] + epsilon * n

    # ----------------------------
    # MCTS simulations
    # ----------------------------
    for _ in range(simulations):
        node = root
        b = board.copy()
        path: list[tuple[MCTSNode, chess.Move]] = []

        # --- SELECTION + EXPANSION ---
        while True:
            # If terminal node, stop here
            if b.is_game_over():
                break

            # If this node has no children yet, expand here
            if not node.children:
                # First time we visit this node: create its children
                # based on current board b.
                expanded = MCTSNode(b)

                # Attach this expanded node to its parent if we came via a move
                if path:
                    parent, last_move = path[-1]
                    parent.children[last_move] = expanded
                else:
                    # We are at root
                    root = expanded

                node = expanded
                break

            # Otherwise, select a child move using UCB
            move = select_child(node, b)
            if move is None:
                # No legal candidate from this node for this board
                break

            path.append((node, move))
            b.push(move)

            # If we've already expanded this move, descend; otherwise create
            if move in node.children:
                node = node.children[move]
            else:
                child = MCTSNode(b)
                node.children[move] = child
                node = child
                break

        # --- EVALUATION ---
        # If game is over, use terminal value; else use network
        if b.is_game_over():
            result = b.result()  # "1-0", "0-1", "1/2-1/2"
            if result == "1-0":
                v = 1.0
            elif result == "0-1":
                v = -1.0
            else:
                v = 0.0
        else:
            v = evaluate_leaf(b, model)

        # Always treat v as from the side-to-move at board 'b' being White.
        # Convert to value from perspective of the player who just moved at each step.
        if not b.turn:
            # If it's Black to move, current value is from Black's perspective,
            # so flip sign to standard "White perspective".
            v = -v

        # --- BACKPROPAGATION ---
        for parent, move in reversed(path):
            parent.N[move] += 1
            parent.W[move] += v
            parent.Q[move] = parent.W[move] / parent.N[move]
            v = -v  # alternate perspectives up the tree

    # ----------------------------
    # Choose move at root
    # ----------------------------
    # Use highest visit count, with random tie-break to avoid deterministic loops
    max_N = max(root.N[m] for m in root.N)
    best_moves = [m for m in root.N if root.N[m] == max_N]

    return random.choice(best_moves)
