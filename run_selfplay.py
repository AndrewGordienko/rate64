# run_selfplay.py

import torch

from rate64.core.selfplay import selfplay_many_games
from rate64.core.train_alphazero import train_on_dataset
from rate64.core.evaluate_new_model import evaluate_models  # <-- evaluation step


def main():

    ITERATIONS = 5               # do 50 full AlphaZero cycles
    NUM_GAMES = 10               # per iteration
    SIMS_PER_MOVE = 200          # MCTS sims

    OLD_MODEL = "policy_value_net.pt"

    for it in range(1, ITERATIONS + 1):
        print(f"\n\n========================")
        print(f" ALPHAZERO ITERATION {it}")
        print(f"========================\n")

        SELFPLAY_OUTPUT = f"selfplay_buffer_{it}.pt"
        NEW_MODEL = f"policy_value_net_new.pt"

        # 1) Self-play
        
        selfplay_many_games(
            model_path=OLD_MODEL,
            output_path=SELFPLAY_OUTPUT,
            games=NUM_GAMES,
            simulations=SIMS_PER_MOVE
        )
        
        # 2) Train
        train_on_dataset(
            dataset_path=SELFPLAY_OUTPUT,
            old_model_path=OLD_MODEL,
            save_path=NEW_MODEL
        )
        
        # 3) Evaluate New vs Old
        passed = evaluate_models(
            old_path=OLD_MODEL,
            new_path=NEW_MODEL,
            games=10,
            sims=SIMS_PER_MOVE
        )

        # 4) Promotion
        if passed:
            import shutil
            shutil.copyfile(NEW_MODEL, OLD_MODEL)
            print(">>> Model improved — promoted!")
        else:
            print(">>> Rejecting new model — keeping old.")



if __name__ == "__main__":
    main()
