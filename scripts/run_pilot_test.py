#!/usr/bin/env python3
"""
Pilot Test: Run testing denial prompting RL to check if the system works.

This runs a 20-step training session and collects statistics.
"""

import sys
import json
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config_loader import load_config
from data.neocoder_loader import NeoCoderLoader, NeoCoderDataset
from models.model_wrapper import ModelWrapper
from rewards.reward_function import RewardFunction
from training.grpo_trainer import GRPOTrainer


def analyze_results(output_dir):
    print("\n" + "="*80)
    print("pilot test results analysis")
    print("="*80)

    # Load metrics
    metrics_file = Path(output_dir) / "metrics.json"
    if not metrics_file.exists():
        print("No metrics file found")
        return

    with open(metrics_file) as f:
        metrics = json.load(f)

    # Extract data
    steps = sorted([int(k) for k in metrics.keys()])
    rewards = [metrics[str(s)]['mean_reward'] for s in steps]
    violations = [metrics[str(s)]['mean_violations'] for s in steps]
    success_rates = [metrics[str(s)]['success_rate'] for s in steps]
    losses = [metrics[str(s)]['loss'] for s in steps]

    # Compute statistics
    print("\nTraining Progression:")
    print(f"Total steps: {len(steps)}")
    print(f"Steps completed: {steps[-1] + 1}")

    print("\nReward Statistics:")
    print(f"Initial mean reward: {rewards[0]:.3f}")
    print(f"Final mean reward: {rewards[-1]:.3f}")
    print(f"Change: {rewards[-1] - rewards[0]:+.3f} ({(rewards[-1] - rewards[0])/abs(rewards[0])*100:+.1f}%)")
    print(f"Max reward achieved: {max(rewards):.3f}")
    print(f"Min reward achieved: {min(rewards):.3f}")

    # Check if improving
    first_half = rewards[:len(rewards)//2]
    second_half = rewards[len(rewards)//2:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)

    print(f"\nLearning Progress:")
    print(f"First half average: {avg_first:.3f}")
    print(f"Second half average: {avg_second:.3f}")
    if avg_second > avg_first:
        print(f"Improving! (+{avg_second - avg_first:.3f})")
    else:
        print(f"Declining ({avg_second - avg_first:.3f})")

    print("\nDenial Constraint Violations:")
    print(f"Initial violations: {violations[0]:.2f}")
    print(f"Final violations: {violations[-1]:.2f}")
    print(f"Change: {violations[-1] - violations[0]:+.2f}")

    avg_viol_first = sum(violations[:len(violations)//2]) / len(violations[:len(violations)//2])
    avg_viol_second = sum(violations[len(violations)//2:]) / len(violations[len(violations)//2:])

    if avg_viol_second < avg_viol_first:
        print(f"Learning to avoid violations! ({avg_viol_first:.2f} → {avg_viol_second:.2f})")
    else:
        print(f"More violations ({avg_viol_first:.2f} → {avg_viol_second:.2f})")

    print("\nCorrectness (Success Rate):")
    print(f"Initial success: {success_rates[0]:.1%}")
    print(f"Final success: {success_rates[-1]:.1%}")
    print(f"Change: {(success_rates[-1] - success_rates[0])*100:+.1f} percentage points")

    avg_succ_first = sum(success_rates[:len(success_rates)//2]) / len(success_rates[:len(success_rates)//2])
    avg_succ_second = sum(success_rates[len(success_rates)//2:]) / len(success_rates[len(success_rates)//2:])

    if avg_succ_second > avg_succ_first:
        print(f"Code quality improving! ({avg_succ_first:.1%} → {avg_succ_second:.1%})")
    else:
        print(f"Code quality declining ({avg_succ_first:.1%} → {avg_succ_second:.1%})")

    print("\nTraining Loss:")
    print(f"Initial loss: {losses[0]:.3f}")
    print(f"Final loss: {losses[-1]:.3f}")
    print(f"Change: {losses[-1] - losses[0]:+.3f}")

    # Overall interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    is_learning = avg_second > avg_first
    reducing_violations = avg_viol_second < avg_viol_first
    improving_correctness = avg_succ_second > avg_succ_first

    if is_learning and reducing_violations:
        print(" successs: The model is learning from RL training")
        print("   - Rewards are increasing over time")
        print("   - Fewer constraint violations")
        print("   This validates that denial prompting + RL works.")
    elif is_learning:
        print("partial success: Rewards increasing but violations not reducing")
        print("   - The model is learning something")
        print("   - May need to adjust denial_penalty_weight")
    elif reducing_violations:
        print("partial success: Violations reducing but rewards not increasing")
        print("   - Model learning to avoid violations")
        print("   - May need to adjust correctness_weight")
    else:
        print("inconclusive: Need more training steps")
        print("   - 20 steps may be too few to see clear trends")
        print("   - Try 50-100 steps on NSCC")

    print("\nFindings:")
    print(f"   • Reward trend: {'Improving' if is_learning else 'Declining'}")
    print(f"   • Violation trend: {'Reducing' if reducing_violations else 'Increasing'}")
    print(f"   • Correctness trend: {'Improving' if improving_correctness else 'Declining'}")

    print("\nRecommendations:")
    if is_learning and reducing_violations:
        print("System validated - ready for full NSCC training!")
        print("Config looks good, proceed with 5000 steps")
    else:
        print("   • Run longer test (50-100 steps) to confirm trends")
        print("   • May need hyperparameter tuning:")
        if not reducing_violations:
            print("     - Increase denial_penalty_weight")
        if not is_learning:
            print("     - Adjust learning_rate")

    print("="*80)


def main():
    """Run pilot test."""
    print("="*80)
    print("DENIAL PROMPTING RL - PILOT TEST")
    print("="*80)
    print("Running minimal 20-step training to validate the system")
    print("="*80)

    # Load config
    print("\nLoading pilot configuration...")
    config = load_config("configs/config_pilot.yaml")
    print(f"Config loaded")
    print(f"Model: {config['model']['name']}")
    print(f"Steps: {config['training']['num_steps']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Group size: {config['training']['group_size']}")

    # Load dataset
    print("\nLoading test dataset...")
    try:
        loader = NeoCoderLoader()
        problems = loader.load()
        dataset = NeoCoderDataset(problems, config)
        print(f"Loaded {len(problems)} test problems")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Initialize model
    print("\nInitializing model...")
    print("   (First time will download GPT-2, ~500MB)")
    try:
        model = ModelWrapper(
            model_name=config['model']['name'],
            device=config['model']['device'],
            max_length=config['model']['max_length'],
        )
        print(f"Model loaded: {config['model']['name']}")
        print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Initialize reward function
    print("\nInitialising reward function...")
    reward_fn = RewardFunction(
        correctness_weight=config['reward']['correctness_weight'],
        denial_penalty_weight=config['reward']['denial_penalty_weight'],
        timeout=config['reward']['timeout'],
    )
    print("Reward function ready")

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    output_dir = "./outputs/pilot_test"
    trainer = GRPOTrainer(
        model=model,
        reward_fn=reward_fn,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
    )
    print("Trainer ready")

    # Run training
    print("\n" + "="*80)
    print("STARTING PILOT TRAINING (20 STEPS)")
    print("="*80)
    print("This will take ~5-10 minutes on CPU")
    print("="*80 + "\n")

    start_time = time.time()

    try:
        results = trainer.train()

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        # Analyze results
        analyze_results(output_dir)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
