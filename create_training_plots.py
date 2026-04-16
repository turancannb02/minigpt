"""
Create training plots from output_minigpt.log for the README.
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: str):
    """Parse the training log file and extract metrics."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Split by training runs (look for "Starting training" or "Model parameters")
    runs = []

    # Pattern to match training steps
    train_step_pattern = r'Step (\d+)/\d+ \| loss=([\d.]+) \| lr=([\d.e-]+)'
    val_step_pattern = r'=== Step (\d+) \| val_loss=([\d.]+) ==='

    # Find all training runs
    current_run = {
        'train_steps': [],
        'train_losses': [],
        'learning_rates': [],
        'val_steps': [],
        'val_losses': [],
        'model_size': None,
        'params': None
    }

    for line in content.split('\n'):
        # Check for model parameters line to identify runs
        if 'Model parameters:' in line:
            if current_run['train_steps']:  # Save previous run
                runs.append(current_run)
            match = re.search(r'Model parameters: ([\d,]+)', line)
            params = int(match.group(1).replace(',', '')) if match else 0

            # Determine model size from params
            if params < 2000000:
                size = 'Tiny (1.5M)'
            elif params < 15000000:
                size = 'Small (6M)'
            else:
                size = f'Medium ({params//1000000}M)'

            current_run = {
                'train_steps': [],
                'train_losses': [],
                'learning_rates': [],
                'val_steps': [],
                'val_losses': [],
                'model_size': size,
                'params': params
            }

        # Parse training steps
        train_match = re.search(train_step_pattern, line)
        if train_match:
            step = int(train_match.group(1))
            loss = float(train_match.group(2))
            lr = float(train_match.group(3))
            current_run['train_steps'].append(step)
            current_run['train_losses'].append(loss)
            current_run['learning_rates'].append(lr)

        # Parse validation steps
        val_match = re.search(val_step_pattern, line)
        if val_match:
            step = int(val_match.group(1))
            loss = float(val_match.group(2))
            current_run['val_steps'].append(step)
            current_run['val_losses'].append(loss)

    # Add last run
    if current_run['train_steps']:
        runs.append(current_run)

    return runs


def create_plots(runs, output_dir: str = "docs"):
    """Create training plots and save to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Colors for different runs
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    # 1. Training Loss Over Time - All Runs
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, run in enumerate(runs):
        if run['train_steps']:
            ax.plot(run['train_steps'], run['train_losses'],
                   label=run['model_size'], color=colors[i % len(colors)], linewidth=2, alpha=0.7)

    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Validation Loss Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, run in enumerate(runs):
        if run['val_steps']:
            ax.plot(run['val_steps'], run['val_losses'],
                   label=run['model_size'], marker='o', color=colors[i % len(colors)],
                   linewidth=2, markersize=6)

    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'validation_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Learning Rate Schedule (from first run)
    if runs and runs[0]['train_steps']:
        fig, ax = plt.subplots(figsize=(12, 4))
        run = runs[0]
        ax.plot(run['train_steps'], run['learning_rates'],
               color=colors[0], linewidth=2)
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Schedule (Cosine with Warmup)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'learning_rate.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Combined Training + Validation Loss (Tiny model)
    if runs:
        fig, ax = plt.subplots(figsize=(12, 6))
        run = runs[0]  # Tiny model

        if run['train_steps']:
            ax.plot(run['train_steps'], run['train_losses'],
                   label='Training Loss', color=colors[0], linewidth=2, alpha=0.7)

        if run['val_steps']:
            ax.plot(run['val_steps'], run['val_losses'],
                   label='Validation Loss', color=colors[1], marker='o',
                   linewidth=2, markersize=6)

        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Training vs Validation Loss - {run["model_size"]}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'train_val_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Model Comparison Bar Chart
    if len(runs) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))

        models = []
        final_vals = []
        for run in runs:
            if run['val_losses']:
                models.append(run['model_size'])
                final_vals.append(run['val_losses'][-1])

        x = np.arange(len(models))
        bars = ax.bar(x, final_vals, color=colors[:len(models)], alpha=0.7)

        ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Validation Loss', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: Final Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, final_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Plots saved to {output_path}/")
    print(f"   - training_loss.png")
    print(f"   - validation_loss.png")
    print(f"   - learning_rate.png")
    print(f"   - train_val_comparison.png")
    print(f"   - model_comparison.png")


if __name__ == "__main__":
    log_file = "output_minigpt.log"
    runs = parse_log_file(log_file)

    print(f"Found {len(runs)} training run(s):")
    for run in runs:
        print(f"  - {run['model_size']}: {len(run['train_steps'])} training steps, {len(run['val_steps'])} validation points")

    create_plots(runs)
