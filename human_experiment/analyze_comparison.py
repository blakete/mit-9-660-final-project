#!/usr/bin/env python3
"""Compare human experiment data to Bayesian model predictions."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.special import softmax
import pandas as pd

HUMAN_DATA_DIR = Path(__file__).parent / "collected_data"
MODEL_PREDICTIONS_PATH = Path(__file__).parent / "model_predictions.json"
FIGURES_DIR = Path(__file__).parent / "comparison_figures"
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_ORDER = {
    "Class I - Fixed": [32],
    "Class II - Periodic": [5, 108],
    "Class III - Chaotic": [30, 60],
    "Class IV - Complex": [54, 110]
}

RULES_BY_CLASS = []
for class_name, rules in CLASS_ORDER.items():
    for rule in rules:
        RULES_BY_CLASS.append((rule, class_name))

CLASS_COLORS = {
    "Class I - Fixed": "#e74c3c",
    "Class II - Periodic": "#f39c12",
    "Class III - Chaotic": "#9b59b6",
    "Class IV - Complex": "#1abc9c"
}

ZOO_DIR = Path(__file__).parent.parent / "zoo_of_rules"
TRIAL_CONFIG_PATH = Path(__file__).parent / "trial_config.json"
CA_WIDTH = 17


def apply_wolfram_rule(rule_num: int, left: int, center: int, right: int) -> int:
    neighborhood = (left << 2) | (center << 1) | right
    return (rule_num >> neighborhood) & 1

def compute_next_row(current_row: list, rule_num: int) -> list:
    n = len(current_row)
    next_row = []
    for i in range(n):
        left = current_row[(i - 1) % n]
        center = current_row[i]
        right = current_row[(i + 1) % n]
        next_row.append(apply_wolfram_rule(rule_num, left, center, right))
    return next_row

def evolve_ca(initial_row: list, rule_num: int, steps: int) -> list:
    row = initial_row.copy()
    for _ in range(steps):
        row = compute_next_row(row, rule_num)
    return row

def load_initial_row(rule_number: int) -> list:
    config_path = ZOO_DIR / f"rule_{rule_number:03d}" / "run_1_config.json"
    with open(config_path) as f:
        config = json.load(f)
    return [1 if x else 0 for x in config["initial_row"]][:CA_WIDTH]

def load_trial_config() -> dict:
    with open(TRIAL_CONFIG_PATH) as f:
        return json.load(f)


def load_human_data(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)

def load_model_predictions() -> dict:
    with open(MODEL_PREDICTIONS_PATH) as f:
        return json.load(f)

def find_latest_human_data() -> Path:
    json_files = list(HUMAN_DATA_DIR.glob("ca_experiment_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No human data files found in {HUMAN_DATA_DIR}")
    return max(json_files, key=lambda p: p.stat().st_mtime)

def find_all_human_data() -> list[Path]:
    json_files = list(HUMAN_DATA_DIR.glob("ca_experiment_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No human data files found in {HUMAN_DATA_DIR}")
    return sorted(json_files)

def load_all_human_data() -> tuple[dict, list[str]]:
    all_files = find_all_human_data()
    participants = []
    all_trials = []
    
    for filepath in all_files:
        data = load_human_data(filepath)
        participant_id = data.get("participant_id", filepath.stem)
        participants.append(participant_id)
        
        for trial in data["trials"]:
            trial["participant_id"] = participant_id
            all_trials.append(trial)
    
    combined = {
        "participant_ids": participants,
        "num_participants": len(participants),
        "trials": all_trials,
        "num_trials": len(all_trials)
    }
    
    return combined, participants


def normalize_ratings_to_probabilities(ratings: list, method: str = "softmax") -> np.ndarray:
    ratings = np.array(ratings, dtype=float)
    
    if method == "softmax":
        temperature = 1.5
        return softmax((ratings - 4) / temperature)
    elif method == "linear":
        shifted = ratings - 1
        return shifted / shifted.sum() if shifted.sum() > 0 else np.ones_like(shifted) / len(shifted)
    else:
        raise ValueError(f"Unknown method: {method}")

def extract_trial_data(human_data: dict, model_data: dict, trial_config: dict = None) -> pd.DataFrame:
    rows = []
    
    if trial_config is None:
        trial_config = load_trial_config()
    
    model_lookup = {trial["trial_id"]: trial for trial in model_data["trials"]}
    trial_config_lookup = {trial["trial_id"]: trial for trial in trial_config["trials"]}
    initial_row_cache = {}
    
    for human_trial in human_data["trials"]:
        trial_id = human_trial["trial_id"]
        
        if trial_id not in model_lookup:
            print(f"Warning: {trial_id} not found in model predictions")
            continue
            
        model_trial = model_lookup[trial_id]
        config_trial = trial_config_lookup.get(trial_id)
        
        ratings_by_option = {}
        rules_by_option = {}
        correct_option = None
        for rating_data in human_trial["ratings"]:
            opt_id = rating_data["option_id"]
            ratings_by_option[opt_id] = rating_data["rating"]
            rules_by_option[opt_id] = rating_data["candidate_rule"]
            if rating_data["is_correct"]:
                correct_option = opt_id
        
        human_ratings = [ratings_by_option.get(f"opt_{i}", 4) for i in range(4)]
        model_probs = [cp["probability"] for cp in model_trial["candidate_probabilities"]]
        
        correct_idx = None
        for i, cp in enumerate(model_trial["candidate_probabilities"]):
            if cp["is_correct"]:
                correct_idx = i
                break
        
        human_probs = normalize_ratings_to_probabilities(human_ratings, method="softmax")
        
        rule_number = human_trial["rule_number"]
        time_step = human_trial["time_step"]
        
        if rule_number not in initial_row_cache:
            initial_row_cache[rule_number] = load_initial_row(rule_number)
        initial_row = initial_row_cache[rule_number]
        
        current_state = evolve_ca(initial_row, rule_number, time_step)
        
        option_manifests = {}
        for opt_id in [f"opt_{i}" for i in range(4)]:
            opt_rule = rules_by_option.get(opt_id)
            if opt_rule is not None:
                manifest = compute_next_row(current_state, opt_rule)
                option_manifests[opt_id] = manifest
        
        correct_manifest = option_manifests.get(correct_option, [])
        
        manifest_equivalent = {}
        for opt_id, manifest in option_manifests.items():
            manifest_equivalent[opt_id] = (manifest == correct_manifest)
        
        num_equivalent_options = sum(manifest_equivalent.values())
        
        max_rating = 0
        max_is_correct = False
        max_selected_option = None
        sorted_ratings = sorted(human_trial["ratings"], key=lambda x: x["candidate_index"])
        for r in sorted_ratings:
            if r["rating"] > max_rating:
                max_rating = r["rating"]
                max_is_correct = r["is_correct"]
                max_selected_option = r["option_id"]
        human_max_is_correct = max_is_correct
        
        human_max_is_manifest_correct = manifest_equivalent.get(max_selected_option, False)
        
        model_max_idx = np.argmax(model_probs)
        model_max_option = f"opt_{model_max_idx}"
        model_max_is_correct = (model_max_idx == correct_idx)
        model_max_is_manifest_correct = manifest_equivalent.get(model_max_option, False)
        
        human_correct_rating = human_ratings[correct_idx] if correct_idx is not None else None
        model_correct_prob = model_probs[correct_idx] if correct_idx is not None else None
        
        rows.append({
            "trial_id": trial_id,
            "rule_number": human_trial["rule_number"],
            "time_step": human_trial["time_step"],
            "rule_class": human_trial["rule_class"],
            "human_ratings": human_ratings,
            "human_probs": human_probs.tolist(),
            "model_probs": model_probs,
            "human_correct_rating": human_correct_rating,
            "human_correct_prob": human_probs[correct_idx] if correct_idx is not None else None,
            "model_correct_prob": model_correct_prob,
            "human_max_is_correct": human_max_is_correct,
            "model_max_is_correct": model_max_is_correct,
            "human_max_is_manifest_correct": human_max_is_manifest_correct,
            "model_max_is_manifest_correct": model_max_is_manifest_correct,
            "num_equivalent_options": num_equivalent_options,
            "correct_idx": correct_idx
        })
    
    return pd.DataFrame(rows)


def compute_correlation(df: pd.DataFrame) -> dict:
    all_human_ratings = []
    all_model_probs = []
    all_human_probs = []
    
    for _, row in df.iterrows():
        all_human_ratings.extend(row["human_ratings"])
        all_model_probs.extend(row["model_probs"])
        all_human_probs.extend(row["human_probs"])
    
    r_rating_prob, p_rating_prob = stats.pearsonr(all_human_ratings, all_model_probs)
    r_prob_prob, p_prob_prob = stats.pearsonr(all_human_probs, all_model_probs)
    rho_rating, p_rho_rating = stats.spearmanr(all_human_ratings, all_model_probs)
    
    return {
        "pearson_rating_vs_prob": {"r": r_rating_prob, "p": p_rating_prob},
        "pearson_prob_vs_prob": {"r": r_prob_prob, "p": p_prob_prob},
        "spearman_rating_vs_prob": {"rho": rho_rating, "p": p_rho_rating},
        "n_datapoints": len(all_human_ratings)
    }

def compute_accuracy_metrics(df: pd.DataFrame) -> dict:
    human_acc_rule = df["human_max_is_correct"].mean()
    model_acc_rule = df["model_max_is_correct"].mean()
    human_acc_manifest = df["human_max_is_manifest_correct"].mean()
    model_acc_manifest = df["model_max_is_manifest_correct"].mean()
    avg_equivalent = df["num_equivalent_options"].mean()
    
    acc_by_time = df.groupby("time_step").agg({
        "human_max_is_correct": "mean",
        "model_max_is_correct": "mean",
        "human_max_is_manifest_correct": "mean",
        "model_max_is_manifest_correct": "mean",
        "num_equivalent_options": "mean"
    }).to_dict()
    
    acc_by_rule = df.groupby("rule_number").agg({
        "human_max_is_correct": "mean",
        "model_max_is_correct": "mean",
        "human_max_is_manifest_correct": "mean",
        "model_max_is_manifest_correct": "mean",
        "num_equivalent_options": "mean"
    }).to_dict()
    
    return {
        "overall_human_accuracy_rule": human_acc_rule,
        "overall_model_accuracy_rule": model_acc_rule,
        "overall_human_accuracy_manifest": human_acc_manifest,
        "overall_model_accuracy_manifest": model_acc_manifest,
        "avg_equivalent_options": avg_equivalent,
        "accuracy_by_time_step": acc_by_time,
        "accuracy_by_rule": acc_by_rule
    }

def compute_kl_divergence(df: pd.DataFrame) -> dict:
    kl_divs = []
    for _, row in df.iterrows():
        human_p = np.array(row["human_probs"]) + 1e-10
        model_p = np.array(row["model_probs"]) + 1e-10
        human_p /= human_p.sum()
        model_p /= model_p.sum()
        kl = np.sum(human_p * np.log(human_p / model_p))
        kl_divs.append(kl)
    
    return {
        "mean_kl_divergence": np.mean(kl_divs),
        "std_kl_divergence": np.std(kl_divs),
        "kl_by_trial": kl_divs
    }


def plot_scatter_correlation(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    all_human_ratings = []
    all_model_probs = []
    all_human_probs = []
    
    for _, row in df.iterrows():
        all_human_ratings.extend(row["human_ratings"])
        all_model_probs.extend(row["model_probs"])
        all_human_probs.extend(row["human_probs"])
    
    ax1 = axes[0]
    ax1.scatter(all_human_ratings, all_model_probs, alpha=0.5, s=30)
    ax1.set_xlabel("Human Rating (1-7)")
    ax1.set_ylabel("Model Probability")
    ax1.set_title("Human Ratings vs Model Probabilities")
    
    z = np.polyfit(all_human_ratings, all_model_probs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 7, 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"Linear fit")
    
    r, _ = stats.pearsonr(all_human_ratings, all_model_probs)
    ax1.text(0.05, 0.95, f"r = {r:.3f}", transform=ax1.transAxes, 
             verticalalignment='top', fontsize=12)
    
    ax2 = axes[1]
    ax2.scatter(all_human_probs, all_model_probs, alpha=0.5, s=30)
    ax2.set_xlabel("Human Probability (softmax)")
    ax2.set_ylabel("Model Probability")
    ax2.set_title("Human vs Model Probabilities")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
    
    r, _ = stats.pearsonr(all_human_probs, all_model_probs)
    ax2.text(0.05, 0.95, f"r = {r:.3f}", transform=ax2.transAxes,
             verticalalignment='top', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_by_time(df: pd.DataFrame, save_path: Path):
    acc_by_time = df.groupby("time_step").agg({
        "human_max_is_manifest_correct": "mean",
        "model_max_is_manifest_correct": "mean"
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    time_steps = acc_by_time.index.values
    width = 0.35
    x = np.arange(len(time_steps))
    
    ax.bar(x - width/2, acc_by_time["human_max_is_manifest_correct"], width, 
           label="Human", color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, acc_by_time["model_max_is_manifest_correct"], width,
           label="Model", color="#3498db", alpha=0.8)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Accuracy (Manifest-Based)")
    ax.set_title("Accuracy Over Time: Human vs Model")
    ax.set_xticks(x)
    ax.set_xticklabels(time_steps)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_by_time_per_class(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (class_name, rules) in enumerate(CLASS_ORDER.items()):
        ax = axes[idx]
        class_df = df[df["rule_number"].isin(rules)]
        
        if len(class_df) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(class_name)
            continue
        
        acc_by_time = class_df.groupby("time_step").agg({
            "human_max_is_manifest_correct": "mean",
            "model_max_is_manifest_correct": "mean"
        })
        
        time_steps = acc_by_time.index.values
        width = 0.35
        x = np.arange(len(time_steps))
        
        ax.bar(x - width/2, acc_by_time["human_max_is_manifest_correct"], width,
               label="Human", color="#2ecc71", alpha=0.8)
        ax.bar(x + width/2, acc_by_time["model_max_is_manifest_correct"], width,
               label="Model", color="#3498db", alpha=0.8)
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{class_name}\nRules: {', '.join([f'R{r}' for r in rules])}", 
                     color=CLASS_COLORS[class_name], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(time_steps)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
        
        if idx == 0:
            ax.legend(loc='upper left')
    
    plt.suptitle("Accuracy Over Time by Wolfram Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_by_time_per_rule(df: pd.DataFrame, save_path: Path):
    ordered_rules = [r for r, _ in RULES_BY_CLASS]
    n_rules = len(ordered_rules)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, rule in enumerate(ordered_rules):
        ax = axes[idx]
        rule_df = df[df["rule_number"] == rule]
        
        if len(rule_df) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Rule {rule}")
            continue
        
        acc_by_time = rule_df.groupby("time_step").agg({
            "human_max_is_manifest_correct": "mean",
            "model_max_is_manifest_correct": "mean"
        })
        
        rule_class = None
        for class_name, rules in CLASS_ORDER.items():
            if rule in rules:
                rule_class = class_name
                break
        
        time_steps = acc_by_time.index.values
        width = 0.35
        x = np.arange(len(time_steps))
        
        ax.bar(x - width/2, acc_by_time["human_max_is_manifest_correct"], width,
               label="Human", color="#2ecc71", alpha=0.8)
        ax.bar(x + width/2, acc_by_time["model_max_is_manifest_correct"], width,
               label="Model", color="#3498db", alpha=0.8)
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Accuracy")
        
        title_color = CLASS_COLORS.get(rule_class, 'black') if rule_class else 'black'
        short_class = rule_class.split(" - ")[0] if rule_class else ""
        ax.set_title(f"Rule {rule} ({short_class})", color=title_color, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(time_steps)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    for idx in range(n_rules, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Accuracy Over Time by Individual Rule", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_correct_option_ratings(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    correct_by_time = df.groupby("time_step")["human_correct_rating"].mean()
    ax1.bar(correct_by_time.index, correct_by_time.values, color="#2ecc71", alpha=0.8)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Mean Rating for Correct Option")
    ax1.set_title("Human Confidence in Correct Answer Over Time")
    ax1.set_ylim(1, 7)
    ax1.axhline(y=4, color='gray', linestyle='--', alpha=0.5)
    
    ax2 = axes[1]
    prob_by_time = df.groupby("time_step")["model_correct_prob"].mean()
    ax2.bar(prob_by_time.index, prob_by_time.values, color="#3498db", alpha=0.8)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Mean Probability for Correct Option")
    ax2.set_title("Model Confidence in Correct Answer Over Time")
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_by_rule(df: pd.DataFrame, save_path: Path):
    acc_by_rule = df.groupby("rule_number").agg({
        "human_max_is_manifest_correct": "mean",
        "model_max_is_manifest_correct": "mean",
        "rule_class": "first"
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ordered_rules = [r for r, _ in RULES_BY_CLASS]
    width = 0.35
    x = np.arange(len(ordered_rules))
    
    human_acc = [acc_by_rule.loc[r, "human_max_is_manifest_correct"] if r in acc_by_rule.index else 0 for r in ordered_rules]
    model_acc = [acc_by_rule.loc[r, "model_max_is_manifest_correct"] if r in acc_by_rule.index else 0 for r in ordered_rules]
    
    ax.bar(x - width/2, human_acc, width, label="Human", color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, model_acc, width, label="Model", color="#3498db", alpha=0.8)
    
    class_start = 0
    class_positions = []
    for class_name, rules in CLASS_ORDER.items():
        class_end = class_start + len(rules)
        ax.axvspan(class_start - 0.5, class_end - 0.5, 
                   alpha=0.1, color=CLASS_COLORS[class_name])
        class_positions.append((class_start + (len(rules) - 1) / 2, class_name))
        if class_end < len(ordered_rules):
            ax.axvline(x=class_end - 0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        class_start = class_end
    
    ax.set_xlabel("Rule Number (grouped by Wolfram Class)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by CA Rule: Human vs Model (Grouped by Class)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r}" for r in ordered_rules])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    for pos, class_name in class_positions:
        short_name = class_name.split(" - ")[0]
        ax.text(pos, 1.08, short_name, ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color=CLASS_COLORS[class_name])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_distribution_comparison(df: pd.DataFrame, save_path: Path):
    sample_trials = df[df["time_step"].isin([1, 3, 6])].head(6)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(sample_trials.iterrows()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        x = np.arange(4)
        width = 0.35
        
        ax.bar(x - width/2, row["human_probs"], width, 
               label="Human", color="#2ecc71", alpha=0.8)
        ax.bar(x + width/2, row["model_probs"], width,
               label="Model", color="#3498db", alpha=0.8)
        
        correct_idx = row["correct_idx"]
        if correct_idx is not None:
            ax.axvline(x=correct_idx, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Option")
        ax.set_ylabel("Probability")
        ax.set_title(f"{row['trial_id']}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"opt_{i}" for i in range(4)])
        
        if idx == 0:
            ax.legend(loc='upper right')
    
    plt.suptitle("Example Trial Distributions: Human vs Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_heatmap(df: pd.DataFrame, save_path: Path):
    ordered_rules = [r for r, _ in RULES_BY_CLASS]
    time_steps = sorted(df["time_step"].unique())
    
    human_matrix = np.zeros((len(ordered_rules), len(time_steps)))
    model_matrix = np.zeros((len(ordered_rules), len(time_steps)))
    
    for i, rule in enumerate(ordered_rules):
        for j, t in enumerate(time_steps):
            subset = df[(df["rule_number"] == rule) & (df["time_step"] == t)]
            if len(subset) > 0:
                human_matrix[i, j] = subset["human_max_is_manifest_correct"].mean()
                model_matrix[i, j] = subset["model_max_is_manifest_correct"].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#e74c3c', '#f1c40f', '#2ecc71']
    cmap = LinearSegmentedColormap.from_list('accuracy', colors)
    
    ax1 = axes[0]
    im1 = ax1.imshow(human_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax1.set_title("Human Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Rule (grouped by class)")
    ax1.set_xticks(np.arange(len(time_steps)))
    ax1.set_xticklabels(time_steps)
    ax1.set_yticks(np.arange(len(ordered_rules)))
    ax1.set_yticklabels([f"R{r}" for r in ordered_rules])
    
    for i in range(len(ordered_rules)):
        for j in range(len(time_steps)):
            val = human_matrix[i, j]
            text_color = 'white' if val < 0.4 or val > 0.7 else 'black'
            ax1.text(j, i, f"{val:.0%}", ha='center', va='center', 
                    color=text_color, fontsize=9, fontweight='bold')
    
    ax2 = axes[1]
    im2 = ax2.imshow(model_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax2.set_title("Model Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Rule (grouped by class)")
    ax2.set_xticks(np.arange(len(time_steps)))
    ax2.set_xticklabels(time_steps)
    ax2.set_yticks(np.arange(len(ordered_rules)))
    ax2.set_yticklabels([f"R{r}" for r in ordered_rules])
    
    for i in range(len(ordered_rules)):
        for j in range(len(time_steps)):
            val = model_matrix[i, j]
            text_color = 'white' if val < 0.4 or val > 0.7 else 'black'
            ax2.text(j, i, f"{val:.0%}", ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')
    
    class_start = 0
    for class_name, rules in CLASS_ORDER.items():
        class_end = class_start + len(rules)
        mid_y = (class_start + class_end - 1) / 2
        
        if class_start > 0:
            ax1.axhline(y=class_start - 0.5, color='white', linewidth=2)
            ax2.axhline(y=class_start - 0.5, color='white', linewidth=2)
        
        short_name = class_name.split(" - ")[0]
        ax1.text(-0.8, mid_y, short_name, ha='right', va='center', fontsize=9,
                fontweight='bold', color=CLASS_COLORS[class_name])
        
        class_start = class_end
    
    fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04,
                 label='Accuracy')
    
    plt.suptitle("Accuracy by Rule and Time Step: Human vs Model", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_gap_heatmap(df: pd.DataFrame, save_path: Path):
    ordered_rules = [r for r, _ in RULES_BY_CLASS]
    time_steps = sorted(df["time_step"].unique())
    
    gap_matrix = np.zeros((len(ordered_rules), len(time_steps)))
    
    for i, rule in enumerate(ordered_rules):
        for j, t in enumerate(time_steps):
            subset = df[(df["rule_number"] == rule) & (df["time_step"] == t)]
            if len(subset) > 0:
                human_acc = subset["human_max_is_manifest_correct"].mean()
                model_acc = subset["model_max_is_manifest_correct"].mean()
                gap_matrix[i, j] = human_acc - model_acc
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(gap_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_title("Accuracy Gap (Human - Model)\nBlue = Human better, Red = Model better", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Rule (grouped by class)")
    ax.set_xticks(np.arange(len(time_steps)))
    ax.set_xticklabels(time_steps)
    ax.set_yticks(np.arange(len(ordered_rules)))
    ax.set_yticklabels([f"R{r}" for r in ordered_rules])
    
    for i in range(len(ordered_rules)):
        for j in range(len(time_steps)):
            val = gap_matrix[i, j]
            text_color = 'white' if abs(val) > 0.4 else 'black'
            sign = "+" if val > 0 else ""
            ax.text(j, i, f"{sign}{val:.0%}", ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    class_start = 0
    for class_name, rules in CLASS_ORDER.items():
        class_end = class_start + len(rules)
        mid_y = (class_start + class_end - 1) / 2
        
        if class_start > 0:
            ax.axhline(y=class_start - 0.5, color='black', linewidth=2)
        
        short_name = class_name.split(" - ")[0]
        ax.text(-0.8, mid_y, short_name, ha='right', va='center', fontsize=9,
               fontweight='bold', color=CLASS_COLORS[class_name])
        
        class_start = class_end
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Human - Model Accuracy', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_manifest_equivalence_by_class(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_data = []
    for class_name, rules in CLASS_ORDER.items():
        class_df = df[df["rule_number"].isin(rules)]
        if len(class_df) > 0:
            avg_equiv = class_df["num_equivalent_options"].mean()
            class_data.append({
                "class": class_name,
                "avg_equivalent": avg_equiv,
                "color": CLASS_COLORS[class_name]
            })
    
    x = np.arange(len(class_data))
    bars = ax.bar(x, [d["avg_equivalent"] for d in class_data],
                  color=[d["color"] for d in class_data], alpha=0.8)
    
    ax.set_xlabel("Wolfram Class")
    ax.set_ylabel("Avg # of Visually Equivalent Options")
    ax.set_title("Manifest Equivalence by Rule Class\n(Higher = more options look identical to correct answer)",
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d["class"].split(" - ")[0] for d in class_data])
    ax.set_ylim(0, 4.5)
    
    for bar, d in zip(bars, class_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No equivalence (1)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_human_vs_model_by_class(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_data = []
    for class_name, rules in CLASS_ORDER.items():
        class_df = df[df["rule_number"].isin(rules)]
        if len(class_df) > 0:
            class_data.append({
                "class": class_name,
                "human": class_df["human_max_is_manifest_correct"].mean(),
                "model": class_df["model_max_is_manifest_correct"].mean(),
                "color": CLASS_COLORS[class_name]
            })
    
    x = np.arange(len(class_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [d["human"] for d in class_data], width,
                   label="Human", color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(x + width/2, [d["model"] for d in class_data], width,
                   label="Model", color="#3498db", alpha=0.8)
    
    ax.set_xlabel("Wolfram Class", fontsize=12)
    ax.set_ylabel("Accuracy (Manifest-Based)", fontsize=12)
    ax.set_title("Human vs Model Accuracy by Wolfram Class", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d["class"].split(" - ")[0] + "\n" + d["class"].split(" - ")[1] 
                        for d in class_data], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax.legend(loc='upper right')
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_by_time_manifest(df: pd.DataFrame, save_path: Path):
    acc_by_time = df.groupby("time_step").agg({
        "human_max_is_manifest_correct": "mean",
        "model_max_is_manifest_correct": "mean"
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    time_steps = acc_by_time.index.values
    width = 0.35
    x = np.arange(len(time_steps))
    
    ax.bar(x - width/2, acc_by_time["human_max_is_manifest_correct"], width, 
           label="Human", color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, acc_by_time["model_max_is_manifest_correct"], width,
           label="Model", color="#3498db", alpha=0.8)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Accuracy (Manifest-Based)")
    ax.set_title("Manifest-Based Accuracy Over Time: Human vs Model", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(time_steps)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("="*60)
    print("Human vs Model Comparison Analysis (Aggregated)")
    print("="*60)
    print()
    
    print("Loading data...")
    human_data, participants = load_all_human_data()
    model_data = load_model_predictions()
    
    print(f"  Participants ({len(participants)}): {', '.join(participants)}")
    print(f"  Human trials: {len(human_data['trials'])} ({len(human_data['trials'])//len(participants)} per participant)")
    print(f"  Model trials: {len(model_data['trials'])}")
    print()
    
    print("Processing trial data...")
    df = extract_trial_data(human_data, model_data)
    print(f"  Aligned trials: {len(df)}")
    print()
    
    print("Computing metrics...")
    
    correlation = compute_correlation(df)
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson (rating vs prob): r={correlation['pearson_rating_vs_prob']['r']:.3f}, "
          f"p={correlation['pearson_rating_vs_prob']['p']:.4f}")
    print(f"  Pearson (prob vs prob):   r={correlation['pearson_prob_vs_prob']['r']:.3f}, "
          f"p={correlation['pearson_prob_vs_prob']['p']:.4f}")
    print(f"  Spearman (rating vs prob): ρ={correlation['spearman_rating_vs_prob']['rho']:.3f}, "
          f"p={correlation['spearman_rating_vs_prob']['p']:.4f}")
    
    accuracy = compute_accuracy_metrics(df)
    print(f"\nAccuracy - Rule-Based (original metric):")
    print(f"  Human: {accuracy['overall_human_accuracy_rule']*100:.1f}%")
    print(f"  Model: {accuracy['overall_model_accuracy_rule']*100:.1f}%")
    print(f"  Chance: 25.0%")
    
    print(f"\nAccuracy - Manifest-Based (fair metric):")
    print(f"  Human: {accuracy['overall_human_accuracy_manifest']*100:.1f}%")
    print(f"  Model: {accuracy['overall_model_accuracy_manifest']*100:.1f}%")
    print(f"  Avg equivalent options per trial: {accuracy['avg_equivalent_options']:.2f}")
    
    kl = compute_kl_divergence(df)
    print(f"\nKL Divergence (Human || Model):")
    print(f"  Mean: {kl['mean_kl_divergence']:.3f}")
    print(f"  Std:  {kl['std_kl_divergence']:.3f}")
    
    print("\nGenerating visualizations...")
    
    plot_scatter_correlation(df, FIGURES_DIR / "scatter_correlation.png")
    plot_accuracy_by_time(df, FIGURES_DIR / "accuracy_by_time.png")
    plot_correct_option_ratings(df, FIGURES_DIR / "correct_option_confidence.png")
    plot_accuracy_by_rule(df, FIGURES_DIR / "accuracy_by_rule.png")
    plot_distribution_comparison(df, FIGURES_DIR / "distribution_examples.png")
    
    plot_accuracy_by_time_per_class(df, FIGURES_DIR / "accuracy_by_time_per_class.png")
    plot_accuracy_by_time_per_rule(df, FIGURES_DIR / "accuracy_by_time_per_rule.png")
    plot_accuracy_heatmap(df, FIGURES_DIR / "accuracy_heatmap.png")
    plot_accuracy_gap_heatmap(df, FIGURES_DIR / "accuracy_gap_heatmap.png")
    
    plot_manifest_equivalence_by_class(df, FIGURES_DIR / "manifest_equivalence_by_class.png")
    plot_human_vs_model_by_class(df, FIGURES_DIR / "human_vs_model_by_class.png")
    
    summary = {
        "participants": participants,
        "num_participants": len(participants),
        "num_trials": len(df),
        "trials_per_participant": len(df) // len(participants),
        "correlation": correlation,
        "accuracy_rule_based": {
            "human": accuracy["overall_human_accuracy_rule"],
            "model": accuracy["overall_model_accuracy_rule"]
        },
        "accuracy_manifest_based": {
            "human": accuracy["overall_human_accuracy_manifest"],
            "model": accuracy["overall_model_accuracy_manifest"],
            "avg_equivalent_options": accuracy["avg_equivalent_options"]
        },
        "kl_divergence": {
            "mean": kl["mean_kl_divergence"],
            "std": kl["std_kl_divergence"]
        }
    }
    
    summary_path = FIGURES_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
