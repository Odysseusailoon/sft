#!/usr/bin/env python3
"""
RedBlackBench SFT Data Preparation Script v3

Converts HuggingFace dataset to training format for Qwen 14B.
Uses the actual prompt structure from redblackbench scenarios.

Key insight: The HuggingFace data's "input" field already contains the complete
scenario briefing (system prompt) embedded in the user message. We should NOT
add another system prompt on top.

Data Format:
- input: Contains scenario briefing + game state + teammate deliberations
- output: Reasoning + VOTE: A or B
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re

# Minimal system prompt - the scenario details are already in the input
MINIMAL_SYSTEM_PROMPT = """You are a strategic advisor in a multi-round negotiation game. 
Your task is to analyze the situation, consider your teammates' views, provide reasoned analysis, and vote A or B.
Format your final vote as: VOTE: A or VOTE: B"""


def load_datasets(data_dir: str = "hf_dataset") -> List[Dict]:
    """Load all JSON data files."""
    all_examples = []
    
    for fname in Path(data_dir).glob("*.json"):
        print(f"Loading {fname}...")
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = data.get('examples', [])
        print(f"  - {len(examples)} examples")
        all_examples.extend(examples)
    
    print(f"\nTotal examples loaded: {len(all_examples)}")
    return all_examples


def convert_to_chat_format(example: Dict, use_system_prompt: bool = True) -> Dict:
    """
    Convert a single example to chat format.
    
    The input already contains the full scenario briefing, so we use a minimal
    system prompt or no system prompt at all.
    """
    agent_name = example.get('agent_name', 'Advisor')
    scenario_id = example.get('metadata', {}).get('scenario_id', 'unknown')
    round_index = example.get('round_index', 0)
    turn = example.get('turn', 0)
    
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    if use_system_prompt:
        messages = [
            {"role": "system", "content": MINIMAL_SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
    else:
        # No system prompt - just user/assistant
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
    
    return {
        "messages": messages,
        "metadata": {
            "trajectory_id": example.get('trajectory_id', ''),
            "round_index": round_index,
            "turn": turn,
            "agent_name": agent_name,
            "scenario_id": scenario_id,
        }
    }


def analyze_data(examples: List[Dict]) -> Dict:
    """Analyze data quality and distribution."""
    stats = {
        "total": len(examples),
        "by_scenario": defaultdict(int),
        "by_round": defaultdict(int),
        "by_turn": defaultdict(int),
        "by_agent": defaultdict(int),
        "vote_a": 0,
        "vote_b": 0,
        "input_lengths": [],
        "output_lengths": [],
    }
    
    vote_pattern = re.compile(r'VOTE:\s*([AB])', re.IGNORECASE)
    
    for ex in examples:
        scenario = ex.get('metadata', {}).get('scenario_id', 'unknown')
        stats['by_scenario'][scenario] += 1
        stats['by_round'][ex.get('round_index', 0)] += 1
        stats['by_turn'][ex.get('turn', 0)] += 1
        stats['by_agent'][ex.get('agent_name', 'unknown')] += 1
        
        output = ex.get('output', '')
        input_text = ex.get('input', '')
        
        match = vote_pattern.search(output)
        if match:
            vote = match.group(1).upper()
            if vote == 'A':
                stats['vote_a'] += 1
            else:
                stats['vote_b'] += 1
        
        stats['input_lengths'].append(len(input_text))
        stats['output_lengths'].append(len(output))
    
    return stats


def split_by_scenario_and_trajectory(
    examples: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data ensuring:
    1. Same trajectory stays in same split
    2. Balanced scenario distribution across splits
    """
    random.seed(seed)
    
    # Group by scenario, then by trajectory
    scenario_trajectories = defaultdict(lambda: defaultdict(list))
    
    for ex in examples:
        scenario = ex.get('metadata', {}).get('scenario_id', 'unknown')
        tid = ex.get('metadata', {}).get('trajectory_id', 'unknown')
        scenario_trajectories[scenario][tid].append(ex)
    
    train_examples = []
    val_examples = []
    test_examples = []
    
    print("\nSplitting by scenario:")
    for scenario, trajectories in sorted(scenario_trajectories.items()):
        trajectory_ids = list(trajectories.keys())
        random.shuffle(trajectory_ids)
        
        n = len(trajectory_ids)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_tids = set(trajectory_ids[:train_end])
        val_tids = set(trajectory_ids[train_end:val_end])
        test_tids = set(trajectory_ids[val_end:])
        
        scenario_train = []
        scenario_val = []
        scenario_test = []
        
        for tid, exs in trajectories.items():
            if tid in train_tids:
                scenario_train.extend(exs)
            elif tid in val_tids:
                scenario_val.extend(exs)
            else:
                scenario_test.extend(exs)
        
        print(f"  {scenario}: train={len(scenario_train)}, val={len(scenario_val)}, test={len(scenario_test)}")
        
        train_examples.extend(scenario_train)
        val_examples.extend(scenario_val)
        test_examples.extend(scenario_test)
    
    return train_examples, val_examples, test_examples


def save_jsonl(data: List[Dict], filepath: str):
    """Save to JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} examples to {filepath}")


def save_by_scenario(examples: List[Dict], output_dir: str, split_name: str):
    """Save examples split by scenario for targeted evaluation."""
    scenario_data = defaultdict(list)
    for ex in examples:
        scenario = ex.get('metadata', {}).get('scenario_id', 'unknown')
        scenario_data[scenario].append(ex)
    
    for scenario, data in sorted(scenario_data.items()):
        filepath = os.path.join(output_dir, f"{split_name}_{scenario}.jsonl")
        save_jsonl(data, filepath)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare RedBlackBench data for SFT v3")
    parser.add_argument("--data_dir", default="hf_dataset", help="Directory with JSON files")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_by_scenario", action="store_true", 
                       help="Also save separate files by scenario")
    parser.add_argument("--no_system_prompt", action="store_true",
                       help="Don't add system prompt (data already contains scenario briefing)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load raw data
    raw_examples = load_datasets(args.data_dir)
    
    # Analyze raw data
    print("\n" + "="*70)
    print("Raw Data Analysis")
    print("="*70)
    raw_stats = analyze_data(raw_examples)
    print(f"Total samples: {raw_stats['total']}")
    print(f"\nBy Scenario:")
    for s, c in sorted(raw_stats['by_scenario'].items()):
        print(f"  {s}: {c} ({100*c/raw_stats['total']:.1f}%)")
    print(f"\nVote Distribution:")
    print(f"  VOTE A: {raw_stats['vote_a']} ({100*raw_stats['vote_a']/raw_stats['total']:.1f}%)")
    print(f"  VOTE B: {raw_stats['vote_b']} ({100*raw_stats['vote_b']/raw_stats['total']:.1f}%)")
    print(f"\nLength Statistics:")
    print(f"  Input: avg={sum(raw_stats['input_lengths'])/len(raw_stats['input_lengths']):.0f} chars")
    print(f"  Output: avg={sum(raw_stats['output_lengths'])/len(raw_stats['output_lengths']):.0f} chars")
    
    # Convert to chat format
    print("\n" + "="*70)
    print("Converting to Chat Format")
    print("="*70)
    use_system = not args.no_system_prompt
    print(f"Using system prompt: {use_system}")
    chat_examples = [convert_to_chat_format(ex, use_system_prompt=use_system) for ex in raw_examples]
    print(f"Converted {len(chat_examples)} examples")
    
    # Split data
    print("\n" + "="*70)
    print("Splitting Data by Scenario")
    print("="*70)
    train, val, test = split_by_scenario_and_trajectory(
        chat_examples,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"\nFinal Split:")
    print(f"  Train: {len(train)} ({100*len(train)/len(chat_examples):.1f}%)")
    print(f"  Val: {len(val)} ({100*len(val)/len(chat_examples):.1f}%)")
    print(f"  Test: {len(test)} ({100*len(test)/len(chat_examples):.1f}%)")
    
    # Save main files
    save_jsonl(train, os.path.join(args.output_dir, "redblack_train.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "redblack_val.jsonl"))
    save_jsonl(test, os.path.join(args.output_dir, "redblack_test.jsonl"))
    
    # Optionally save by scenario
    if args.save_by_scenario:
        print("\n" + "="*70)
        print("Saving by Scenario")
        print("="*70)
        save_by_scenario(train, args.output_dir, "train")
        save_by_scenario(val, args.output_dir, "val")
        save_by_scenario(test, args.output_dir, "test")
    
    # Save statistics
    stats = {
        "total_examples": len(chat_examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "scenarios": dict(raw_stats['by_scenario']),
        "vote_distribution": {
            "A": raw_stats['vote_a'],
            "B": raw_stats['vote_b']
        },
        "use_system_prompt": use_system,
        "system_prompt": MINIMAL_SYSTEM_PROMPT if use_system else None,
    }
    with open(os.path.join(args.output_dir, "data_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {os.path.join(args.output_dir, 'data_stats.json')}")
    
    # Show sample
    print("\n" + "="*70)
    print("Sample Output")
    print("="*70)
    if train:
        sample = train[0]
        print(f"\nMessages count: {len(sample['messages'])}")
        for i, msg in enumerate(sample['messages']):
            print(f"\n--- Message {i+1}: {msg['role'].upper()} ---")
            content = msg['content']
            if len(content) > 500:
                print(content[:500] + "...")
            else:
                print(content)


if __name__ == "__main__":
    main()

