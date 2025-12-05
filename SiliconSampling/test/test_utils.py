"""Test utilities for persona validation."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def parse_synthetic_responses(responses):
    """Parse synthetic responses into DataFrame format."""
    data = []
    skipped_personas = []
    
    for resp in responses:
        if resp['responses'] == 'FAILED':
            skipped_personas.append((resp['persona_id'], 'FAILED response'))
            continue
        
        row = {'persona_id': resp['persona_id']}
        try:
            for item_resp in resp['responses']:
                if 'statement_id' not in item_resp:
                    # Check for malformed keys
                    malformed_keys = [k for k in item_resp.keys() if 'statement_id' in k]
                    if malformed_keys:
                        raise KeyError(f"Malformed key '{malformed_keys[0]}' instead of 'statement_id'")
                    else:
                        raise KeyError("Missing 'statement_id' key")
                
                q_id = item_resp['statement_id']
                answer = item_resp.get('answer', 3)
                if isinstance(answer, str):
                    answer = 3
                row[f"Q{q_id}"] = answer
            
            data.append(row)
            
        except Exception as e:
            skipped_personas.append((resp['persona_id'], str(e)))
            continue
    
    # Print summary
    total = len(responses)
    loaded = len(data)
    skipped = len(skipped_personas)
    
    print(f"\n{'='*80}")
    print(f"PARSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total responses:        {total:,}")
    print(f"Successfully loaded:    {loaded:,} ({loaded/total*100:.1f}%)")
    print(f"Skipped (malformed):    {skipped:,} ({skipped/total*100:.1f}%)")
    
    if skipped_personas:
        print(f"\nMalformed persona IDs:")
        for persona_id, reason in skipped_personas:
            print(f"  - {persona_id}: {reason}")
    print(f"{'='*80}\n")
    
    return pd.DataFrame(data)


def map_question_to_trait(question_num):
    """Map question number (1-50) to IPIP-50 trait column (E1-E10, N1-N10, etc.)."""
    if question_num <= 10:
        return f'E{question_num}'
    elif question_num <= 20:
        return f'N{question_num - 10}'
    elif question_num <= 30:
        return f'A{question_num - 20}'
    elif question_num <= 40:
        return f'C{question_num - 30}'
    else:
        return f'O{question_num - 40}'


def calculate_mae(synthetic_df, ground_truth_df):
    """Calculate Mean Absolute Error per item and overall."""
    test_cols = [col for col in synthetic_df.columns if col.startswith('Q')]
    mae_per_item = {}
    
    for col in test_cols:
        item_num = int(col.replace('Q', ''))
        ground_col = map_question_to_trait(item_num)
        
        if ground_col in ground_truth_df.columns:
            mae_per_item[col] = abs(synthetic_df[col].mean() - ground_truth_df[ground_col].mean())
    
    return mae_per_item, np.mean(list(mae_per_item.values()))


def calculate_js_divergence(synthetic_df, ground_truth_df):
    """Calculate Jensen-Shannon Divergence per item and overall."""
    test_cols = [col for col in synthetic_df.columns if col.startswith('Q')]
    js_per_item = {}
    epsilon = 1e-10
    
    for col in test_cols:
        item_num = int(col.replace('Q', ''))
        ground_col = map_question_to_trait(item_num)
        
        if ground_col in ground_truth_df.columns:
            synth_dist = synthetic_df[col].value_counts(normalize=True).sort_index()
            truth_dist = ground_truth_df[ground_col].value_counts(normalize=True).sort_index()
            
            all_values = sorted(set(synth_dist.index) | set(truth_dist.index))
            synth_probs = np.array([synth_dist.get(v, 0) for v in all_values]) + epsilon
            truth_probs = np.array([truth_dist.get(v, 0) for v in all_values]) + epsilon
            
            synth_probs /= synth_probs.sum()
            truth_probs /= truth_probs.sum()
            
            mixture = (synth_probs + truth_probs) / 2
            js_per_item[col] = (stats.entropy(synth_probs, mixture, base=2) + 
                               stats.entropy(truth_probs, mixture, base=2)) / 2
    
    return js_per_item, np.mean(list(js_per_item.values()))


def calculate_correlation(synthetic_df, ground_truth_df):
    """Calculate correlation between synthetic and ground truth means."""
    test_cols = [col for col in synthetic_df.columns if col.startswith('Q')]
    synth_means = []
    truth_means = []
    
    for col in test_cols:
        item_num = int(col.replace('Q', ''))
        ground_col = map_question_to_trait(item_num)
        
        if ground_col in ground_truth_df.columns:
            synth_means.append(synthetic_df[col].mean())
            truth_means.append(ground_truth_df[ground_col].mean())
    
    correlation = np.corrcoef(synth_means, truth_means)[0, 1] if len(synth_means) > 1 else 0
    return correlation, synth_means, truth_means


def calculate_ks_test(synthetic_df, ground_truth_df):
    """
    K-S test on distribution of 50 question mean scores.
    Tests if synthetic and ground truth question means follow same distribution.
    """
    test_cols = [col for col in synthetic_df.columns if col.startswith('Q')]
    synth_means = []
    truth_means = []
    
    for col in test_cols:
        item_num = int(col.replace('Q', ''))
        ground_col = map_question_to_trait(item_num)
        
        if ground_col in ground_truth_df.columns:
            synth_means.append(synthetic_df[col].mean())
            truth_means.append(ground_truth_df[ground_col].mean())
    
    statistic, p_value = stats.ks_2samp(synth_means, truth_means)
    return statistic, p_value, p_value > 0.05


def load_item_metadata():
    """Load item keying information from test_questions.json."""
    questions_path = Path(__file__).parent / 'test_questions.json'
    with open(questions_path) as f:
        questions = json.load(f)
    
    metadata = {}
    for q in questions:
        metadata[q['question_id']] = {
            'trait': q['trait'],
            'reverse': q.get('reverse', False)
        }
    return metadata


def calculate_persona_consistency(synthetic_df):
    """
    Calculate persona-level consistency metrics.
    These test whether responses show coherent personality patterns,
    independent of population distributions.
    """
    metadata = load_item_metadata()
    test_cols = [col for col in synthetic_df.columns if col.startswith('Q')]
    
    # 1. Reverse-item coherence: Do reverse items anti-correlate with forward items?
    reverse_coherence_by_trait = {}
    
    for trait in ['E', 'N', 'A', 'C', 'O']:
        forward_items = []
        reverse_items = []
        
        for col in test_cols:
            q_id = int(col.replace('Q', ''))
            if metadata[q_id]['trait'] == trait:
                if metadata[q_id]['reverse']:
                    reverse_items.append(col)
                else:
                    forward_items.append(col)
        
        if forward_items and reverse_items:
            # Calculate mean of forward vs reverse items per person
            forward_means = synthetic_df[forward_items].mean(axis=1)
            reverse_means = synthetic_df[reverse_items].mean(axis=1)
            
            # They should be negatively correlated
            corr = np.corrcoef(forward_means, reverse_means)[0, 1]
            reverse_coherence_by_trait[trait] = {
                'correlation': corr,
                'coherent': corr < -0.3  # Threshold for "good" reverse-item behavior
            }
        else:
            reverse_coherence_by_trait[trait] = {'correlation': np.nan, 'coherent': False}
    
    # Overall reverse coherence
    coherent_count = sum(1 for v in reverse_coherence_by_trait.values() if v['coherent'])
    reverse_coherence = coherent_count / len(reverse_coherence_by_trait)
    
    # 2. Ordinal correlations (Spearman, Kendall) between items within same trait
    spearman_by_trait = {}
    kendall_by_trait = {}
    
    for trait in ['E', 'N', 'A', 'C', 'O']:
        trait_items = [col for col in test_cols 
                      if metadata[int(col.replace('Q', ''))]['trait'] == trait]
        
        if len(trait_items) >= 2:
            # Average pairwise correlation
            correlations_sp = []
            correlations_ke = []
            
            for i in range(len(trait_items)):
                for j in range(i+1, len(trait_items)):
                    sp, _ = stats.spearmanr(synthetic_df[trait_items[i]], 
                                           synthetic_df[trait_items[j]])
                    ke, _ = stats.kendalltau(synthetic_df[trait_items[i]], 
                                            synthetic_df[trait_items[j]])
                    if not np.isnan(sp):
                        correlations_sp.append(sp)
                    if not np.isnan(ke):
                        correlations_ke.append(ke)
            
            spearman_by_trait[trait] = np.mean(correlations_sp) if correlations_sp else np.nan
            kendall_by_trait[trait] = np.mean(correlations_ke) if correlations_ke else np.nan
        else:
            spearman_by_trait[trait] = np.nan
            kendall_by_trait[trait] = np.nan
    
    # 3. Response variance and entropy (measures non-determinism)
    all_responses = synthetic_df[test_cols].values.flatten()
    response_variance = np.var(all_responses)
    
    # Entropy of response distribution
    value_counts = pd.Series(all_responses).value_counts(normalize=True)
    response_entropy = stats.entropy(value_counts.values)
    
    return {
        'reverse_coherence': reverse_coherence,
        'reverse_coherence_by_trait': reverse_coherence_by_trait,
        'spearman_correlation': np.nanmean(list(spearman_by_trait.values())),
        'spearman_by_trait': spearman_by_trait,
        'kendall_correlation': np.nanmean(list(kendall_by_trait.values())),
        'kendall_by_trait': kendall_by_trait,
        'response_variance': response_variance,
        'response_entropy': response_entropy
    }


def calculate_chi_square_test(synthetic_df, ground_truth_df):
    """
    Chi-square test on raw response frequency distributions.
    Tests per-trait (E, N, A, C, O) and overall.
    """
    trait_mapping = {
        'E': (1, 10), 'N': (11, 20), 'A': (21, 30), 
        'C': (31, 40), 'O': (41, 50)
    }
    
    chi_square_per_trait = {}
    
    # Per-trait tests
    for trait, (start, end) in trait_mapping.items():
        trait_qs = [f'Q{i}' for i in range(start, end + 1) if f'Q{i}' in synthetic_df.columns]
        trait_cols = [f'{trait}{i}' for i in range(1, 11) if f'{trait}{i}' in ground_truth_df.columns]
        
        if trait_qs and trait_cols:
            synth_responses = pd.concat([synthetic_df[q] for q in trait_qs])
            truth_responses = pd.concat([ground_truth_df[c] for c in trait_cols])
            
            synth_counts = synth_responses.value_counts().reindex(range(1, 6), fill_value=0).values
            truth_counts = truth_responses.value_counts().reindex(range(1, 6), fill_value=0).values
            
            if truth_counts.sum() > 0 and synth_counts.sum() > 0:
                expected = (truth_counts / truth_counts.sum()) * synth_counts.sum()
                valid = expected >= 5
                
                if valid.sum() >= 2:
                    stat, p = stats.chisquare(synth_counts[valid], f_exp=expected[valid])
                    chi_square_per_trait[trait] = {'statistic': stat, 'p_value': p, 'pass': p > 0.05}
                else:
                    chi_square_per_trait[trait] = {'statistic': np.nan, 'p_value': np.nan, 'pass': False}
            else:
                chi_square_per_trait[trait] = {'statistic': np.nan, 'p_value': np.nan, 'pass': False}
    
    # Overall test
    test_cols = [col for col in synthetic_df.columns if col.startswith('Q')]
    all_synth = pd.concat([synthetic_df[col] for col in test_cols])
    
    all_truth = []
    for col in test_cols:
        ground_col = map_question_to_trait(int(col.replace('Q', '')))
        if ground_col in ground_truth_df.columns:
            all_truth.append(ground_truth_df[ground_col])
    
    if all_truth:
        all_truth = pd.concat(all_truth)
        synth_counts = all_synth.value_counts().reindex(range(1, 6), fill_value=0).values
        truth_counts = all_truth.value_counts().reindex(range(1, 6), fill_value=0).values
        
        if truth_counts.sum() > 0 and synth_counts.sum() > 0:
            expected = (truth_counts / truth_counts.sum()) * synth_counts.sum()
            valid = expected >= 5
            
            if valid.sum() >= 2:
                stat, p = stats.chisquare(synth_counts[valid], f_exp=expected[valid])
                chi_square_overall = {'statistic': stat, 'p_value': p, 'pass': p > 0.05}
            else:
                chi_square_overall = {'statistic': np.nan, 'p_value': np.nan, 'pass': False}
        else:
            chi_square_overall = {'statistic': np.nan, 'p_value': np.nan, 'pass': False}
    else:
        chi_square_overall = {'statistic': np.nan, 'p_value': np.nan, 'pass': False}
    
    trait_pass_rate = sum(v['pass'] for v in chi_square_per_trait.values()) / len(chi_square_per_trait) if chi_square_per_trait else 0
    
    return {
        'chi_square_per_trait': chi_square_per_trait,
        'chi_square_overall': chi_square_overall,
        'trait_pass_rate': trait_pass_rate
    }


def calculate_all_metrics(synthetic_df, ground_truth_df):
    """Calculate all validation metrics."""
    # Persona consistency metrics
    persona_metrics = calculate_persona_consistency(synthetic_df)
    
    # Population similarity metrics
    mae_per_item, mae_overall = calculate_mae(synthetic_df, ground_truth_df)
    js_per_item, js_overall = calculate_js_divergence(synthetic_df, ground_truth_df)
    correlation, synth_means, truth_means = calculate_correlation(synthetic_df, ground_truth_df)
    ks_statistic, ks_p_value, ks_pass = calculate_ks_test(synthetic_df, ground_truth_df)
    chi_square = calculate_chi_square_test(synthetic_df, ground_truth_df)
    
    return {
        # Persona consistency
        'reverse_coherence': persona_metrics['reverse_coherence'],
        'reverse_coherence_by_trait': persona_metrics['reverse_coherence_by_trait'],
        'spearman_correlation': persona_metrics['spearman_correlation'],
        'kendall_correlation': persona_metrics['kendall_correlation'],
        'response_variance': persona_metrics['response_variance'],
        'response_entropy': persona_metrics['response_entropy'],
        
        # Population similarity
        'mae_per_item': mae_per_item,
        'mae_overall': mae_overall,
        'js_per_item': js_per_item,
        'js_overall': js_overall,
        'correlation': correlation,
        'synth_means': synth_means,
        'truth_means': truth_means,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'ks_pass': ks_pass,
        'chi_square_per_trait': chi_square['chi_square_per_trait'],
        'chi_square_overall': chi_square['chi_square_overall'],
        'chi_square_trait_pass_rate': chi_square['trait_pass_rate']
    }


def print_validation_results(metrics, strategy, num_synthetic, num_ground_truth):
    """Print formatted validation results."""
    print("="*80)
    print(f"VALIDATION RESULTS - {strategy.upper()} STRATEGY")
    print("="*80)
    print()
    
    print("PERSONA CONSISTENCY METRICS")
    print("-"*80)
    print(f"Reverse-Item Coherence:          {metrics['reverse_coherence']:.1%} ({int(metrics['reverse_coherence']*5)}/5 traits)")
    print(f"Spearman Correlation (avg):      {metrics['spearman_correlation']:.4f}")
    print(f"Kendall Correlation (avg):       {metrics['kendall_correlation']:.4f}")
    print(f"Response Variance:               {metrics['response_variance']:.4f}")
    print(f"Response Entropy:                {metrics['response_entropy']:.4f}")
    print()
    
    print("REVERSE-ITEM COHERENCE BY TRAIT")
    print("-"*80)
    trait_names = {'E': 'Extraversion', 'N': 'Neuroticism', 'A': 'Agreeableness', 
                   'C': 'Conscientiousness', 'O': 'Openness'}
    for trait in ['E', 'N', 'A', 'C', 'O']:
        if trait in metrics['reverse_coherence_by_trait']:
            r = metrics['reverse_coherence_by_trait'][trait]
            status = '✓ COHERENT' if r['coherent'] else '✗ INCOHERENT'
            corr_val = f"{r['correlation']:.3f}" if not np.isnan(r['correlation']) else "N/A"
            print(f"{trait} ({trait_names[trait]:17s}): r={corr_val:>7s} {status}")
    print()
    
    print("POPULATION SIMILARITY METRICS")
    print("-"*80)
    print(f"Mean Absolute Error (MAE):       {metrics['mae_overall']:.4f}")
    print(f"Jensen-Shannon Divergence:       {metrics['js_overall']:.4f}")
    print(f"Mean Correlation:                {metrics['correlation']:.4f}")
    print()
    
    print("STATISTICAL TESTS")
    print("-"*80)
    print(f"K-S Test (Overall):              D={metrics['ks_statistic']:.4f}, p={metrics['ks_p_value']:.4f} {'✓ PASS' if metrics['ks_pass'] else '✗ FAIL'}")
    print()
    print(f"Chi-Square Pass Rate (traits):   {metrics['chi_square_trait_pass_rate']:.1%} ({int(metrics['chi_square_trait_pass_rate']*5)}/5 traits)")
    print(f"Chi-Square Overall:              χ²={metrics['chi_square_overall']['statistic']:.2f}, p={metrics['chi_square_overall']['p_value']:.4f} {'✓ PASS' if metrics['chi_square_overall']['pass'] else '✗ FAIL'}")
    print()
    
    print("CHI-SQUARE BY TRAIT")
    print("-"*80)
    for trait in ['E', 'N', 'A', 'C', 'O']:
        if trait in metrics['chi_square_per_trait']:
            r = metrics['chi_square_per_trait'][trait]
            status = '✓ PASS' if r['pass'] else '✗ FAIL'
            print(f"{trait} ({trait_names[trait]:17s}): χ²={r['statistic']:.2f}, p={r['p_value']:.4f} {status}")
    print()
    
    print("DATA SUMMARY")
    print("-"*80)
    print(f"Synthetic responses:             {num_synthetic:,}")
    print(f"Ground truth responses:          {num_ground_truth:,}")
    print(f"Test items evaluated:            {len(metrics['mae_per_item'])}")
    print()
    print("="*80)
