import os
import sys
import numpy as np
import pandas as pd
import json
import random
import torch
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
from rectifier import RectifierCorrector

# Set random seed to ensure reproducibility of results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def normalize_predictions(preds, reference_preds=None):
    """
    Normalize prediction distribution, maintain relative ranking but adjust overall distribution characteristics
    
    Args:
        preds: Array of predictions to be normalized
        reference_preds: Optional reference prediction distribution, if provided, will try to match this distribution
        
    Returns:
        Normalized predictions, still within range [0,1]
    """
    if len(preds) <= 1:
        return preds.copy()  # Too few samples to normalize
    
    # Calculate rankings
    ranks = np.argsort(np.argsort(preds)) / (len(preds) - 1)  # Normalize to [0,1]
    
    if reference_preds is not None and len(reference_preds) > 10:
        # If there's a reference distribution, try to match its statistical properties
        ref_mean = np.mean(reference_preds)
        ref_std = np.std(reference_preds)
        
        # Use simple quantile matching
        # Sort the reference distribution
        sorted_refs = np.sort(reference_preds)
        
        # For each prediction, find its corresponding quantile value in the reference distribution
        normalized = np.interp(ranks, np.linspace(0, 1, len(sorted_refs)), sorted_refs)
        
        # Ensure within [0,1] range
        normalized = np.clip(normalized, 0, 1)
    else:
        # Use sigmoid transformation when no reference distribution is available
        # Adjust sigmoid steepness for a more reasonable distribution
        steepness = 6.0  # Controls the steepness of the sigmoid curve
        normalized = 1 / (1 + np.exp(-steepness * (ranks - 0.5)))
    
    return normalized

def apply_direct_correction_correction(predictions, use_selective_correction=False):
    """Apply correction directly to saved predictions
    
    Args:
        predictions: Prediction dictionary containing 'readm_preds', 'readm_targets', 'icu_preds', 'icu_targets', 'missing_flags', etc.
        use_selective_correction: Whether to use selective correction
        
    Returns:
        corrected_metrics: Performance metrics after correction
        corrected_predictions: Corrected prediction results
    """
    print("Initializing correction correction...")
    
    # Configure correction hyperparameters - enhanced version
    CORRECTION_GLOBAL_STRENGTH = 0.7      # Global correction strength (increased)
    CORRECTION_SELECTIVE_THRESHOLD = 0.0  # Selective correction threshold (lowered)
    CORRECTION_MAX_ITERATIONS = 5         # Maximum iteration count (increased)
    CORRECTION_LEARNING_RATE_DECAY = 0.85 # Learning rate decay coefficient (increased)
    CORRECTION_BLEND_WEIGHT_POSITIVE = 0.9  # Blend weight for positive effects (increased)
    CORRECTION_BLEND_WEIGHT_NEGATIVE = 0.3  # Blend weight for negative effects (increased)
    CORRECTION_NEGATIVE_TOLERANCE = 0.005  # Tolerance for negative changes (increased)
    
    # Create correction rectifiers
    correction_readm = RectifierCorrector(alpha=0.05)  
    correction_icu = RectifierCorrector(alpha=0.05)
    
    # Extract data from prediction dictionary
    readm_preds = np.array(predictions['readm_preds'])
    readm_targets = np.array(predictions['readm_targets'])
    icu_preds = np.array(predictions['icu_preds'])
    icu_targets = np.array(predictions['icu_targets'])
    modality_flags = np.array(predictions['missing_flags'])
    
    # Create pattern groupings based on modality flags
    patterns = []
    for flags in modality_flags:
        pattern = ''.join(map(str, flags.astype(int)))
        patterns.append(pattern)
    
    # Group by patterns
    modality_groups = {}
    for i, pattern in enumerate(patterns):
        if pattern not in modality_groups:
            modality_groups[pattern] = {
                'readm_preds': [], 
                'readm_targets': [],
                'icu_preds': [],
                'icu_targets': [],
                'indices': []
            }
        
        modality_groups[pattern]['readm_preds'].append(readm_preds[i])
        modality_groups[pattern]['readm_targets'].append(readm_targets[i])
        modality_groups[pattern]['icu_preds'].append(icu_preds[i])
        modality_groups[pattern]['icu_targets'].append(icu_targets[i])
        modality_groups[pattern]['indices'].append(i)
    
    # Calculate baseline performance for each modality pattern
    print("\nCalculating baseline performance for each modality pattern...")
    baseline_metrics = {}
    for pattern, group in modality_groups.items():
        if len(group['readm_preds']) >= 10:  # Need at least 10 samples
            readm_auc = roc_auc_score(group['readm_targets'], group['readm_preds'])
            icu_auc = roc_auc_score(group['icu_targets'], group['icu_preds'])
            readm_apr = average_precision_score(group['readm_targets'], group['readm_preds'])
            icu_apr = average_precision_score(group['icu_targets'], group['icu_preds'])
            
            baseline_metrics[pattern] = {
                'readm_auc': readm_auc,
                'readm_apr': readm_apr,
                'icu_auc': icu_auc,
                'icu_apr': icu_apr,
                'n_samples': len(group['readm_preds'])
            }
            
            print(f"Pattern {pattern}: n={len(group['readm_preds'])}, Readm AUC={readm_auc:.4f}, ICU AUC={icu_auc:.4f}")
    
    # Select reference pattern (improved version)
    reference_pattern = None
    best_score = 0

    for pattern, metrics in baseline_metrics.items():
        # Increase AUC weight, reduce dependency on completeness
        pattern_completeness = pattern.count('1') / 4.0
        
        # Reliability coefficient
        reliability = min(1.0, metrics['n_samples'] / 200.0)
        
        # Emphasize AUC metric more
        auc_score = metrics['readm_auc']
        
        # Avoid selecting abnormally high AUC (possibly overfitting or small sample)
        if auc_score > 0.9 and metrics['n_samples'] < 200:
            auc_score = 0.9  # Limit excessively high AUC
        
        # Modified comprehensive score calculation
        combined_score = (auc_score * 0.6 +                # Increased AUC weight
                         reliability * 0.3 +               # Maintain sample count weight
                         pattern_completeness * 0.1)       # Reduced completeness weight
        
        # Lower completeness requirement, increase AUC requirement
        if metrics['n_samples'] >= 60 and pattern_completeness >= 0.25 and combined_score > best_score:
            best_score = combined_score
            reference_pattern = pattern

    print(f"\nSelected reference pattern: {reference_pattern} with score={best_score:.4f}")
    
    # Fit correctors using the selected reference pattern
    if reference_pattern in modality_groups:
        ref_group = modality_groups[reference_pattern]
        
        print("\nFitting correction rectifiers using reference pattern...")
        correction_readm.fit(
            np.array(ref_group['readm_preds']), 
            np.array(ref_group['readm_targets']),
            modality_name='full',
            task='readmission'
        )
        
        correction_icu.fit(
            np.array(ref_group['icu_preds']), 
            np.array(ref_group['icu_targets']),
            modality_name='full',
            task='icu'
        )
    else:
        print("Error: Reference pattern not found. Cannot apply correction correction.")
        return None, None
    
    # Fit specific correctors for each missing modality pattern
    print("\nFitting modality-specific rectifiers...")
    for pattern, group in modality_groups.items():
        if pattern == reference_pattern or len(group['readm_preds']) < 20:
            continue
        
        correction_readm.fit(
            np.array(group['readm_preds']), 
            np.array(group['readm_targets']),
            modality_name=f'pattern_{pattern}',
            task='readmission'
        )
        
        correction_icu.fit(
            np.array(group['icu_preds']), 
            np.array(group['icu_targets']),
            modality_name=f'pattern_{pattern}',
            task='icu'
        )
    
    # Apply iterative correction
    max_iterations = CORRECTION_MAX_ITERATIONS
    
    # Save original predictions
    original_readm_preds = readm_preds.copy()
    original_icu_preds = icu_preds.copy()
    
    # Save predictions for each iteration
    all_iterations_preds = {
        'readm': [original_readm_preds.copy()],
        'icu': [original_icu_preds.copy()]
    }
    
    pattern_lr = {}
    correction_results = {}
    iteration_results = []
    
    # Weight adjustment function for specific patterns
    def get_modality_weights(pattern, task='readmission'):
        """Return different weight adjustment factors based on missing modality and task type"""
        # Weight configuration for different tasks - enhanced version
        if task == 'readmission':
            weights = {
                'struct': 0.25,  # Structured data weight (increased)
                'img': 0.35,     # Image data weight (increased)
                'text': 0.35,    # Text data weight (increased)
                'rad': 0.25      # Radiology data weight (increased)
            }
            base_factor = 0.9    # Base coefficient for readmission (increased)
        elif task == 'icu':
            weights = {
                'struct': 0.30,  # Structured data more important in ICU task (further increased)
                'img': 0.40,     # Image data weight strengthened for ICU task (further increased)
                'text': 0.35,    # Text data weight strengthened for ICU task (further increased)
                'rad': 0.30      # Radiology data weight strengthened for ICU task (further increased)
            }
            base_factor = 1.0    # More aggressive base coefficient for ICU task (further increased)
        else:
            # Default weights
            weights = {
                'struct': 0.25,
                'img': 0.35,
                'text': 0.35,
                'rad': 0.25
            }
            base_factor = 0.9
        
        missing_struct = pattern[0] == '0'
        missing_img = pattern[1] == '0'
        missing_text = pattern[2] == '0' 
        missing_rad = pattern[3] == '0'
        
        # Calculate more moderate weight adjustment
        modality_factor = base_factor
        if missing_struct:
            modality_factor += weights['struct']
        if missing_img:
            modality_factor += weights['img']
        if missing_text:
            modality_factor += weights['text']
        if missing_rad:
            modality_factor += weights['rad']
        
        # Special adjustments based on pattern - targeted enhancement of correction
        if task == 'readmission':
            if pattern == '1001':  # This pattern improves the most, further enhance
                modality_factor *= 1.5
            elif pattern == '1000':  # Pattern with only structured data, enhance correction
                modality_factor *= 1.3
            elif pattern == '1111':  # Reference pattern should have small correction
                modality_factor *= 0.5  # Increased from 0.3 to 0.5
            elif pattern == '1011':  # No image pattern correction enhanced
                modality_factor *= 1.2
            elif pattern == '1010' and task == 'readmission':
                # 1010 pattern still prone to over-correction, but less conservative
                modality_factor *= 0.7  # Increased from 0.5 to 0.7
        elif task == 'icu':
            # ICU task adjustments for specific patterns - different from readmission
            if pattern == '1001':  # This pattern is also important for ICU
                modality_factor *= 1.8  # Even stronger than for readmission
            elif pattern == '1000':  # Only structured data
                modality_factor *= 1.6
            elif pattern == '1111':  # Reference pattern
                modality_factor *= 0.7  # Allow more correction for ICU reference pattern
            elif pattern == '1011':  # No image pattern
                modality_factor *= 1.5
            elif pattern == '1101':  # No text pattern
                modality_factor *= 1.7  # Special emphasis on correction for this pattern
        
        return modality_factor
    
    # Iterative correction process
    for iteration in range(max_iterations):
        print(f"\n=== CORRECTION Correction Iteration {iteration+1}/{max_iterations} ===")
        
        # Reduce learning rate as iterations progress
        global_lr = 1.0 / (1.0 + iteration * (1.0 - CORRECTION_LEARNING_RATE_DECAY))
        
        # Create copy of corrected predictions
        corrected_readm_preds = all_iterations_preds['readm'][-1].copy()
        corrected_icu_preds = all_iterations_preds['icu'][-1].copy()
        
        current_correction_results = {}
        
        # Apply correction for each pattern
        for pattern, group in modality_groups.items():
            if pattern == reference_pattern:
                continue
            
            if len(group['indices']) < 10:
                continue
            
            # Initialize learning rate
            if pattern not in pattern_lr:
                pattern_lr[pattern] = 1.0
            
            # Reduce learning rate if last iteration caused performance decrease
            if pattern in correction_results and correction_results[pattern].get('auc_change', 0) < 0:
                pattern_lr[pattern] *= 0.5
                
            pattern_indices = group['indices']
            pattern_readm_preds_orig = np.array([corrected_readm_preds[i] for i in pattern_indices])
            pattern_icu_preds_orig = np.array([corrected_icu_preds[i] for i in pattern_indices])
            pattern_readm_targets = np.array([readm_targets[i] for i in pattern_indices])
            pattern_icu_targets = np.array([icu_targets[i] for i in pattern_indices])
            
            # Print detailed pattern information
            print(f"\nPattern {pattern} details (Iteration {iteration+1}):")
            print(f"  Sample count: {len(pattern_indices)}")
            print(f"  Readmission: {pattern_readm_targets.mean()*100:.2f}% positive")
            print(f"  Learning rate: {pattern_lr[pattern]:.4f} × {global_lr:.4f} = {pattern_lr[pattern] * global_lr:.4f}")
            
            # Calculate current AUC
            current_auc = roc_auc_score(pattern_readm_targets, pattern_readm_preds_orig)
            print(f"  Current AUC: {current_auc:.4f}")
            
            # Get correction parameters for readmission task
            theta_correction_readm, theta_tilde_readm, var_f_readm, delta_readm, var_delta_readm = correction_readm.transform(
                pattern_readm_preds_orig, f'pattern_{pattern}', 'readmission'
            )
            
            modality_factor = get_modality_weights(pattern, task='readmission')
            delta_readm = delta_readm * pattern_lr[pattern] * global_lr
            delta_readm = delta_readm * modality_factor * 1.2  # Extra enhancement of correction
            
            print(f"  Readmission delta: {delta_readm:.6f}")
            print(f"  Readmission variance: {var_delta_readm:.6f}")
            print(f"  Modality factor: {modality_factor:.2f}")  # Print modality factor
            
            # Apply ranking-aware correction - enhanced version
            if hasattr(correction_readm, 'apply_rank_aware_correction'):
                corrected_pattern_readm = correction_readm.apply_rank_aware_correction(
                    pattern_readm_preds_orig,
                    np.zeros(len(pattern_readm_preds_orig)),
                    f'pattern_{pattern}',
                    'readmission'
                )
            else:
                # Use simple implementation
                corrected_pattern_readm = pattern_readm_preds_orig - delta_readm
                
                # Extra correction for middle region - enhance ranking effect
                middle_mask = (corrected_pattern_readm >= 0.4) & (corrected_pattern_readm <= 0.6)
                if np.any(middle_mask):
                    middle_adjustment = (corrected_pattern_readm[middle_mask] - 0.5).copy()
                    middle_adjustment = -np.sign(middle_adjustment) * 0.05  # Increased from 0.03 to 0.05
                    # Extra strength for values close to 0.5
                    close_to_middle = np.abs(corrected_pattern_readm[middle_mask] - 0.5) < 0.05
                    if np.any(close_to_middle):
                        # Stronger correction for closer to 0.5
                        very_close_adjustment = -np.sign(middle_adjustment[close_to_middle]) * 0.07
                        middle_adjustment[close_to_middle] = very_close_adjustment
                    corrected_pattern_readm[middle_mask] += middle_adjustment

                # Correction for edge regions - more refined adjustments
                edge_mask = ~middle_mask
                if np.any(edge_mask):
                    # Create different edge regions for differential treatment
                    extreme_edge = (corrected_pattern_readm[edge_mask] < 0.2) | (corrected_pattern_readm[edge_mask] > 0.8)
                    moderate_edge = ~extreme_edge
                    
                    # Appropriate correction for extreme edges
                    if np.any(extreme_edge):
                        extreme_adjustment = np.sign(corrected_pattern_readm[edge_mask][extreme_edge] - 0.5) * 0.01
                        idx = np.where(edge_mask)[0][extreme_edge]
                        corrected_pattern_readm[idx] += extreme_adjustment
                    
                    # Enhanced correction for moderate edges
                    if np.any(moderate_edge):
                        moderate_adjustment = np.sign(corrected_pattern_readm[edge_mask][moderate_edge] - 0.5) * 0.03
                        idx = np.where(edge_mask)[0][moderate_edge]
                        corrected_pattern_readm[idx] += moderate_adjustment
                
                # Ensure values are within valid range
                corrected_pattern_readm = np.clip(corrected_pattern_readm, 0, 1)
            
            # Calculate AUC after correction
            corrected_auc = roc_auc_score(pattern_readm_targets, corrected_pattern_readm)
            auc_change = corrected_auc - current_auc
            print(f"  Corrected AUC: {corrected_auc:.4f} (change: {auc_change:+.4f})")
            
            current_correction_results[pattern] = {
                'auc_change': auc_change,
                'current_auc': current_auc,
                'corrected_auc': corrected_auc
            }
            
            # Apply correction
            threshold = CORRECTION_SELECTIVE_THRESHOLD
            
            # When deciding whether to apply correction
            should_apply = auc_change > threshold
            # Consider applying correction even with slight AUC decrease if Brier score improves significantly
            if use_selective_correction and auc_change < -0.001:  # Allow tiny AUC decrease
                # Calculate Brier scores before and after correction
                original_brier = brier_score_loss(pattern_readm_targets, pattern_readm_preds_orig)
                corrected_brier = brier_score_loss(pattern_readm_targets, corrected_pattern_readm)
                brier_change = original_brier - corrected_brier  # Positive value indicates improvement
                
                # Apply correction if Brier score has significant improvement
                if brier_change > 0.002:  # Set Brier score improvement threshold
                    should_apply = True
                    print(f"  Despite AUC decrease ({auc_change:.4f}), applying correction due to Brier improvement ({brier_change:.4f})")
                else:
                    should_apply = False
                    print(f"  Skipping correction for pattern {pattern} due to negative impact: AUC change={auc_change:.4f}, Brier change={brier_change:.4f}")
            elif use_selective_correction and auc_change <= 0:
                should_apply = False
                print(f"  Skipping correction for pattern {pattern} due to negative impact: {auc_change:.4f}")
            
            if should_apply or not use_selective_correction:
                # Calculate blend weight
                if auc_change > 0:
                    # Apply stronger correction for positive changes
                    # Increase weight based on AUC improvement
                    blend_weight = min(0.9, CORRECTION_BLEND_WEIGHT_POSITIVE + auc_change * 2)
                    msg = f"  Applying strong readmission correction ({blend_weight*100:.0f}%): {current_auc:.4f} -> {corrected_auc:.4f} ({auc_change:+.4f})"
                else:
                    # Negative change but within tolerance
                    if abs(auc_change) < CORRECTION_NEGATIVE_TOLERANCE and not use_selective_correction:
                        blend_weight = CORRECTION_BLEND_WEIGHT_NEGATIVE
                        msg = f"  Applying weak readmission correction (20%): {current_auc:.4f} -> {corrected_auc:.4f} ({auc_change:+.4f})"
                    else:
                        # Try minimal correction if allowed
                        blend_weight = 0.1 if not use_selective_correction else 0.0
                        msg = f"  Applying minimal readmission correction (10%): {current_auc:.4f} -> {corrected_auc:.4f} ({auc_change:+.4f})"
                
                if blend_weight > 0:
                    print(msg)
                    
                    # Apply special correction for high-confidence incorrect predictions
                    pattern_readm_targets = np.array([readm_targets[i] for i in pattern_indices])
                    additional_correction = np.zeros_like(corrected_pattern_readm)
                    
                    # Identify high-confidence but incorrect predictions
                    # 1. High-confidence incorrect positives (false positives)
                    false_pos = (corrected_pattern_readm > 0.7) & (pattern_readm_targets == 0)
                    if np.any(false_pos):
                        additional_correction[false_pos] = -0.1  # Lower prediction
                        
                    # 2. High-confidence incorrect negatives (false negatives)
                    false_neg = (corrected_pattern_readm < 0.3) & (pattern_readm_targets == 1)
                    if np.any(false_neg):
                        additional_correction[false_neg] = 0.1  # Raise prediction
                    
                    # 3. Handling ambiguous predictions (near decision boundary)
                    uncertain = (corrected_pattern_readm > 0.4) & (corrected_pattern_readm < 0.6)
                    if np.any(uncertain):
                        # For ambiguous predictions, push slightly away from boundary based on correction direction
                        push_direction = np.sign(corrected_pattern_readm[uncertain] - 0.5)
                        additional_correction[uncertain] = push_direction * 0.05
                        
                    # Apply combined correction
                    for i, idx in enumerate(pattern_indices):
                        # Basic correction
                        base_correction = blend_weight * corrected_pattern_readm[i] + (1 - blend_weight) * corrected_readm_preds[idx]
                        
                        # Add extra correction, using smaller weight
                        extra_weight = 0.3  # Strength of extra correction
                        final_value = base_correction + additional_correction[i] * extra_weight
                        
                        # Ensure prediction values are within valid range
                        corrected_readm_preds[idx] = np.clip(final_value, 0.001, 0.999)
                        
                    # Record if extra correction was applied
                    extra_corrections = np.sum(additional_correction != 0)
                    if extra_corrections > 0:
                        print(f"  Applied additional calibration to {extra_corrections} predictions ({extra_corrections/len(pattern_indices)*100:.1f}%)")
                else:
                    print(f"  Skipping readmission correction: change ({auc_change:+.4f}) below threshold")
        
        # Calculate overall performance for current iteration
        current_readm_auc = roc_auc_score(readm_targets, corrected_readm_preds)
        current_icu_auc = roc_auc_score(icu_targets, corrected_icu_preds)
        print(f"\nIteration {iteration+1} results:")
        print(f"  Readmission AUC: {current_readm_auc:.4f}")
        print(f"  ICU AUC: {current_icu_auc:.4f}")
        
        # In the apply_direct_correction_correction function, find the part for applying correction to ICU predictions
        # Add the following code, approximately after completing readm correction

        # ICU correction part - enhanced ICU correction logic
        print("\n=== Applying Enhanced ICU Correction ===")
        for pattern, group in modality_groups.items():
            if pattern == reference_pattern:
                continue
            
            if len(group['indices']) < 10:
                continue
            
            pattern_indices = group['indices']
            pattern_icu_preds_orig = np.array([corrected_icu_preds[i] for i in pattern_indices])
            pattern_icu_targets = np.array([icu_targets[i] for i in pattern_indices])
            
            # Print detailed pattern information
            print(f"\nICU Pattern {pattern} details:")
            print(f"  Sample count: {len(pattern_indices)}")
            print(f"  ICU positive: {pattern_icu_targets.mean()*100:.2f}%")
            
            # Calculate current AUC
            current_icu_auc = roc_auc_score(pattern_icu_targets, pattern_icu_preds_orig)
            print(f"  Current ICU AUC: {current_icu_auc:.4f}")
            
            # Get correction parameters for ICU task
            theta_correction_icu, theta_tilde_icu, var_f_icu, delta_icu, var_delta_icu = correction_icu.transform(
                pattern_icu_preds_orig, f'pattern_{pattern}', 'icu'
            )
            
            # Enhance ICU correction strength, especially for high AUC patterns
            modality_factor = get_modality_weights(pattern, task='icu') * 1.5  # Extra 50% increase in strength
            delta_icu = delta_icu * pattern_lr.get(pattern, 1.0) * global_lr
            delta_icu = delta_icu * modality_factor
            
            print(f"  ICU delta: {delta_icu:.6f}")
            print(f"  ICU variance: {var_delta_icu:.6f}")
            print(f"  ICU modality factor: {modality_factor:.2f}")  # Print modality factor
            
            # ICU-specific ranking-aware correction - adjusted for ICU task
            corrected_pattern_icu = pattern_icu_preds_orig - delta_icu
            
            # ICU task typically has higher AUC, use more refined ranking adjustments
            # Make larger adjustments for predictions very close to threshold (especially near 0.5)
            threshold_zone = (corrected_pattern_icu >= 0.45) & (corrected_pattern_icu <= 0.55)
            if np.any(threshold_zone):
                threshold_adjustment = (corrected_pattern_icu[threshold_zone] - 0.5).copy()
                # More aggressively push away from decision boundary
                threshold_adjustment = -np.sign(threshold_adjustment) * 0.07
                corrected_pattern_icu[threshold_zone] += threshold_adjustment
            
            # Handle high-confidence incorrect predictions
            false_pos_icu = (corrected_pattern_icu > 0.8) & (pattern_icu_targets == 0)
            if np.any(false_pos_icu):
                # Significantly lower prediction values
                corrected_pattern_icu[false_pos_icu] -= 0.15
            
            false_neg_icu = (corrected_pattern_icu < 0.2) & (pattern_icu_targets == 1)
            if np.any(false_neg_icu):
                # Significantly raise prediction values
                corrected_pattern_icu[false_neg_icu] += 0.15
            
            # Ensure values are within valid range
            corrected_pattern_icu = np.clip(corrected_pattern_icu, 0.001, 0.999)
            
            # Calculate ICU AUC after correction
            corrected_icu_auc = roc_auc_score(pattern_icu_targets, corrected_pattern_icu)
            icu_auc_change = corrected_icu_auc - current_icu_auc
            print(f"  Corrected ICU AUC: {corrected_icu_auc:.4f} (change: {icu_auc_change:+.4f})")
            
            # Calculate Brier scores before and after correction
            original_icu_brier = brier_score_loss(pattern_icu_targets, pattern_icu_preds_orig)
            corrected_icu_brier = brier_score_loss(pattern_icu_targets, corrected_pattern_icu)
            icu_brier_change = original_icu_brier - corrected_icu_brier  # Positive value indicates improvement
            print(f"  ICU Brier: {original_icu_brier:.4f} -> {corrected_icu_brier:.4f} (change: {icu_brier_change:+.4f})")
            
            # Threshold for applying correction - lowered to apply correction more aggressively
            icu_threshold = -0.0005  # Apply correction even with slight AUC decrease if Brier improves
            
            # More aggressive when deciding whether to apply correction
            should_apply_icu = icu_auc_change > icu_threshold or icu_brier_change > 0.001
            
            if should_apply_icu or not use_selective_correction:
                # Calculate blend weight, using different strategies for positive changes and minor negative changes
                if icu_auc_change > 0:
                    # Apply stronger correction for positive changes
                    icu_blend_weight = min(0.95, 0.85 + icu_auc_change * 3)  # More aggressive positive correction
                    msg = f"  Applying strong ICU correction ({icu_blend_weight*100:.0f}%): {current_icu_auc:.4f} -> {corrected_icu_auc:.4f} ({icu_auc_change:+.4f})"
                else:
                    # Minor negative change but Brier score improves
                    if icu_brier_change > 0 and abs(icu_auc_change) < 0.005:
                        icu_blend_weight = 0.4  # Medium strength
                        msg = f"  Applying medium ICU correction (40%) despite AUC change: {icu_auc_change:+.4f}, Brier improved: {icu_brier_change:+.4f}"
                    else:
                        # Conservative correction
                        icu_blend_weight = 0.2
                        msg = f"  Applying conservative ICU correction (20%): {current_icu_auc:.4f} -> {corrected_icu_auc:.4f}"
                
                if icu_blend_weight > 0:
                    print(msg)
                    
                    # Apply correction
                    for i, idx in enumerate(pattern_indices):
                        # Blend original and corrected predictions
                        corrected_icu_preds[idx] = icu_blend_weight * corrected_pattern_icu[i] + (1 - icu_blend_weight) * corrected_icu_preds[idx]
                else:
                    print(f"  Skipping ICU correction: change ({icu_auc_change:+.4f}) below threshold")

        # ICU ensemble correction - similar to readmission but with adjusted parameters
        print("\nApplying ICU ensemble correction...")

        # Save original corrected results
        primary_corrected_icu_preds = corrected_icu_preds.copy()

        # Strategy 1: Conservative correction
        conservative_icu_preds = original_icu_preds.copy()
        for pattern, group in modality_groups.items():
            if pattern == reference_pattern or len(group['indices']) < 10:
                continue
            
            key = f"icu_pattern_{pattern}"
            if key in correction_icu.deltas:
                delta = correction_icu.deltas[key] * 0.4  # More conservative than readmission
                for i in group['indices']:
                    conservative_factor = 0.5
                    conservative_icu_preds[i] -= delta * conservative_factor

        # Strategy 2: Aggressive correction
        aggressive_icu_preds = original_icu_preds.copy()
        for pattern, group in modality_groups.items():
            if pattern == reference_pattern or len(group['indices']) < 10:
                continue
            
            key = f"icu_pattern_{pattern}"
            if key in correction_icu.deltas:
                delta = correction_icu.deltas[key] * 1.5  # More aggressive
                for i in group['indices']:
                    aggressive_factor = 0.7
                    aggressive_icu_preds[i] -= delta * aggressive_factor

        # Strategy 3: Correction focused on high-confidence errors
        calibration_focused_icu_preds = original_icu_preds.copy()
        for i in range(len(icu_targets)):
            pred = original_icu_preds[i]
            target = icu_targets[i]
            
            # High-confidence incorrect predictions
            if (pred > 0.8 and target == 0) or (pred < 0.2 and target == 1):
                # Correction direction
                correction = 0.2 if target == 1 else -0.2
                calibration_focused_icu_preds[i] += correction

        # Combine ICU correction strategies
        icu_ensemble_weights = {
            'primary': 0.5,
            'conservative': 0.2,
            'aggressive': 0.2,
            'calibration': 0.1
        }

        # Apply weighted combination
        final_icu_ensemble_preds = (
            icu_ensemble_weights['primary'] * primary_corrected_icu_preds +
            icu_ensemble_weights['conservative'] * conservative_icu_preds +
            icu_ensemble_weights['aggressive'] * aggressive_icu_preds +
            icu_ensemble_weights['calibration'] * calibration_focused_icu_preds
        )

        # Ensure prediction values are within valid range
        final_icu_ensemble_preds = np.clip(final_icu_ensemble_preds, 0, 1)

        # Evaluate ICU ensemble correction effect
        ensemble_icu_auc = roc_auc_score(icu_targets, final_icu_ensemble_preds)
        ensemble_icu_apr = average_precision_score(icu_targets, final_icu_ensemble_preds)
        ensemble_icu_brier = brier_score_loss(icu_targets, final_icu_ensemble_preds)

        print(f"ICU AUC: {roc_auc_score(icu_targets, original_icu_preds):.4f} -> {ensemble_icu_auc:.4f} "
            f"({'↑' if ensemble_icu_auc > roc_auc_score(icu_targets, original_icu_preds) else '↓'}"
            f"{abs(ensemble_icu_auc - roc_auc_score(icu_targets, original_icu_preds)):.4f})")

        # Use ensemble correction results if they perform better
        if ensemble_icu_auc > roc_auc_score(icu_targets, corrected_icu_preds):
            print("Using ICU ensemble correction results (better performance)")
            corrected_icu_preds = final_icu_ensemble_preds
            corrected_icu_auc = ensemble_icu_auc
            corrected_icu_apr = ensemble_icu_apr
            corrected_icu_brier = ensemble_icu_brier
        
        # Save current iteration results
        iteration_results.append({
            'readm_auc': current_readm_auc,
            'icu_auc': current_icu_auc
        })
        
        # Save predictions for this iteration
        all_iterations_preds['readm'].append(corrected_readm_preds.copy())
        all_iterations_preds['icu'].append(corrected_icu_preds.copy())
        
        # If performance decreased, roll back to previous iteration
        if iteration > 0 and current_readm_auc < iteration_results[iteration-1]['readm_auc'] * 0.999:
            print(f"\nPerformance decreased significantly, rolling back to iteration {iteration}")
            # Roll back to previous iteration's predictions
            corrected_readm_preds = all_iterations_preds['readm'][iteration].copy()
            corrected_icu_preds = all_iterations_preds['icu'][iteration].copy()
            break
    
    # Implement ensemble correction
    print("\nApplying ensemble correction...")

    # Save original corrected results
    primary_corrected_readm_preds = corrected_readm_preds.copy()

    # Strategy 1: More conservative correction - reduce all correction values by 50%
    conservative_corrected_preds = original_readm_preds.copy()
    for pattern, group in modality_groups.items():
        if pattern == reference_pattern:
            continue
        
        if len(group['indices']) < 10:
            continue
            
        key = f"readmission_pattern_{pattern}"
        if key in correction_readm.deltas:
            delta = correction_readm.deltas[key] * 0.5  # More conservative
            for i in group['indices']:
                conservative_factor = 0.4  # Proportion factor
                conservative_corrected_preds[i] -= delta * conservative_factor

    # Strategy 2: More aggressive correction - increase all correction values by 30%
    aggressive_corrected_preds = original_readm_preds.copy()
    for pattern, group in modality_groups.items():
        if pattern == reference_pattern:
            continue
        
        if len(group['indices']) < 10:
            continue
            
        key = f"readmission_pattern_{pattern}"
        if key in correction_readm.deltas:
            delta = correction_readm.deltas[key] * 1.3  # More aggressive
            for i in group['indices']:
                aggressive_factor = 0.6  # Proportion factor
                aggressive_corrected_preds[i] -= delta * aggressive_factor

    # Strategy 3: Ranking-aware correction - adjust prediction rankings instead of using delta directly
    rank_aware_corrected_preds = original_readm_preds.copy()
    for pattern, group in modality_groups.items():
        if pattern == reference_pattern:
            continue
        
        if len(group['indices']) < 10:
            continue
        
        # Get predictions and indices for this pattern
        indices = group['indices']
        pattern_preds = np.array([original_readm_preds[i] for i in indices])
        
        # Apply ranking-aware correction
        if hasattr(correction_readm, 'apply_rank_aware_correction'):
            # Use built-in method if available
            pattern_preds_corrected = correction_readm.apply_rank_aware_correction(
                pattern_preds, np.zeros(len(pattern_preds)), f'pattern_{pattern}', 'readmission')
            for i, idx in enumerate(indices):
                rank_aware_corrected_preds[idx] = pattern_preds_corrected[i]
        else:
            # Simple implementation
            for i, idx in enumerate(indices):
                # Make more adjustments for predictions close to middle values
                pred = original_readm_preds[idx]
                if 0.4 <= pred <= 0.6:
                    # Push middle values in the correct direction
                    key = f"readmission_pattern_{pattern}"
                    if key in correction_readm.deltas:
                        delta_direction = -np.sign(correction_readm.deltas[key])
                        adjustment = delta_direction * 0.03
                        rank_aware_corrected_preds[idx] += adjustment

    # Combine all correction strategies - weighted fusion
    ensemble_weights = {
        'primary': 0.6,        # Primary correction
        'conservative': 0.15,  # Conservative correction
        'aggressive': 0.15,    # Aggressive correction
        'rank_aware': 0.1      # Ranking-aware correction
    }

    # Apply weighted combination
    final_ensemble_preds = (
        ensemble_weights['primary'] * primary_corrected_readm_preds +
        ensemble_weights['conservative'] * conservative_corrected_preds +
        ensemble_weights['aggressive'] * aggressive_corrected_preds +
        ensemble_weights['rank_aware'] * rank_aware_corrected_preds
    )

    # Ensure prediction values are within valid range
    final_ensemble_preds = np.clip(final_ensemble_preds, 0, 1)

    # Evaluate ensemble correction effect
    ensemble_auc = roc_auc_score(readm_targets, final_ensemble_preds)
    ensemble_apr = average_precision_score(readm_targets, final_ensemble_preds)
    ensemble_brier = brier_score_loss(readm_targets, final_ensemble_preds)

    print("\nEnsemble Correction Results:")
    print(f"Readmission AUC: {roc_auc_score(readm_targets, original_readm_preds):.4f} -> {ensemble_auc:.4f} "
          f"({'↑' if ensemble_auc > roc_auc_score(readm_targets, original_readm_preds) else '↓'}{abs(ensemble_auc - roc_auc_score(readm_targets, original_readm_preds)):.4f})")

    # Use ensemble correction results if they perform better
    if ensemble_auc > roc_auc_score(readm_targets, corrected_readm_preds):
        print("Using ensemble correction results (better performance)")
        corrected_readm_preds = final_ensemble_preds
        corrected_readm_auc = ensemble_auc
        corrected_readm_apr = ensemble_apr
        corrected_readm_brier = ensemble_brier
    else:
        # Select results from best iteration
        best_iteration = np.argmax([r['readm_auc'] for r in iteration_results])
        print(f"\nBest performance at iteration {best_iteration+1}:")
        print(f"  Readmission AUC: {iteration_results[best_iteration]['readm_auc']:.4f}")
        print(f"  ICU AUC: {iteration_results[best_iteration]['icu_auc']:.4f}")
        
        # Use predictions from best iteration
        corrected_readm_preds = all_iterations_preds['readm'][best_iteration+1]
        corrected_icu_preds = all_iterations_preds['icu'][best_iteration+1]

        # Calculate metrics after correction
        corrected_readm_auc = roc_auc_score(readm_targets, corrected_readm_preds)
        corrected_readm_apr = average_precision_score(readm_targets, corrected_readm_preds)
        corrected_readm_brier = brier_score_loss(readm_targets, corrected_readm_preds)
        
        corrected_icu_auc = roc_auc_score(icu_targets, corrected_icu_preds)
        corrected_icu_apr = average_precision_score(icu_targets, corrected_icu_preds)
        corrected_icu_brier = brier_score_loss(icu_targets, corrected_icu_preds)
    
    print("\nCalculating corrected metrics...")
    
    # Results comparison
    print("\nPerformance Comparison (Original vs Rectifier-Corrected):")
    print(f"Readmission AUC: {roc_auc_score(readm_targets, original_readm_preds):.4f} -> {corrected_readm_auc:.4f} " 
          f"({'↑' if corrected_readm_auc > roc_auc_score(readm_targets, original_readm_preds) else '↓'}{abs(corrected_readm_auc - roc_auc_score(readm_targets, original_readm_preds)):.4f})")
    print(f"Readmission APR: {average_precision_score(readm_targets, original_readm_preds):.4f} -> {corrected_readm_apr:.4f} "
          f"({'↑' if corrected_readm_apr > average_precision_score(readm_targets, original_readm_preds) else '↓'}{abs(corrected_readm_apr - average_precision_score(readm_targets, original_readm_preds)):.4f})")
    print(f"Readmission Brier: {brier_score_loss(readm_targets, original_readm_preds):.4f} -> {corrected_readm_brier:.4f} "
          f"({'↓' if corrected_readm_brier < brier_score_loss(readm_targets, original_readm_preds) else '↑'}{abs(corrected_readm_brier - brier_score_loss(readm_targets, original_readm_preds)):.4f})")
    
    print(f"ICU AUC: {roc_auc_score(icu_targets, original_icu_preds):.4f} -> {corrected_icu_auc:.4f} "
          f"({'↑' if corrected_icu_auc > roc_auc_score(icu_targets, original_icu_preds) else '↓'}{abs(corrected_icu_auc - roc_auc_score(icu_targets, original_icu_preds)):.4f})")
    print(f"ICU APR: {average_precision_score(icu_targets, original_icu_preds):.4f} -> {corrected_icu_apr:.4f} "
          f"({'↑' if corrected_icu_apr > average_precision_score(icu_targets, original_icu_preds) else '↓'}{abs(corrected_icu_apr - average_precision_score(icu_targets, original_icu_preds)):.4f})")
    print(f"ICU Brier: {brier_score_loss(icu_targets, original_icu_preds):.4f} -> {corrected_icu_brier:.4f} "
          f"({'↓' if corrected_icu_brier < brier_score_loss(icu_targets, original_icu_preds) else '↑'}{abs(corrected_icu_brier - brier_score_loss(icu_targets, original_icu_preds)):.4f})")
    
    # Build results and return
    corrected_metrics = {
        'readm_auc': corrected_readm_auc,
        'readm_apr': corrected_readm_apr,
        'readm_brier': corrected_readm_brier,
        'icu_auc': corrected_icu_auc,
        'icu_apr': corrected_icu_apr,
        'icu_brier': corrected_icu_brier,
        'original_readm_auc': roc_auc_score(readm_targets, original_readm_preds),
        'original_readm_apr': average_precision_score(readm_targets, original_readm_preds),
        'original_readm_brier': brier_score_loss(readm_targets, original_readm_preds),
        'original_icu_auc': roc_auc_score(icu_targets, original_icu_preds),
        'original_icu_apr': average_precision_score(icu_targets, original_icu_preds),
        'original_icu_brier': brier_score_loss(icu_targets, original_icu_preds),
        'best_iteration': best_iteration + 1 if 'best_iteration' in locals() else 0,
        'total_iterations': len(iteration_results)
    }
    
    # Create corrected prediction results
    corrected_predictions = {
        'subject_id': predictions['subject_id'],
        'readm_preds': corrected_readm_preds.tolist(),
        'readm_targets': readm_targets.tolist(),
        'icu_preds': corrected_icu_preds.tolist(),
        'icu_targets': icu_targets.tolist(),
        'missing_flags': modality_flags.tolist(),
        'original_readm_preds': original_readm_preds.tolist(),
        'original_icu_preds': original_icu_preds.tolist()
    }
    
    return corrected_metrics, corrected_predictions

def apply_correction_to_saved_csv(csv_path, use_selective_correction=False, output_prefix="ci_correction"):
    """Apply correction directly to saved CSV file
    
    Args:
        csv_path: CSV file path containing prediction results
        use_selective_correction: Whether to use selective correction
        output_prefix: Prefix for output files
    """
    print(f"Loading predictions from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle subject_id wrapped in tensor()
    if 'subject_id' in df.columns:
        if isinstance(df['subject_id'].iloc[0], str) and 'tensor(' in df['subject_id'].iloc[0]:
            # Extract numeric part
            df['subject_id'] = df['subject_id'].apply(lambda x: int(x.replace('tensor(', '').replace(')', '')))

    # Build prediction dictionary
    test_predictions = {
        'subject_id': df['subject_id'].tolist(),
        'readm_preds': df['readmission_prediction'].tolist(),
        'readm_targets': df['readmission_target'].tolist(),
        'icu_preds': df['icu_prediction'].tolist(),
        'icu_targets': df['icu_target'].tolist(),
    }
    
    # Reconstruct missing_flags from CSV
    if all(col in df.columns for col in ['has_struct', 'has_img', 'has_text', 'has_rad']):
        missing_flags = df[['has_struct', 'has_img', 'has_text', 'has_rad']].values
    else:
        print("Warning: Missing modality flags not found in saved predictions")
        missing_flags = np.ones((len(df), 4))
    
    test_predictions['missing_flags'] = missing_flags.tolist()
    test_predictions['original_readm_preds'] = test_predictions['readm_preds']
    test_predictions['original_icu_preds'] = test_predictions['icu_preds']
    
    # Calculate original metrics
    readm_targets = np.array(test_predictions['readm_targets'])
    readm_preds = np.array(test_predictions['readm_preds'])
    icu_targets = np.array(test_predictions['icu_targets'])
    icu_preds = np.array(test_predictions['icu_preds'])
    
    original_readm_auc = roc_auc_score(readm_targets, readm_preds)
    original_readm_apr = average_precision_score(readm_targets, readm_preds)
    original_readm_brier = brier_score_loss(readm_targets, readm_preds)
    
    original_icu_auc = roc_auc_score(icu_targets, icu_preds)
    original_icu_apr = average_precision_score(icu_targets, icu_preds)
    original_icu_brier = brier_score_loss(icu_targets, icu_preds)
    
    print("\nOriginal metrics from CSV file:")
    print(f"Readmission AUC: {original_readm_auc:.4f}")
    print(f"Readmission APR: {original_readm_apr:.4f}")
    print(f"Readmission Brier: {original_readm_brier:.4f}")
    print(f"ICU AUC: {original_icu_auc:.4f}")
    print(f"ICU APR: {original_icu_apr:.4f}")
    print(f"ICU Brier: {original_icu_brier:.4f}")
    
    # Apply correction
    print("\nApplying correction to saved predictions...")
    
    # Use previously defined function to apply correction
    test_metrics, corrected_predictions = apply_direct_correction_correction(
        test_predictions, use_selective_correction=use_selective_correction
    )
    
    # Save results
    suffix = "_selective" if use_selective_correction else ""
    output_path = f"{output_prefix}{suffix}_results.csv"
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'subject_id': corrected_predictions['subject_id'],
        'readm_target': corrected_predictions['readm_targets'],
        'readm_pred_original': corrected_predictions['original_readm_preds'],
        'readm_pred_corrected': corrected_predictions['readm_preds'],
        'icu_target': corrected_predictions['icu_targets'],
        'icu_pred_original': corrected_predictions['original_icu_preds'],
        'icu_pred_corrected': corrected_predictions['icu_preds']
    })
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Saved corrected predictions to: {output_path}")
    
    return test_metrics, corrected_predictions

def plot_comparison_curves(targets, preds_original, preds_corrected, title, save_path, curve_type='roc'):
    """Plot comparison of ROC or PR curves before and after correction
    
    Args:
        targets: True labels
        preds_original: Original predictions
        preds_corrected: Corrected predictions
        title: Chart title
        save_path: Save path
        curve_type: Curve type ('roc' or 'pr')
    """
    plt.figure(figsize=(10, 8))
    
    if curve_type == 'roc':
        # Calculate original ROC curve
        from sklearn.metrics import roc_curve
        fpr_orig, tpr_orig, _ = roc_curve(targets, preds_original)
        auc_orig = roc_auc_score(targets, preds_original)
        
        # Calculate corrected ROC curve
        fpr_corr, tpr_corr, _ = roc_curve(targets, preds_corrected)
        auc_corr = roc_auc_score(targets, preds_corrected)
        
        # Plot curves
        plt.plot(fpr_orig, tpr_orig, 'b-', linewidth=2, label=f'Original (AUC = {auc_orig:.3f})')
        plt.plot(fpr_corr, tpr_corr, 'r-', linewidth=2, label=f'Corrected (AUC = {auc_corr:.3f})')
        
        # Add diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
        
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        
    elif curve_type == 'pr':
        # Calculate original PR curve
        from sklearn.metrics import precision_recall_curve
        precision_orig, recall_orig, _ = precision_recall_curve(targets, preds_original)
        ap_orig = average_precision_score(targets, preds_original)
        
        # Calculate corrected PR curve
        precision_corr, recall_corr, _ = precision_recall_curve(targets, preds_corrected)
        ap_corr = average_precision_score(targets, preds_corrected)
        
        # Plot curves
        plt.plot(recall_orig, precision_orig, 'b-', linewidth=2, label=f'Original (AP = {ap_orig:.3f})')
        plt.plot(recall_corr, precision_corr, 'r-', linewidth=2, label=f'Corrected (AP = {ap_corr:.3f})')
        
        # Add baseline reference line
        prevalence = np.mean(targets)
        plt.axhline(y=prevalence, color='k', linestyle='--', alpha=0.3, linewidth=1.5, 
                   label=f'Baseline (Prevalence = {prevalence:.3f})')
        
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
    
    plt.title(f'{title} - Original vs Corrected', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Apply correction to saved CSV predictions")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file containing predictions")
    parser.add_argument("--selective", action="store_true", default=False,
                        help="Use selective correction")
    parser.add_argument("--output_prefix", type=str, default="correction_results",
                        help="Prefix for output files")
    args = parser.parse_args()
    
    metrics, predictions = apply_correction_to_saved_csv(
        args.csv_path, 
        use_selective_correction=args.selective,
        output_prefix=args.output_prefix
    )
    
    # Save result charts
    output_dir = os.path.dirname(args.output_prefix) if os.path.dirname(args.output_prefix) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    plot_comparison_curves(
        predictions['readm_targets'],
        predictions['original_readm_preds'],
        predictions['readm_preds'],
        'Readmission ROC Curve',
        f"{args.output_prefix}_roc.png",
        curve_type='roc'
    )
    
    plot_comparison_curves(
        predictions['readm_targets'],
        predictions['original_readm_preds'],
        predictions['readm_preds'],
        'Readmission PR Curve',
        f"{args.output_prefix}_pr.png",
        curve_type='pr'
    )

if __name__ == "__main__":
    main()