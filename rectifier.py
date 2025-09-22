
import numpy as np
from scipy.stats import norm
import warnings
from typing import Dict, Tuple, Union, Optional, List, Any


class RectifierCorrector:
    
    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        
        self.alpha = alpha
        self.deltas: Dict[str, float] = {}         # Rectifiers (Δ_m)
        self.var_deltas: Dict[str, float] = {}     # Variances of rectifiers
        self.sample_counts: Dict[str, int] = {}    # Sample counts for each group
        self.initial_sample_counts: Dict[str, int] = {}
        
    def _get_key(self, modality_name: str, task: str = 'default') -> str:
        return f"{task}_{modality_name}"
    
    def fit(self, 
            preds_gold: np.ndarray, 
            labels_gold: np.ndarray, 
            modality_name: str,
            task: str = 'default',
            sample_weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
        
        # Input validation
        if len(preds_gold) == 0 or len(labels_gold) == 0:
            raise ValueError("Gold predictions or labels are empty")
        
        if len(preds_gold) != len(labels_gold):
            raise ValueError(f"Prediction length ({len(preds_gold)}) does not match label length ({len(labels_gold)})")
        
        # Convert inputs to numpy arrays if they aren't already
        preds_gold = np.asarray(preds_gold)
        labels_gold = np.asarray(labels_gold)
        
        # Check for NaN/inf values
        if np.isnan(preds_gold).any() or np.isnan(labels_gold).any() or \
        np.isinf(preds_gold).any() or np.isinf(labels_gold).any():
            warnings.warn("NaN or inf values detected in gold data. They will be removed.")
            mask = ~(np.isnan(preds_gold) | np.isnan(labels_gold) | 
                    np.isinf(preds_gold) | np.isinf(labels_gold))
            preds_gold = preds_gold[mask]
            labels_gold = labels_gold[mask]
            
            # Adjust sample weights if provided
            if sample_weights is not None:
                sample_weights = np.asarray(sample_weights)[mask]
            
        # Check if we still have enough data after filtering
        if len(preds_gold) < 10:
            warnings.warn(f"Too few valid samples ({len(preds_gold)}) for {modality_name}/{task} after filtering. "
                        f"Using zero rectifier for stability.")
            key = self._get_key(modality_name, task)
            self.deltas[key] = 0.0
            self.var_deltas[key] = 0.0
            self.sample_counts[key] = len(preds_gold)
            return 0.0, 0.0
        
        # Ensure predictions are within valid range [0, 1] for probability estimates
        if (preds_gold < 0).any() or (preds_gold > 1).any():
            warnings.warn("Some predictions are outside the valid range [0, 1]. Clipping values.")
            preds_gold = np.clip(preds_gold, 0.0, 1.0)
        
        # Similarly for binary labels
        if not np.all(np.isin(labels_gold, [0, 1])):
            warnings.warn("Some labels are not binary (0 or 1). Rounding to nearest binary value.")
            labels_gold = np.round(labels_gold).clip(0, 1)
        
        # Compute errors
        errors = preds_gold - labels_gold
        
        # Handle sample weights for weighted statistics
        if sample_weights is not None:
            # Validate sample weights
            if len(sample_weights) != len(preds_gold):
                warnings.warn("Sample weights length doesn't match predictions, ignoring weights.")
                sample_weights = None
            else:
                # Normalize weights to sum to 1
                weights = np.array(sample_weights, dtype=np.float64)
                if np.sum(weights) == 0:
                    warnings.warn("Sum of sample weights is zero, ignoring weights.")
                    sample_weights = None
                else:
                    weights = weights / np.sum(weights)
        
        # Compute the mean error (delta) - potentially weighted
        if sample_weights is not None:
            # Weighted mean
            delta = np.sum(weights * errors)
            
            # Weighted variance with Bessel's correction
            if len(errors) > 1:
                # Calculate effective sample size for Bessel's correction
                n_eff = 1.0 / np.sum(weights**2)
                var_delta = np.sum(weights * (errors - delta)**2) * (n_eff / (n_eff - 1))
            else:
                var_delta = 0.1
                warnings.warn("Only one sample available for variance estimation. Using default value.")
        else:
            # Unweighted mean
            delta = np.mean(errors)
            
            # Calculate sample size-aware variance with robust handling
            if len(errors) > 1:
                var_delta = np.var(errors, ddof=1)  # Use unbiased estimator with Bessel's correction
            else:
                # Can't compute variance with only one sample
                var_delta = 0.1  # Default to a reasonable non-zero variance for stability
                warnings.warn("Only one sample available for variance estimation. Using default value.")
        
        # Ensure variance is not too small (avoid numerical issues)
        var_delta = max(var_delta, 1e-4)
        
        # Clip the rectifier to avoid extreme corrections
        # This helps prevent training instability when applying corrections
        MAX_RECTIFIER = 0.3  # Maximum allowed absolute value for rectifier
        if abs(delta) > MAX_RECTIFIER:
            warnings.warn(f"Large rectifier value detected ({delta:.4f}) for {task}/{modality_name}. "
                        f"Clipping to ±{MAX_RECTIFIER}.")
            delta = np.sign(delta) * MAX_RECTIFIER
        
        # Check if the rectifier is stable given the sample size
        # For small sample sizes, we want to be more conservative
        effective_size = len(preds_gold)
        if sample_weights is not None:
            # Calculate effective sample size for weighted data
            effective_size = int((np.sum(weights) ** 2) / np.sum(weights**2))
        
        if effective_size < 30:
            # Scale down the rectifier for small sample sizes, but less conservatively 
            adjustment_factor = max(0.3, effective_size / 30)
            original_delta = delta
            delta = delta * adjustment_factor
            warnings.warn(f"Small effective sample size ({effective_size}) for {task}/{modality_name}. "
                        f"Scaling rectifier from {original_delta:.4f} to {delta:.4f}.")
        
        # Check for high variance relative to the rectifier magnitude
        # This indicates unstable estimation
        if var_delta > 0.2 and abs(delta) > 0.01:
            confidence_ratio = abs(delta) / np.sqrt(var_delta)
            if confidence_ratio < 0.5:  # Low confidence
                adjustment_factor = min(0.8, max(0.2, confidence_ratio))
                original_delta = delta
                delta = delta * adjustment_factor
                warnings.warn(f"Low confidence rectifier (ratio={confidence_ratio:.2f}) for {task}/{modality_name}. "
                            f"Scaling rectifier from {original_delta:.4f} to {delta:.4f}.")
        
        # Ensure the delta is not NaN
        if np.isnan(delta) or np.isinf(delta):
            warnings.warn(f"Invalid rectifier value. Setting to zero for {task}/{modality_name}.")
            delta = 0.0
        
        # Store results
        key = self._get_key(modality_name, task)
        self.deltas[key] = delta
        self.var_deltas[key] = var_delta
        self.sample_counts[key] = len(preds_gold)
        
        # Keep track of initial sample counts for future comparisons
        if not hasattr(self, 'initial_sample_counts'):
            self.initial_sample_counts = {}
            
        if key not in self.initial_sample_counts:
            self.initial_sample_counts[key] = len(preds_gold)
        
        # Log the rectifier information for debugging
        weight_info = "(weighted)" if sample_weights is not None else ""
        print(f"Rectifier for {task}/{modality_name} {weight_info}: Δ={delta:.4f}, σ²={var_delta:.6f}, n={len(preds_gold)}")
        
        return delta, var_delta
    
    def transform(self, 
                preds_unlabeled: np.ndarray, 
                modality_name: str, 
                task: str = 'default',
                epoch: int = None) -> Tuple[float, float, float, float, float]:

        # Input validation
        if len(preds_unlabeled) == 0:
            raise ValueError("Unlabeled predictions are empty")
        
        # Convert input to numpy array if it isn't already
        preds_unlabeled = np.asarray(preds_unlabeled)
        
        # Check for NaN/inf values
        if np.isnan(preds_unlabeled).any() or np.isinf(preds_unlabeled).any():
            warnings.warn("NaN or inf values detected in unlabeled predictions. They will be removed.")
            mask = ~(np.isnan(preds_unlabeled) | np.isinf(preds_unlabeled))
            preds_unlabeled = preds_unlabeled[mask]
            
            # If too many values were removed, be extra cautious
            if len(preds_unlabeled) < 0.5 * len(mask):
                warnings.warn(f"More than 50% of predictions contained NaN/inf. Using reduced correction.")
                
        # Ensure we still have data after filtering
        if len(preds_unlabeled) == 0:
            warnings.warn("All unlabeled predictions were invalid. Returning uncorrected zero.")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Check if rectifier exists for this modality/task
        key = self._get_key(modality_name, task)
        if key not in self.deltas:
            raise ValueError(f"No rectifier found for modality '{modality_name}' and task '{task}'. "
                            f"Call fit() first with gold standard data.")
            
        # Compute statistics for unlabeled data
        theta_tilde = np.mean(preds_unlabeled)
        var_f = np.var(preds_unlabeled, ddof=1) if len(preds_unlabeled) > 1 else 0.1
        
        # Get base correction
        original_delta = self.deltas[key]
        var_delta = self.var_deltas[key]
        
        # Initialize adjusted delta with original value
        adjusted_delta = original_delta * 1.2
        
        pred_std = np.sqrt(var_f)
        if pred_std > 0.25:
            variance_factor = max(0.3, 0.7 - (pred_std - 0.25))
            adjusted_delta = adjusted_delta * variance_factor
            
        impact_ratio = abs(adjusted_delta / max(0.01, theta_tilde))
        if impact_ratio > 0.15:
            adjusted_delta = 0.15 * theta_tilde * np.sign(adjusted_delta)
        
        # 1. Apply sample size-based adjustment
        # Check if we have initial sample count stored for this key
        if hasattr(self, 'initial_sample_counts') and key in self.initial_sample_counts:
            current_samples = self.sample_counts.get(key, 0)
            initial_samples = self.initial_sample_counts.get(key)
            
            # If sample size has decreased significantly
            if current_samples < initial_samples * 0.7:  # 30% or more decrease
                sample_ratio = max(0.3, current_samples / initial_samples)
                adjustment_factor = sample_ratio
                adjusted_delta = original_delta * adjustment_factor
                warnings.warn(f"Reducing delta for {task}/{modality_name} due to sample size decrease " 
                            f"({current_samples}/{initial_samples}). "
                            f"Adjustment: {original_delta:.4f} -> {adjusted_delta:.4f}")
        
        # 2. Apply epoch-based gradual introduction
        if epoch is not None:
            # Apply correction more aggressively
            phase_in_factor = min(1.0, (epoch - 10) / 20.0) if epoch > 10 else 0.1
            adjusted_delta = adjusted_delta * phase_in_factor
            
            # For debugging
            if phase_in_factor > 0:
                print(f"Epoch {epoch}: Applying {phase_in_factor:.2f} of correction for {task}/{modality_name}")
        
        # 3. Apply prediction variance-based adjustment
        # If variance of predictions is very high, be more conservative
        pred_std = np.sqrt(var_f)
        if pred_std > 0.3:  # High variance threshold
            variance_factor = max(0.4, 0.8 - (pred_std - 0.3))  # Scale down as variance increases
            prev_delta = adjusted_delta
            adjusted_delta = adjusted_delta * variance_factor
            if variance_factor < 0.9:
                warnings.warn(f"High prediction variance ({pred_std:.4f}) for {task}/{modality_name}. "
                            f"Reducing correction: {prev_delta:.4f} -> {adjusted_delta:.4f}")
        
        # 4. Apply magnitude-based thresholding
        # If the raw correction would change predictions by more than 20%, cap it
        impact_ratio = abs(adjusted_delta / max(0.01, theta_tilde))
        if impact_ratio > 0.2:  # 20% threshold
            prev_delta = adjusted_delta
            adjusted_delta = 0.2 * theta_tilde * np.sign(adjusted_delta)
            warnings.warn(f"Correction magnitude ({impact_ratio:.2f}x) too large for {task}/{modality_name}. "
                        f"Capping: {prev_delta:.4f} -> {adjusted_delta:.4f}")
        
        # 5. Apply final sanity check
        # Ensure the final corrected value is in [0, 1] for probabilities
        uncorrected = theta_tilde
        corrected = theta_tilde - adjusted_delta
        
        if corrected < 0 or corrected > 1:
            prev_delta = adjusted_delta
            # Adjust delta to keep corrected value in valid range with a small margin
            if corrected < 0:
                adjusted_delta = theta_tilde - 0.01  # Leave a small margin
            elif corrected > 1:
                adjusted_delta = theta_tilde - 0.99  # Leave a small margin
                
            warnings.warn(f"Correction would push estimate outside [0,1] for {task}/{modality_name}. "
                        f"Limiting: {prev_delta:.4f} -> {adjusted_delta:.4f}")
        
        # Compute final corrected estimate
        delta = adjusted_delta
        theta_rectifier = theta_tilde - delta
        
        # For logging & debugging
        if abs(original_delta - delta) > 0.01:
            print(f"Applied adaptive correction for {task}/{modality_name}: "
                f"Original Δ={original_delta:.4f} -> Adjusted Δ={delta:.4f}")
        
        return theta_rectifier, theta_tilde, var_f, delta, var_delta
    
    def rectifier_ci(self, 
              theta_tilde: float, 
              var_f: float, 
              delta: float, 
              var_delta: float, 
              n_gold: int, 
              n_unlabeled: int) -> Tuple[float, float, float]:

        # Input validation
        if n_gold <= 1 or n_unlabeled <= 1:
            raise ValueError(f"Sample sizes must be > 1, got n_gold={n_gold}, n_unlabeled={n_unlabeled}")
        
        if var_delta < 0 or var_f < 0:
            raise ValueError(f"Variances must be non-negative, got var_delta={var_delta}, var_f={var_f}")
        
        # Compute standard error
        se = np.sqrt(var_delta / n_gold + var_f / n_unlabeled)
        
        # Determine critical value
        z = norm.ppf(1 - self.alpha/2)
        
        # Compute point estimate and confidence interval
        center = theta_tilde - delta
        lower = center - z * se
        upper = center + z * se
        
        return lower, center, upper
    
    def get_all_corrections(self) -> Dict[str, Dict[str, Any]]:

        result = {}
        for key in self.deltas:
            result[key] = {
                'delta': self.deltas[key],
                'var_delta': self.var_deltas[key],
                'n_samples': self.sample_counts.get(key, 0)
            }
        return result
    
    def correct_batch_predictions(self, 
                                 predictions: np.ndarray, 
                                 missing_flags: np.ndarray,
                                 modality_idx: int,
                                 modality_name: str,
                                 task: str = 'default') -> np.ndarray:

        # Check if rectifier exists
        key = self._get_key(modality_name, task)
        if key not in self.deltas:
            raise ValueError(f"No rectifier found for modality '{modality_name}' and task '{task}'")
        
        # Convert to numpy arrays
        predictions = np.asarray(predictions)
        missing_flags = np.asarray(missing_flags)
        
        # Create a copy to avoid modifying the input
        corrected_preds = predictions.copy()
        
        # Apply correction only to samples missing this modality
        mask = missing_flags[:, modality_idx] == 0
        corrected_preds[mask] -= self.deltas[key]
        
        return corrected_preds
    
    def apply_rank_aware_correction(self, preds, missing_mask, modality_name, task='default'):
            key = self._get_key(modality_name, task)
            if key not in self.deltas:
                return preds.copy()  # Return original predictions if no correction value is available
            
            # Copy predictions to avoid modifying original data
            corrected = preds.copy()
            
            # Get correction parameters
            delta = self.deltas[key] * 0.7  # Reduce correction strength
            var_delta = self.var_deltas.get(key, 0.1)
            
            # Apply correction only to samples with missing modalities
            mask = missing_mask == 0
            
            if mask.any():
                # 1. Basic correction (affects Brier score)
                corrected[mask] -= delta
                
                # 2. Apply ranking-aware correction (affects AUC/APR)
                # Get predictions for missing samples
                missing_preds = corrected[mask]
                
                # Calculate correction strength factor based on prediction values
                # Middle region (0.4-0.6) gets stronger correction, edge regions get weaker correction
                middle_mask = (missing_preds >= 0.4) & (missing_preds <= 0.6)
                edge_mask = ~middle_mask
                
                # Extra correction for middle region - enhance correction effect
                if np.any(middle_mask):
                    middle_adjustment = (missing_preds[middle_mask] - 0.5).copy()
                    middle_adjustment = -np.sign(middle_adjustment) * 0.05  # Increased from 0.03 to 0.05
                    # Extra enhancement for values close to 0.5
                    close_to_middle = np.abs(missing_preds[middle_mask] - 0.5) < 0.05
                    if np.any(close_to_middle):
                        # The closer to 0.5, the stronger the correction
                        very_close_adjustment = -np.sign(middle_adjustment[close_to_middle]) * 0.07
                        middle_adjustment[close_to_middle] = very_close_adjustment
                    missing_preds[middle_mask] += middle_adjustment

                # Correction for edge regions - more refined adjustments
                if np.any(edge_mask):
                    # Create different edge regions for differential treatment
                    extreme_edge = (missing_preds[edge_mask] < 0.2) | (missing_preds[edge_mask] > 0.8)
                    moderate_edge = ~extreme_edge
                    
                    # Appropriate correction for extreme edges
                    if np.any(extreme_edge):
                        extreme_adjustment = np.sign(missing_preds[edge_mask][extreme_edge] - 0.5) * 0.01
                        missing_preds[edge_mask][extreme_edge] += extreme_adjustment
                    
                    # Enhanced correction for moderate edges
                    if np.any(moderate_edge):
                        moderate_adjustment = np.sign(missing_preds[edge_mask][moderate_edge] - 0.5) * 0.03
                        missing_preds[edge_mask][moderate_edge] += moderate_adjustment
                
                # Update corrected predictions
                corrected[mask] = missing_preds
            
            # Ensure prediction values are within valid range
            return np.clip(corrected, 0.0, 1.0)