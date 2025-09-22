import os
import numpy as np
import pandas as pd
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, brier_score_loss
from sklearn.impute import SimpleImputer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from collections import defaultdict
import argparse
import warnings
import sklearn
from functools import partial
import copy
import gc
from imblearn.over_sampling import SMOTENC, ADASYN
from imblearn.combine import SMOTETomek

# Disable unnecessary warnings
warnings.filterwarnings('ignore')

# Global constants
USE_AMP = False

#########################################
# 1. Enhanced Configuration
#########################################
class Config:
    # Data paths
    DATA_ROOT = "mimiciv_data/final_dataset"
    CXR_PATH = os.path.join(DATA_ROOT, "cxr_embeddings_aggregated.csv")
    FEATURES_PATH = os.path.join(DATA_ROOT, "patient_features_final_corrected.csv")
    TEXT_PATH = os.path.join(DATA_ROOT, "patient_text_embeddings_nan.csv")
    OUTPUT_DIR = "outputs_model"
    FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    
    # Create output directories
    for directory in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Model parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING = 15
    
    # NEW: Self-supervised pretraining parameters
    PRETRAIN_EPOCHS = 20
    PRETRAIN_LR = 1e-4
    PRETRAIN_WEIGHT_DECAY = 1e-4
    ENABLE_PRETRAINING = True
    
    # NEW: Focal Loss parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA_START = 2.0
    FOCAL_GAMMA_END = 5.0
    ENABLE_FOCAL_LOSS = True
    
    # NEW: Label Smoothing
    LABEL_SMOOTHING = 0.1
    ENABLE_LABEL_SMOOTHING = True
    
    # NEW: Test Time Augmentation
    TTA_ENABLED = True
    TTA_SAMPLES = 5
    TTA_NOISE_LEVEL = 0.01
    
    # NEW: Enhanced Curriculum Learning
    CURRICULUM_MODE = "adaptive"  # "fixed" or "adaptive"
    ENABLE_CURRICULUM = True
    DIFFICULTY_THRESHOLD = 0.8
    
    # Architecture parameters
    HIDDEN_DIM = 128
    
    # Input dimensions from data
    STRUCT_INPUT_DIM = 0  # Will be set dynamically
    IMG_EMB_DIM = 1024
    TEXT_EMB_DIM = 300
    RAD_EMB_DIM = 300
    
    # Training parameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True
    RANDOM_SEED = 42
    
    # Task parameters
    LABEL_READM = "readmission_30d"
    LABEL_ICU = "icu_need_after_discharge_90d"
    LABEL_MORTALITY = "death_after_discharge_180d"
    
    # Enhanced Task weights (dynamic adjustment)
    TASK_WEIGHTS = {
        'readmission': 1.2,
        'icu': 1.0,
        'mortality': 1.5,
        'missing': 0.2
    }
    
    # NEW: Adaptive task weights
    ADAPTIVE_TASK_WEIGHTS = {
        'readmission': {'min': 0.5, 'max': 2.0},
        'icu': {'min': 0.5, 'max': 1.5},
        'mortality': {'min': 1.0, 'max': 3.0}
    }
    
    # Class weights for imbalanced mortality (will be computed from data)
    CLASS_WEIGHTS = {
        'mortality': None  # Will be set based on data
    }
    
    # RB specific parameters
    RB_WEIGHT = 0.2
    RB_CONTRASTIVE_WEIGHT = 0.15
    RB_PRED_WEIGHT = 0.15
    
    # Attention parameters
    NUM_ATTENTION_HEADS = 8
    ATTENTION_DROPOUT = 0.1
    
    # Graph parameters
    GRAPH_HIDDEN_DIM = 64
    GRAPH_NUM_LAYERS = 2
    
    # Progressive training parameters
    STAGE1_RATIO = 0.3  # First 30% epochs with high-quality data
    STAGE2_RATIO = 0.7  # Next 40% epochs add medium-quality data
    
    # Column names
    IMG_EMB_COL = "agg_embedding"
    TEXT_EMB_COL = "discharge_embedding"
    RAD_EMB_COL = "radiology_embedding"
    
    # Regularization
    DROPOUT_RATE = 0.2
    
    # Feature normalization
    NORMALIZATION = "robust"  # Options: standard, robust, minmax, power
    
    # For reproducibility
    def set_seed(self, seed=None):
        if seed is None:
            seed = self.RANDOM_SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

config = Config()
config.set_seed()

#########################################
# 2. Enhanced Loss Functions
#########################################
class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss with dynamic gamma scheduling"""
    
    def __init__(self, alpha=None, gamma_start=2.0, gamma_end=5.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.reduction = reduction
        
    def forward(self, inputs, targets, epoch_ratio=0.0):
        """
        Args:
            inputs: Logits tensor [N, 1]
            targets: Target tensor [N, 1] 
            epoch_ratio: Training progress ratio (0.0 to 1.0)
        """
        # Dynamic gamma based on training progress
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * epoch_ratio
        
        # Convert logits to probabilities
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term: (1-p_t)^gamma
        focal_weight = (1 - p_t) ** gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for binary classification"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        """Apply label smoothing to targets"""
        smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, smooth_target)

class EnhancedCompoundLoss(nn.Module):
    """Enhanced compound loss with focal loss and label smoothing"""
    
    def __init__(self):
        super().__init__()
        
        # Task-specific loss functions
        if config.ENABLE_FOCAL_LOSS:
            self.readm_loss_fn = AdaptiveFocalLoss(
                alpha=config.FOCAL_ALPHA, 
                gamma_start=config.FOCAL_GAMMA_START,
                gamma_end=config.FOCAL_GAMMA_END
            )
            self.icu_loss_fn = AdaptiveFocalLoss(
                alpha=0.5, 
                gamma_start=1.5,
                gamma_end=3.0
            )
            self.mortality_loss_fn = AdaptiveFocalLoss(
                alpha=0.75,           # 降低到 0.5
                gamma_start=2.5,      # 降低到 1.5  
                gamma_end=5.0         # 降低到 3.0
            )
        else:
            self.readm_loss_fn = nn.BCEWithLogitsLoss()
            self.icu_loss_fn = nn.BCEWithLogitsLoss()
            self.mortality_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=config.CLASS_WEIGHTS['mortality'][1] if config.CLASS_WEIGHTS['mortality'] is not None else None
            )
        
        # Label smoothing
        if config.ENABLE_LABEL_SMOOTHING:
            self.label_smoothing = LabelSmoothingLoss(config.LABEL_SMOOTHING)
    
    def forward(self, outputs, targets, epoch_ratio=0.0, task_weights=None):
        """
        Args:
            outputs: Model outputs dict
            targets: Target dict
            epoch_ratio: Training progress (0.0 to 1.0)
            task_weights: Dynamic task weights
        """
        if task_weights is None:
            task_weights = config.TASK_WEIGHTS
            
        # Extract targets
        y_readm = targets['y_readm']
        y_icu = targets['y_icu'] 
        y_mortality = targets['y_mortality']
        
        # Calculate individual losses
        if config.ENABLE_FOCAL_LOSS:
            readm_loss = self.readm_loss_fn(outputs['readmission'], y_readm, epoch_ratio)
            icu_loss = self.icu_loss_fn(outputs['icu'], y_icu, epoch_ratio) 
            mortality_loss = self.mortality_loss_fn(outputs['mortality'], y_mortality, epoch_ratio)
        else:
            readm_loss = self.readm_loss_fn(outputs['readmission'], y_readm)
            icu_loss = self.icu_loss_fn(outputs['icu'], y_icu)
            mortality_loss = self.mortality_loss_fn(outputs['mortality'], y_mortality)
        
        # Apply label smoothing if enabled
        if config.ENABLE_LABEL_SMOOTHING:
            readm_loss = 0.7 * readm_loss + 0.3 * self.label_smoothing(outputs['readmission'], y_readm)
            icu_loss = 0.7 * icu_loss + 0.3 * self.label_smoothing(outputs['icu'], y_icu)
            mortality_loss = 0.9 * mortality_loss + 0.1 * self.label_smoothing(outputs['mortality'], y_mortality)
        
        # Combine with dynamic weights
        total_loss = (task_weights['readmission'] * readm_loss + 
                      task_weights['icu'] * icu_loss +
                      task_weights['mortality'] * mortality_loss)
        
        return total_loss, {
            'readm_loss': readm_loss.item(),
            'icu_loss': icu_loss.item(),
            'mortality_loss': mortality_loss.item(),
            'total_loss': total_loss.item()
        }

#########################################
# 3. Self-Supervised Pretraining Module
#########################################
class SelfSupervisedTasks(nn.Module):
    """Self-supervised learning tasks for MMNAR pretraining"""
    
    def __init__(self, hidden_dim, num_modalities=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Masked modality reconstruction heads
        self.modality_reconstruction_heads = nn.ModuleDict({
            'struct': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, config.STRUCT_INPUT_DIM)
            ),
            'img': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, config.IMG_EMB_DIM)
            ),
            'text': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, config.TEXT_EMB_DIM)
            ),
            'rad': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, config.RAD_EMB_DIM)
            )
        })
        
        # Missing pattern prediction head
        self.pattern_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 16)  # 16 possible patterns
        )
        
        # Contrastive learning projection
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
    def forward(self, shared_representation, original_modalities, missing_flags):
        """
        Args:
            shared_representation: [batch_size, hidden_dim]
            original_modalities: List of original modality tensors
            missing_flags: [batch_size, 4]
        """
        outputs = {}
        
        # Masked modality reconstruction
        modality_names = ['struct', 'img', 'text', 'rad']
        reconstructed_modalities = {}
        
        for i, modality_name in enumerate(modality_names):
            reconstructed = self.modality_reconstruction_heads[modality_name](shared_representation)
            reconstructed_modalities[modality_name] = reconstructed
        
        outputs['reconstructed_modalities'] = reconstructed_modalities
        
        # Missing pattern prediction
        pattern_logits = self.pattern_prediction_head(shared_representation)
        outputs['pattern_prediction'] = pattern_logits
        
        # Contrastive projection
        contrastive_features = self.contrastive_projection(shared_representation)
        outputs['contrastive_features'] = contrastive_features
        
        return outputs

class SelfSupervisedLoss(nn.Module):
    """Loss functions for self-supervised pretraining"""
    
    def __init__(self, reconstruction_weight=1.0, pattern_weight=0.5, contrastive_weight=0.3):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.pattern_weight = pattern_weight
        self.contrastive_weight = contrastive_weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperature = 0.1
        
    def reconstruction_loss(self, reconstructed, original, missing_flags):
        """Reconstruction loss for available modalities"""
        total_loss = 0.0
        valid_modalities = 0
        
        modality_names = ['struct', 'img', 'text', 'rad']
        for i, modality_name in enumerate(modality_names):
            if modality_name in reconstructed and len(original) > i:
                # Only calculate loss for available modalities
                mask = missing_flags[:, i].bool()
                if mask.sum() > 0:
                    recon_loss = self.mse_loss(
                        reconstructed[modality_name][mask],
                        original[i][mask]
                    )
                    total_loss += recon_loss
                    valid_modalities += 1
        
        return total_loss / max(valid_modalities, 1)
    
    def contrastive_loss(self, features, missing_patterns):
        """Contrastive loss for similar missing patterns"""
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarities
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive pairs mask (same missing pattern)
        pattern_mask = (missing_patterns.unsqueeze(0) == missing_patterns.unsqueeze(1)).float()
        
        # Remove self-similarities
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        pattern_mask = pattern_mask * eye_mask.float()
        
        # InfoNCE loss
        if pattern_mask.sum() > 0:
            # Compute log probabilities
            exp_similarities = torch.exp(similarity_matrix)
            denominator = exp_similarities.sum(dim=1, keepdim=True)
            
            # Positive similarities
            positive_similarities = similarity_matrix * pattern_mask
            
            # Calculate loss for samples with positive pairs
            valid_samples = (pattern_mask.sum(dim=1) > 0)
            if valid_samples.sum() > 0:
                loss = 0.0
                for i in range(batch_size):
                    if valid_samples[i]:
                        positive_exp = torch.exp(positive_similarities[i]).sum()
                        if positive_exp > 0:
                            loss += -torch.log(positive_exp / denominator[i])
                
                return loss / valid_samples.sum()
        
        return torch.tensor(0.0, device=features.device)
    
    def forward(self, ssl_outputs, original_modalities, missing_flags, missing_patterns):
        """
        Calculate total self-supervised loss
        
        Args:
            ssl_outputs: Outputs from SelfSupervisedTasks
            original_modalities: List of original modality tensors  
            missing_flags: [batch_size, 4]
            missing_patterns: [batch_size]
        """
        losses = {}
        
        # Reconstruction loss
        if 'reconstructed_modalities' in ssl_outputs:
            recon_loss = self.reconstruction_loss(
                ssl_outputs['reconstructed_modalities'],
                original_modalities,
                missing_flags
            )
            losses['reconstruction'] = recon_loss
        else:
            losses['reconstruction'] = torch.tensor(0.0)
        
        # Pattern prediction loss
        if 'pattern_prediction' in ssl_outputs:
            pattern_loss = self.ce_loss(ssl_outputs['pattern_prediction'], missing_patterns)
            losses['pattern'] = pattern_loss
        else:
            losses['pattern'] = torch.tensor(0.0)
        
        # Contrastive loss
        if 'contrastive_features' in ssl_outputs:
            contrastive_loss = self.contrastive_loss(
                ssl_outputs['contrastive_features'],
                missing_patterns
            )
            losses['contrastive'] = contrastive_loss
        else:
            losses['contrastive'] = torch.tensor(0.0)
        
        # Total loss
        total_loss = (self.reconstruction_weight * losses['reconstruction'] +
                     self.pattern_weight * losses['pattern'] +
                     self.contrastive_weight * losses['contrastive'])
        
        return total_loss, losses

#########################################
# 4. Enhanced Curriculum Learning
#########################################
class EnhancedCurriculumScheduler:
    """Enhanced curriculum learning with adaptive task weighting"""
    
    def __init__(self, dataset, train_indices, total_epochs=50):
        self.dataset = dataset
        self.train_indices = train_indices
        self.total_epochs = total_epochs
        
        # Task difficulty order (easiest to hardest)
        self.task_difficulty = ['icu', 'mortality', 'readmission']
        
        # Missing pattern complexity (simple to complex)
        self.pattern_complexity = self._analyze_pattern_complexity()
        
        # Performance tracking for adaptive weighting
        self.performance_history = {
            'readmission': [],
            'icu': [],
            'mortality': []
        }
        
        # Current adaptive weights
        self.current_task_weights = config.TASK_WEIGHTS.copy()
        
        print(f"\n=== Enhanced Curriculum Learning Initialized ===")
        print(f"Task difficulty order: {self.task_difficulty}")
        print(f"Pattern complexity levels: {len(self.pattern_complexity)} patterns")
        
    def _analyze_pattern_complexity(self):
        """Analyze missing pattern complexity based on modality count and clinical logic"""
        pattern_complexity = {}
        
        for idx in self.train_indices:
            pattern = self.dataset[idx]['missing_pattern'].item()
            if pattern not in pattern_complexity:
                # Calculate complexity score
                modality_count = bin(pattern).count('1')
                # More modalities = less complex missing pattern (more data available)
                complexity_score = 4 - modality_count
                
                # Adjust based on clinical importance
                if pattern == 15:  # Complete data (1111)
                    complexity_score = 0  # Easiest
                elif pattern & 8:  # Has structured data (most important)
                    complexity_score -= 0.5
                elif pattern & 4:  # Has text data
                    complexity_score -= 0.3
                    
                pattern_complexity[pattern] = max(0, complexity_score)
        
        # Sort patterns by complexity (easy to hard)
        sorted_patterns = sorted(pattern_complexity.items(), key=lambda x: x[1])
        return dict(sorted_patterns)
    
    def get_curriculum_stage(self, epoch):
        """Determine current curriculum stage"""
        progress = epoch / self.total_epochs
        
        if progress <= 0.3:
            return "easy"
        elif progress <= 0.7:
            return "medium"
        else:
            return "hard"
    
    def get_active_tasks(self, epoch):
        """Get active tasks based on curriculum stage"""
        stage = self.get_curriculum_stage(epoch)
        
        if stage == "easy":
            return ['icu']  # Start with easiest task
        elif stage == "medium":
            return ['icu', 'mortality']  # Add medium difficulty
        else:
            return ['icu', 'mortality', 'readmission']  # All tasks
    
    def get_pattern_difficulty_filter(self, epoch):
        """Get allowed missing patterns based on curriculum stage"""
        stage = self.get_curriculum_stage(epoch)
        total_patterns = len(self.pattern_complexity)
        
        if stage == "easy":
            # Use 40% easiest patterns
            allowed_count = max(1, int(total_patterns * 0.4))
        elif stage == "medium":
            # Use 70% easiest patterns
            allowed_count = max(1, int(total_patterns * 0.7))
        else:
            # Use all patterns
            allowed_count = total_patterns
        
        # Get patterns sorted by complexity
        sorted_patterns = sorted(self.pattern_complexity.items(), key=lambda x: x[1])
        allowed_patterns = [p[0] for p in sorted_patterns[:allowed_count]]
        
        return allowed_patterns
    
    def update_performance_history(self, epoch_metrics):
        """Update performance history for adaptive weighting"""
        # Map task names to metric keys
        task_metric_map = {
            'readmission': 'readm_auc',
            'icu': 'icu_auc', 
            'mortality': 'mortality_auc'
        }
        
        for task in self.task_difficulty:
            metric_key = task_metric_map[task]
            if metric_key in epoch_metrics:
                self.performance_history[task].append(epoch_metrics[metric_key])
                
                # Keep only recent history
                if len(self.performance_history[task]) > 10:
                    self.performance_history[task] = self.performance_history[task][-10:]
    
    def get_adaptive_task_weights(self, epoch):
        """Calculate adaptive task weights based on performance"""
        if not config.ENABLE_CURRICULUM or config.CURRICULUM_MODE != "adaptive":
            return config.TASK_WEIGHTS
        
        # Start with base weights
        weights = config.TASK_WEIGHTS.copy()
        
        # Adjust based on recent performance
        if epoch > 5:  # Only adjust after some training
            for task in self.task_difficulty:
                if len(self.performance_history[task]) >= 3:
                    recent_perf = np.mean(self.performance_history[task][-3:])
                    
                    # If performance is low, increase weight
                    if recent_perf < 0.75:  # Threshold for poor performance
                        multiplier = 1.5 - recent_perf  # Lower performance = higher weight
                        weights[task] = min(
                            config.ADAPTIVE_TASK_WEIGHTS[task]['max'],
                            weights[task] * multiplier
                        )
                    elif recent_perf > 0.85:  # Good performance, can reduce weight
                        multiplier = 0.8 + 0.2 * recent_perf
                        weights[task] = max(
                            config.ADAPTIVE_TASK_WEIGHTS[task]['min'],
                            weights[task] * multiplier
                        )
        
        self.current_task_weights = weights
        return weights
    
    def get_curriculum_indices(self, epoch):
        """Get training indices based on curriculum stage"""
        if not config.ENABLE_CURRICULUM:
            return self.train_indices
        
        allowed_patterns = self.get_pattern_difficulty_filter(epoch)
        
        # Filter indices based on allowed patterns
        curriculum_indices = []
        for idx in self.train_indices:
            pattern = self.dataset[idx]['missing_pattern'].item()
            if pattern in allowed_patterns:
                curriculum_indices.append(idx)
        
        # Ensure we have enough samples
        if len(curriculum_indices) < len(self.train_indices) * 0.1:
            print(f"Warning: Curriculum filtering too aggressive, using all indices")
            return self.train_indices
            
        return curriculum_indices

#########################################
# 5. Advanced Sampling Techniques (Enhanced)
#########################################
class AdvancedSampler:
    """Enhanced sampling techniques with curriculum awareness"""
    
    def __init__(self, dataset, train_indices):
        self.dataset = dataset
        self.train_indices = train_indices
        self.mortality_labels = np.array([dataset[i]['y_mortality'].item() for i in train_indices])
        self.readm_labels = np.array([dataset[i]['y_readm'].item() for i in train_indices])
        self.icu_labels = np.array([dataset[i]['y_icu'].item() for i in train_indices])
        self.missing_patterns = np.array([dataset[i]['missing_pattern'].item() for i in train_indices])
        
    def create_curriculum_aware_sampler(self, allowed_patterns=None, focus_task='mortality'):
        """Create curriculum-aware weighted sampler"""
        
        # Filter indices if patterns are specified
        if allowed_patterns is not None:
            valid_mask = np.isin(self.missing_patterns, allowed_patterns)
            if valid_mask.sum() == 0:
                print("Warning: No valid samples for curriculum, using all")
                valid_mask = np.ones(len(self.train_indices), dtype=bool)
        else:
            valid_mask = np.ones(len(self.train_indices), dtype=bool)
        
        # Get labels for the focus task
        if focus_task == 'mortality':
            focus_labels = self.mortality_labels[valid_mask]
        elif focus_task == 'readmission':
            focus_labels = self.readm_labels[valid_mask]
        else:  # icu
            focus_labels = self.icu_labels[valid_mask]
        
        # Calculate inverse frequency weights
        unique_labels, label_counts = np.unique(focus_labels, return_counts=True)
        total_samples = len(focus_labels)
        
        if len(unique_labels) > 1:
            # Inverse frequency weighting
            label_weights = total_samples / (len(unique_labels) * label_counts)
            sample_weights = label_weights[focus_labels.astype(int)]
        else:
            sample_weights = np.ones(len(focus_labels))
        
        # Add pattern diversity weighting
        patterns_in_curriculum = self.missing_patterns[valid_mask]
        unique_patterns, pattern_counts = np.unique(patterns_in_curriculum, return_counts=True)
        pattern_weights = total_samples / (len(unique_patterns) * pattern_counts)
        pattern_weight_dict = dict(zip(unique_patterns, pattern_weights))
        pattern_sample_weights = np.array([pattern_weight_dict[p] for p in patterns_in_curriculum])
        
        # Combine weights (focus on main task, some pattern diversity)
        combined_weights = 0.8 * sample_weights + 0.2 * pattern_sample_weights
        
        # Normalize
        combined_weights = combined_weights / np.sum(combined_weights) * len(combined_weights)
        
        # Create full weight array
        full_weights = np.ones(len(self.train_indices))
        full_weights[valid_mask] = combined_weights
        full_weights[~valid_mask] = 0  # Zero weight for excluded samples
        
        return WeightedRandomSampler(
            weights=full_weights,
            num_samples=int(valid_mask.sum()),
            replacement=True
        )

#########################################
# 6. Test Time Augmentation
#########################################
class TestTimeAugmentation:
    """Test time augmentation for multimodal data"""
    
    def __init__(self, num_augments=5, noise_level=0.01):
        self.num_augments = num_augments
        self.noise_level = noise_level
    
    def augment_batch(self, batch):
        """Apply augmentation to a batch"""
        augmented_batch = {}
        
        for key, value in batch.items():
            if key in ['x_struct', 'x_img', 'x_text', 'x_rad']:
                # Add small gaussian noise
                noise = torch.randn_like(value) * self.noise_level
                augmented_batch[key] = value + noise
            else:
                # Keep other fields unchanged
                augmented_batch[key] = value
                
        return augmented_batch
    
    def predict_with_tta(self, model, batch, device):
        """Make predictions with test time augmentation"""
        if not config.TTA_ENABLED:
            # No TTA, just normal prediction
            with torch.no_grad():
                return model(
                    batch['x_struct'].to(device),
                    batch['x_img'].to(device), 
                    batch['x_text'].to(device),
                    batch['x_rad'].to(device),
                    batch['missing_flags'].to(device)
                )
        
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            original_pred = model(
                batch['x_struct'].to(device),
                batch['x_img'].to(device),
                batch['x_text'].to(device), 
                batch['x_rad'].to(device),
                batch['missing_flags'].to(device)
            )
            predictions.append(original_pred)
            
            # Augmented predictions
            for _ in range(self.num_augments - 1):
                aug_batch = self.augment_batch(batch)
                aug_pred = model(
                    aug_batch['x_struct'].to(device),
                    aug_batch['x_img'].to(device),
                    aug_batch['x_text'].to(device),
                    aug_batch['x_rad'].to(device),
                    aug_batch['missing_flags'].to(device)
                )
                predictions.append(aug_pred)
        
        # Average predictions
        avg_outputs = {}
        for key in predictions[0].keys():
            if key in ['readmission', 'icu', 'mortality']:
                stacked = torch.stack([pred[key] for pred in predictions])
                avg_outputs[key] = stacked.mean(dim=0)
            else:
                avg_outputs[key] = predictions[0][key]  # Use original for other outputs
        
        return avg_outputs

#########################################
# 7. Continue with existing classes (with minor enhancements)
#########################################

# Cross-Modal Attention (keep existing)
class CrossModalAttention(nn.Module):
    """
    Cross-modal fusion with explicit z-gating.
    - Gates for each modality are computed from z = MLP(δ), not from (feature + pattern).
    - Pattern embeddings are still used for cross-attention, preserving pattern awareness.
    Expected inputs:
      modality_features: list of 4 tensors [B, D]
      missing_flags:     tensor [B, 4] in {0,1}
      missing_patterns:  tensor [B]  (categorical index 0..15)
      missing_repr:      tensor [B, D]  (this is z = MLP(δ))
    Output:
      aggregated:        tensor [B, D]
      attn_weights, cross_attn_weights: attention maps from self- and cross-attention
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # (1) Self-/Cross-attention blocks
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # (2) Norms + FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # (3) Modality-specific gates: W_m z + b -> sigmoid
        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(4)
        ])

        # (4) Missing-pattern embeddings for cross-attention (keep existing)
        self.pattern_embeddings = nn.Embedding(16, hidden_dim)

    def forward(self, modality_features, missing_flags, missing_patterns, missing_repr):
        """
        modality_features: list of 4 tensors [B, D]
        missing_flags:     [B, 4] bool/int {0,1}
        missing_patterns:  [B]
        missing_repr:      [B, D]  (z = MLP(δ))
        """
        B = modality_features[0].size(0)

        # ---- z-gate: gate_m = sigmoid(W_m z + b), then  \tilde{e}_m = δ_m * gate_m * e_m
        gated_features = []
        for i, (feat, gate_m) in enumerate(zip(modality_features, self.modality_gates)):
            gate_w = gate_m(missing_repr)                  # [B, D]
            gated = feat * gate_w * missing_flags[:, i:i+1].float()
            gated_features.append(gated)                   # 4 × [B, D]

        # Stack to [B, 4, D]
        stacked = torch.stack(gated_features, dim=1)

        # key_padding_mask: True = pad/ignore; 我们要“屏蔽缺失模态”
        key_pad_mask = ~(missing_flags.bool())            # [B, 4]

        # ---- self-attention over available modalities
        self_attended, self_attn_w = self.self_attention(
            stacked, stacked, stacked, key_padding_mask=key_pad_mask
        )
        self_attended = self.norm1(self_attended + stacked)

        # ---- cross-attention with pattern embedding
        pat = self.pattern_embeddings(missing_patterns)    # [B, D]
        pat = pat.unsqueeze(1).expand(-1, 4, -1)           # [B, 4, D]
        cross_attended, cross_attn_w = self.cross_attention(
            self_attended, pat, pat, key_padding_mask=key_pad_mask
        )
        cross_attended = self.norm2(cross_attended + self_attended)

        # ---- FFN + residual
        out = self.ffn(cross_attended)
        out = self.norm3(out + cross_attended)            # [B, 4, D]

        # ---- availability-weighted aggregation to patient-level h_i
        weights = missing_flags.float().unsqueeze(-1)     # [B, 4, 1]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        aggregated = (out * weights).sum(dim=1)           # [B, D]

        return aggregated, self_attn_w, cross_attn_w

# Graph Neural Network Fusion (keep existing)
class GraphModalityFusion(nn.Module):
    """Graph-based fusion inspired by MUSE for population-level MMNAR modeling"""
    
    def __init__(self, hidden_dim, graph_hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.num_layers = num_layers
        
        # Patient and modality node embeddings
        self.patient_proj = nn.Linear(hidden_dim, graph_hidden_dim)
        self.modality_embeddings = nn.Parameter(torch.randn(4, graph_hidden_dim))
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(graph_hidden_dim, graph_hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(graph_hidden_dim, hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, patient_features, missing_flags, batch_indices=None):
        batch_size = patient_features.size(0)
        device = patient_features.device
        
        # Project patient features to graph space
        patient_graph_features = self.patient_proj(patient_features)
        
        # Create bipartite graph adjacency
        adjacency = missing_flags.float()
        
        # Expand modality embeddings for the batch
        modality_features = self.modality_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Apply graph attention layers
        current_patient_features = patient_graph_features
        current_modality_features = modality_features
        
        for layer in self.graph_layers:
            # Patient-to-modality attention
            updated_patient_features = layer(
                current_patient_features,
                current_modality_features,
                adjacency
            )
            
            # Update features
            current_patient_features = updated_patient_features
        
        # Project back to original space
        graph_enhanced_features = self.output_proj(current_patient_features)
        
        # Residual connection and normalization
        output_features = self.norm(graph_enhanced_features + patient_features)
        
        return output_features

class GraphAttentionLayer(nn.Module):
    """Graph attention layer for patient-modality bipartite graph"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
    def forward(self, patient_features, modality_features, adjacency):
        batch_size = patient_features.size(0)
        
        # Prepare for attention: patient as query, modalities as key/value
        queries = patient_features.unsqueeze(1)
        keys = modality_features
        values = modality_features
        
        # Create attention mask from adjacency
        attn_mask = (adjacency == 0)
        
        # Apply attention
        attended_output, attn_weights = self.attention(
            queries, keys, values,
            key_padding_mask=attn_mask
        )
        
        # Remove sequence dimension
        attended_output = attended_output.squeeze(1)
        
        # Residual connection and normalization
        attended_output = self.norm1(attended_output + patient_features)
        
        # Feed-forward network
        ffn_output = self.ffn(attended_output)
        
        # Final normalization
        output = self.norm2(ffn_output)
        
        return output

# Modality Encoder (keep existing)
class ModalityEncoder(nn.Module):
    """Enhanced encoder for a specific modality with MMNAR awareness"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Representation Balancing Module (keep existing)
class RepresentationBalancingModule(nn.Module):
    """Enhanced representation balancing module for MMNAR patterns"""
    def __init__(self, hidden_dim, num_modalities=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Modality predictors with enhanced architecture
        self.modality_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * (num_modalities - 1), hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE * 0.5),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_modalities)
        ])
        
        # Enhanced feature calibration network
        self.calibration = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Contrastive projection head
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
    
    def forward(self, features, modality_features, missing_flags):
        batch_size = features.size(0)
        
        # Predict each modality from others
        predicted_modalities = []
        
        for i in range(self.num_modalities):
            # Collect features from all other modalities
            other_features = []
            for j in range(self.num_modalities):
                if j != i:
                    masked_feature = modality_features[j] * missing_flags[:, j].unsqueeze(1)
                    other_features.append(masked_feature)
            
            # Concatenate other modality features
            others_concat = torch.cat(other_features, dim=1)
            
            # Predict this modality
            pred_modality = self.modality_predictors[i](others_concat)
            predicted_modalities.append(pred_modality)
        
        # Create enhanced representation with missing pattern information
        calibrated_features = self.calibration(
            torch.cat([features, missing_flags], dim=1)
        )
        
        # Apply contrastive projection
        contrastive_features = self.contrastive_projection(calibrated_features)
        
        # Reconstruct missing modalities with improved strategy
        reconstructed_modalities = []
        for i in range(self.num_modalities):
            is_present = missing_flags[:, i].bool()
            reconstructed = torch.zeros_like(modality_features[i])
            
            # Copy original features where present
            if is_present.any():
                reconstructed[is_present] = modality_features[i][is_present]
            
            # Fill in predicted features where missing
            is_missing = ~is_present
            if is_missing.any():
                pred = predicted_modalities[i]
                if pred.dtype != reconstructed.dtype:
                    pred = pred.to(reconstructed.dtype)
                reconstructed[is_missing] = pred[is_missing]
            
            reconstructed_modalities.append(reconstructed)
        
        return {
            'enhanced_features': calibrated_features,
            'predicted_modalities': predicted_modalities,
            'reconstructed_modalities': reconstructed_modalities,
            'contrastive_features': contrastive_features
        }

#########################################
# 8. Enhanced Model with Self-Supervised Capabilities
#########################################
class EnhancedMMNARModelWithSSL(nn.Module):
    """Enhanced MMNAR model with self-supervised learning capabilities"""
    
    def __init__(self, ssl_mode=False):
        super().__init__()
        self.ssl_mode = ssl_mode
        
        # Modality encoders
        self.struct_encoder = ModalityEncoder(config.STRUCT_INPUT_DIM, config.HIDDEN_DIM)
        self.img_encoder = ModalityEncoder(config.IMG_EMB_DIM, config.HIDDEN_DIM)
        self.text_encoder = ModalityEncoder(config.TEXT_EMB_DIM, config.HIDDEN_DIM)
        self.rad_encoder = ModalityEncoder(config.RAD_EMB_DIM, config.HIDDEN_DIM)
        
        # Missing embedding with enhanced architecture
        self.missing_embedding = nn.Sequential(
            nn.Linear(4, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM)
        )
        
        # Initial fusion layer
        self.initial_fusion = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 4 + config.HIDDEN_DIM, config.HIDDEN_DIM * 2),
            nn.LayerNorm(config.HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM)
        )
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            config.HIDDEN_DIM, 
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.ATTENTION_DROPOUT
        )
        
        # Graph-based fusion
        self.graph_fusion = GraphModalityFusion(
            config.HIDDEN_DIM,
            graph_hidden_dim=config.GRAPH_HIDDEN_DIM,
            num_layers=config.GRAPH_NUM_LAYERS
        )
        
        # Representation Balancing module
        self.representation_balancing = RepresentationBalancingModule(
            hidden_dim=config.HIDDEN_DIM,
            num_modalities=4
        )
        
        # Self-supervised learning tasks
        if config.ENABLE_PRETRAINING:
            self.ssl_tasks = SelfSupervisedTasks(config.HIDDEN_DIM, num_modalities=4)
        
        # Missing pattern classifier
        self.missing_pattern_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, 16)  # 16 possible missing patterns
        )
        
        # Enhanced prediction heads
        self.readm_head = self._create_prediction_head(task_specific=True)
        self.icu_head = self._create_prediction_head(task_specific=True)
        self.mortality_head = self._create_mortality_head()
        
        print("Enhanced MMNAR Model with Self-Supervised Learning initialized")
    
    def _create_prediction_head(self, task_specific=True):
        """Create enhanced prediction head"""
        if task_specific:
            return nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
                nn.LayerNorm(config.HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.LayerNorm(config.HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE * 0.5),
                nn.Linear(config.HIDDEN_DIM // 2, 1)
            )
        else:
            return nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(config.HIDDEN_DIM // 2, 1)
            )
    
    def _create_mortality_head(self):
        """Create enhanced mortality prediction head"""
        return nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM * 2),
            nn.LayerNorm(config.HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 1.2),
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )
    
    def forward(self, x_struct, x_img, x_text, x_rad, missing_flags):
        # 1) Encode each modality
        h_struct = self.struct_encoder(x_struct)
        h_img    = self.img_encoder(x_img)
        h_text   = self.text_encoder(x_text)
        h_rad    = self.rad_encoder(x_rad)

        # 2) Apply missing masks
        h_struct = h_struct * missing_flags[:, 0:1]
        h_img    = h_img    * missing_flags[:, 1:2]
        h_text   = h_text   * missing_flags[:, 2:3]
        h_rad    = h_rad    * missing_flags[:, 3:4]
        modality_features = [h_struct, h_img, h_text, h_rad]

        # 3) z = MLP(δ)
        missing_repr = self.missing_embedding(missing_flags)   # [B, D]

        # 4) Initial fusion (unchanged)
        combined = torch.cat(modality_features + [missing_repr], dim=1)
        initial_features = self.initial_fusion(combined)

        # 5) Cross-modal attention with explicit z-gate
        #    pattern index: sum( δ * 2^idx )
        pattern_ids = torch.sum(
            missing_flags * (2 ** torch.arange(4, device=missing_flags.device)),
            dim=1
        ).long()
        attended_features, self_attn, cross_attn = self.cross_modal_attention(
            modality_features, missing_flags, pattern_ids, missing_repr
        )

        # 6) Graph fusion (unchanged)
        graph_enhanced = self.graph_fusion(attended_features, missing_flags)

        # 7) Merge enhanced features
        enhanced_features = (initial_features + attended_features + graph_enhanced) / 3.0

        # 8) Representation balancing
        rb = self.representation_balancing(enhanced_features, modality_features, missing_flags)

        # 9) Final features for prediction
        final_features = rb['enhanced_features']

        # 10) Outputs
        outputs = {
            'enhanced_features': final_features,
            'original_features': modality_features,
            'attention_weights': self_attn,
            'cross_attention_weights': cross_attn
        }

        # 11) SSL heads (if enabled)
        if self.ssl_mode and hasattr(self, 'ssl_tasks'):
            ssl_outputs = self.ssl_tasks(final_features, modality_features, missing_flags)
            outputs.update(ssl_outputs)

        # 12) Supervised heads
        if not self.ssl_mode:
            # *** Align with Algorithm 3: g(h_i) ***
            missing_pattern_logits = self.missing_pattern_classifier(final_features)

            readm_pred    = self.readm_head(final_features)
            icu_pred      = self.icu_head(final_features)
            mortality_pred= self.mortality_head(final_features)

            outputs.update({
                'readmission': readm_pred,
                'icu': icu_pred,
                'mortality': mortality_pred,
                'missing_pattern': missing_pattern_logits,
                'reconstructed_features': rb['reconstructed_modalities'],
                'contrastive_features': rb['contrastive_features']
            })

        return outputs

#########################################
# 9. Dataset (with SSL support)
#########################################
class MultiModalDataset(Dataset):
    def _clean_nan_values(self):
        """Clean NaN values in numerical columns while preserving modality missing information."""
        print("\n=== Data Cleaning Process Started ===")
        
        try:
            # 1. Clean structured data
            struct_num_cols = self.df.select_dtypes(include=["number"]).columns
            struct_obj_cols = self.df.select_dtypes(include=["object"]).columns
            
            # Exclude ID and target variables
            exclude_cols = ['subject_id', config.LABEL_READM, config.LABEL_ICU, config.LABEL_MORTALITY]
            feature_num_cols = [col for col in struct_num_cols if col not in exclude_cols]
            
            # Check for NaNs in structured data
            nan_counts = self.df[feature_num_cols].isna().sum()
            if nan_counts.sum() > 0:
                print(f"Found {nan_counts.sum()} NaN values in structured data, will fill with column means")
                
                # Calculate mean for each column and fill
                for col in feature_num_cols:
                    col_mean = self.df[col].mean()
                    nan_count = self.df[col].isna().sum()
                    
                    if nan_count > 0:
                        if pd.isna(col_mean):  # Handle all-NaN columns
                            print(f"  Warning: '{col}' column all NaN, filling with 0")
                            self.df[col] = self.df[col].fillna(0)
                        else:
                            self.df[col] = self.df[col].fillna(col_mean)
                            print(f"  '{col}' column: {nan_count} NaNs filled with mean {col_mean:.4f}")
            
            # Check categorical structured data
            if len(struct_obj_cols) > 0:
                for col in struct_obj_cols:
                    if col not in exclude_cols:
                        nan_count = self.df[col].isna().sum()
                        if nan_count > 0:
                            print(f"  '{col}' column: {nan_count} NaNs filled with empty string")
                            self.df[col] = self.df[col].fillna("")
            
            # 2. Image embeddings - handle NaNs but preserve missing modality
            if hasattr(self, 'img_df'):
                print("\nProcessing image embeddings:")
                if config.IMG_EMB_COL in self.img_df.columns:
                    valid_rows = ~self.img_df[config.IMG_EMB_COL].isna()
                    nan_in_embeddings = 0
                    
                    for idx, embedding_str in self.img_df.loc[valid_rows, config.IMG_EMB_COL].items():
                        if isinstance(embedding_str, str):
                            try:
                                clean_str = embedding_str.strip()
                                if not (clean_str.startswith('[') and clean_str.endswith(']')):
                                    clean_str = '[' + clean_str.strip('[]') + ']'
                                
                                arr = np.array(json.loads(clean_str), dtype=np.float32)
                                
                                if np.isnan(arr).any():
                                    nan_in_embeddings += 1
                                    arr = np.nan_to_num(arr, nan=0.0)
                                    self.img_df.at[idx, config.IMG_EMB_COL] = json.dumps(arr.tolist())
                            except:
                                pass
                    
                    if nan_in_embeddings > 0:
                        print(f"  Fixed {nan_in_embeddings} image embeddings containing NaN values")
                else:
                    print(f"  Warning: '{config.IMG_EMB_COL}' column not found in image DataFrame")
            
            # 3. Text embeddings - handle NaNs but preserve missing modality
            if hasattr(self, 'text_df'):
                print("\nProcessing text embeddings:")
                
                for embed_col in [config.TEXT_EMB_COL, config.RAD_EMB_COL]:
                    if embed_col in self.text_df.columns:
                        valid_rows = ~self.text_df[embed_col].isna()
                        nan_in_embeddings = 0
                        
                        for idx, embedding_str in self.text_df.loc[valid_rows, embed_col].items():
                            if isinstance(embedding_str, str) and embedding_str.strip():
                                try:
                                    clean_str = embedding_str.strip()
                                    if not (clean_str.startswith('[') and clean_str.endswith(']')):
                                        clean_str = '[' + clean_str.strip('[]') + ']'
                                    
                                    arr = np.array(json.loads(clean_str), dtype=np.float32)
                                    
                                    if np.isnan(arr).any():
                                        nan_in_embeddings += 1
                                        arr = np.nan_to_num(arr, nan=0.0)
                                        self.text_df.at[idx, embed_col] = json.dumps(arr.tolist())
                                except:
                                    pass
                        
                        if nan_in_embeddings > 0:
                            print(f"  Fixed {nan_in_embeddings} {embed_col} embeddings containing NaN values")
                    else:
                        print(f"  Warning: '{embed_col}' column not found in text DataFrame")
            
            # 4. Handle outliers in structured data
            print("\nHandling outliers in structured data:")
            outlier_cols = []
            
            for col in feature_num_cols:
                try:
                    # Calculate statistics
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    
                    # Skip constant columns
                    if std_val == 0:
                        continue
                    
                    # Identify outliers beyond 3 std
                    upper_bound = mean_val + 3*std_val
                    lower_bound = mean_val - 3*std_val
                    
                    # Count outliers
                    upper_outliers = (self.df[col] > upper_bound).sum()
                    lower_outliers = (self.df[col] < lower_bound).sum()
                    
                    if upper_outliers > 0 or lower_outliers > 0:
                        outlier_cols.append((col, upper_outliers, lower_outliers))
                        
                        # Cap outliers
                        self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                        self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                except Exception as e:
                    print(f"  Error processing outliers in '{col}': {e}")
            
            if outlier_cols:
                print(f"  Found and processed outliers in {len(outlier_cols)} columns:")
                for col, upper, lower in outlier_cols:
                    total = upper + lower
                    print(f"  - {col}: {total} outliers ({upper} high, {lower} low) - "
                        f"{total/len(self.df)*100:.2f}% of values")
            else:
                print("  No significant outliers detected in numerical columns")
            
            # 5. Final validation
            final_nan_count = self.df[feature_num_cols].isna().sum().sum()
            if final_nan_count > 0:
                print(f"\nWarning: {final_nan_count} NaN values remain after cleaning!")
            else:
                print("\nStructured data cleaning completed. No NaNs remain in numerical columns.")
                
            print("Note: We preserve NaN values for missing modalities as this is information the model needs to learn.")
            
        except Exception as e:
            print(f"Error in data cleaning: {e}")
            print("Falling back to basic NaN handling...")
            
            # Simple cleaning as fallback
            num_cols = self.df.select_dtypes(include=["number"]).columns
            exclude_cols = ['subject_id', config.LABEL_READM, config.LABEL_ICU, config.LABEL_MORTALITY]
            feature_num_cols = [col for col in num_cols if col not in exclude_cols]
            
            for col in feature_num_cols:
                col_mean = self.df[col].mean()
                if pd.isna(col_mean):
                    self.df[col] = self.df[col].fillna(0)
                else:
                    self.df[col] = self.df[col].fillna(col_mean)
            
            obj_cols = self.df.select_dtypes(include=["object"]).columns
            self.df[obj_cols] = self.df[obj_cols].fillna("")
                
    def __init__(self, struct_path, img_path, text_path, normalize=True, train_mode=True):
        # Load structured data
        struct_df = pd.read_csv(struct_path)
        
        # Load image embeddings
        img_df = pd.read_csv(img_path)
        img_df = img_df.set_index('subject_id')
        
        # Load text embeddings
        import csv
        text_df = pd.read_csv(
            text_path,
            engine='python',
            on_bad_lines='warn',
            quotechar='"',
            escapechar='\\',
            quoting=csv.QUOTE_MINIMAL
        )
        text_df = text_df.set_index('subject_id')
        
        # Merge dataframes
        self.df = struct_df.copy()

        # Simple basic imputation for structured data
        num = self.df.select_dtypes(include=["number"]).columns
        obj = self.df.select_dtypes(include=["object"]).columns
        self.df[num] = self.df[num].fillna(0.0)
        self.df[obj] = self.df[obj].fillna("")       
        
        # Set input dimension based on structured data
        self.train_mode = train_mode
        
        # CRITICAL: Calculate the correct input dimension
        exclude_cols = ['subject_id', config.LABEL_READM, config.LABEL_ICU, config.LABEL_MORTALITY]
        leaking_features = ['idx_discharge_location', 'idx_severity_mode']
        
        # Add modality existence flags (these will be added to df later)
        self.df['has_struct'] = 1  # Structured data is always available
        self.df['has_img'] = self.df['subject_id'].isin(img_df.index).astype(int)
        self.df['has_text'] = self.df['subject_id'].apply(
            lambda x: 1 if x in text_df.index and isinstance(text_df.loc[x, config.TEXT_EMB_COL], str) 
                        and len(text_df.loc[x, config.TEXT_EMB_COL].strip()) > 0 else 0
        )
        self.df['has_rad'] = self.df['subject_id'].apply(
            lambda x: 1 if x in text_df.index and isinstance(text_df.loc[x, config.RAD_EMB_COL], str) 
                        and len(text_df.loc[x, config.RAD_EMB_COL].strip()) > 0 else 0
        )
        
        # Add missing pattern code (0-15)
        self.df['missing_pattern'] = self.df.apply(
            lambda row: (int(row['has_struct']) << 3) + 
                    (int(row['has_img']) << 2) + 
                    (int(row['has_text']) << 1) + 
                    int(row['has_rad']), axis=1
        )
        
        # Calculate the exact number of features that will be processed
        modality_flags = ['has_struct', 'has_img', 'has_text', 'has_rad', 'missing_pattern']
        all_exclude = exclude_cols + leaking_features + modality_flags
        
        # Count features that will actually be used
        feature_count = 0
        for col in self.df.columns:
            if col not in all_exclude:
                feature_count += 1
        
        config.STRUCT_INPUT_DIM = feature_count
        
        print(f"Calculated STRUCT_INPUT_DIM: {config.STRUCT_INPUT_DIM}")
        print(f"Total columns: {len(self.df.columns)}")
        print(f"Excluded columns: {len(all_exclude)}")
        
        # Save original dataframes
        self.struct_df = struct_df
        self.img_df = img_df
        self.text_df = text_df
        
        # Normalization setup but don't apply yet
        self.normalize = normalize
        self.scalers = {}
        self.is_normalized = False
        
        self._clean_nan_values()
        
        # Calculate class weights for mortality
        mortality_counts = self.df[config.LABEL_MORTALITY].value_counts()
        total = len(self.df)
        # Calculate balanced class weights
        weight_for_0 = total / (2.0 * mortality_counts[0])
        weight_for_1 = total / (2.0 * mortality_counts[1])
        config.CLASS_WEIGHTS['mortality'] = torch.tensor([weight_for_0, weight_for_1]).to(config.DEVICE)
        
        # Print dataset statistics
        self.print_statistics()
    
    def print_statistics(self):
        # Basic statistics
        total = len(self.df)
        print(f"\n=== Dataset Statistics ===")
        print(f"Total samples: {total}")
        print(f"Structured data: {self.df['has_struct'].sum()} ({self.df['has_struct'].mean()*100:.1f}%)")
        print(f"CXR Image available: {self.df['has_img'].sum()} ({self.df['has_img'].mean()*100:.1f}%)")
        print(f"Text Notes available: {self.df['has_text'].sum()} ({self.df['has_text'].mean()*100:.1f}%)")
        print(f"Radiology Reports available: {self.df['has_rad'].sum()} ({self.df['has_rad'].mean()*100:.1f}%)")
        
        # Target variable distribution
        print(f"\n=== Target Variables ===")
        print(f"30-day Readmission rate: {self.df[config.LABEL_READM].mean()*100:.2f}%")
        print(f"ICU Admission rate: {self.df[config.LABEL_ICU].mean()*100:.2f}%")
        print(f"In-Hospital Mortality rate: {self.df[config.LABEL_MORTALITY].mean()*100:.2f}%")
        
        # Missing pattern analysis
        print("\n=== Missing Pattern Distribution ===")
        pattern_counts = self.df['missing_pattern'].value_counts().sort_index()
        for pattern, count in pattern_counts.items():
            pattern_bin = format(int(pattern), '04b')
            pattern_desc = f"Pattern {pattern} ({pattern_bin}): [S={pattern_bin[0]}, I={pattern_bin[1]}, T={pattern_bin[2]}, R={pattern_bin[3]}]"
            print(f"{pattern_desc}: {count} samples ({count/total*100:.1f}%)")
    
    def __len__(self):
        return len(self.df)
        
    def _process_struct(self, row):
        """Process structured data - UPDATED TO REMOVE LEAKING FEATURES"""
        features = []
        
        # Exclude subject_id, labels, missing flags AND leaking features
        exclude_cols = ['subject_id', config.LABEL_READM, config.LABEL_ICU, config.LABEL_MORTALITY,
                        'has_struct', 'has_img', 'has_text', 'has_rad', 'missing_pattern']
        
        # CRITICAL: Add leaking features to exclude list
        leaking_features = ['idx_discharge_location', 'idx_severity_mode']
        exclude_cols.extend(leaking_features)
        
        for col in self.df.columns:
            if col not in exclude_cols:
                val = row[col]
                
                # Apply scaling for non-zero numerical values if normalized
                if self.is_normalized and col in self.scalers and pd.to_numeric(val, errors='coerce') != 0:
                    try:
                        val = float(val)
                        val = self.scalers[col].transform([[val]])[0, 0]
                    except:
                        val = 0.0
                else:
                    # Convert to float, handling non-numeric values
                    try:
                        val = float(val)
                    except:
                        val = 0.0
                
                features.append(val)
        
        return np.array(features, dtype=np.float32)
    
    def _parse_embedding(self, embedding_str, dim):
        """Parse embedding string to numpy array, handling exceptions"""
        if isinstance(embedding_str, str) and embedding_str.strip():
            try:
                # Clean up potentially malformed JSON
                clean_str = embedding_str.strip()
                if not (clean_str.startswith('[') and clean_str.endswith(']')):
                    clean_str = '[' + clean_str.strip('[]') + ']'
                
                arr = np.array(json.loads(clean_str), dtype=np.float32)
                
                # Replace NaNs with zeros
                arr = np.nan_to_num(arr, nan=0.0)
                
                # Ensure correct dimensionality
                if arr.size != dim:
                    if arr.size > dim:
                        arr = arr[:dim]  # Truncate
                    else:
                        # Pad with zeros
                        temp = np.zeros(dim, dtype=np.float32)
                        temp[:arr.size] = arr
                        arr = temp
                
                # Normalize
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                
                return arr
            except Exception as e:
                print(f"Error parsing embedding: {str(e)[:100]}")
        
        # Default: return zero vector
        return np.zeros(dim, dtype=np.float32)
        
    def __getitem__(self, idx):
        # Get row data
        row = self.df.iloc[idx]
        subject_id = row['subject_id']
        
        # Process each modality
        # Structured data
        x_struct = torch.tensor(self._process_struct(row))
        
        # Image embedding
        if row['has_img'] == 1 and subject_id in self.img_df.index:
            img_embedding = self.img_df.loc[subject_id, config.IMG_EMB_COL]
            x_img = torch.tensor(self._parse_embedding(img_embedding, config.IMG_EMB_DIM))
        else:
            x_img = torch.zeros(config.IMG_EMB_DIM, dtype=torch.float32)
        
        # Text embedding
        if row['has_text'] == 1 and subject_id in self.text_df.index:
            text_embedding = self.text_df.loc[subject_id, config.TEXT_EMB_COL]
            x_text = torch.tensor(self._parse_embedding(text_embedding, config.TEXT_EMB_DIM))
        else:
            x_text = torch.zeros(config.TEXT_EMB_DIM, dtype=torch.float32)
        
        # Radiology embedding
        if row['has_rad'] == 1 and subject_id in self.text_df.index:
            rad_embedding = self.text_df.loc[subject_id, config.RAD_EMB_COL]
            x_rad = torch.tensor(self._parse_embedding(rad_embedding, config.RAD_EMB_DIM))
        else:
            x_rad = torch.zeros(config.RAD_EMB_DIM, dtype=torch.float32)
        
        # Target variables
        y_readm = torch.tensor([float(row[config.LABEL_READM])], dtype=torch.float32)
        y_icu = torch.tensor([float(row[config.LABEL_ICU])], dtype=torch.float32)
        y_mortality = torch.tensor([float(row[config.LABEL_MORTALITY])], dtype=torch.float32)
        
        # Missing flags
        missing_flags = torch.tensor([
            float(row['has_struct']),
            float(row['has_img']),
            float(row['has_text']),
            float(row['has_rad'])
        ], dtype=torch.float32)
        
        # Missing pattern (for classification auxiliary task)
        missing_pattern = torch.tensor(row['missing_pattern'], dtype=torch.long)
        
        return {
            'subject_id': subject_id,
            'x_struct': x_struct,
            'x_img': x_img,
            'x_text': x_text,
            'x_rad': x_rad,
            'y_readm': y_readm,
            'y_icu': y_icu,
            'y_mortality': y_mortality,
            'missing_flags': missing_flags,
            'missing_pattern': missing_pattern,
            'idx': idx
        }

    def normalize_with_training_stats(self, train_indices=None):
        """Normalize all data using training set statistics"""
        if not self.normalize or self.is_normalized:
            return
        
        if train_indices is None:
            print("Warning: No training indices provided for normalization. Using all data.")
            train_indices = list(range(len(self.df)))
        
        print(f"Normalizing data using statistics from {len(train_indices)} training samples...")
        
        # Get training set data to calculate statistics
        train_df = self.df.iloc[train_indices]
        
        # Only normalize numerical features, excluding target variables and IDs
        leaking_features = ['idx_discharge_location', 'idx_severity_mode']
        exclude_list = [config.LABEL_READM, config.LABEL_ICU, config.LABEL_MORTALITY, 
                        'subject_id', 'missing_pattern'] + leaking_features
        
        num_features = train_df.select_dtypes(include=["number"]).columns.drop(
            exclude_list, errors='ignore'
        )
        
        # Calculate scalers using training data
        for c in num_features:
            vals = train_df[c].replace(0, np.nan).dropna().values.reshape(-1, 1)
            if len(vals) > 0:
                if config.NORMALIZATION == "robust":
                    sc = RobustScaler()
                elif config.NORMALIZATION == "minmax":
                    sc = MinMaxScaler()
                elif config.NORMALIZATION == "power":
                    sc = PowerTransformer(method='yeo-johnson')
                else:  # default to standard
                    sc = StandardScaler()
                
                sc.fit(vals)
                self.scalers[c] = sc
        
        # Mark as normalized
        self.is_normalized = True
        
        print(f"Normalization complete using {len(self.scalers)} scalers.")

#########################################
# 10. Self-Supervised Pretraining Function
#########################################
def self_supervised_pretraining(dataset, train_indices, device):
    """Self-supervised pretraining phase"""
    print("\n" + "="*70)
    print("SELF-SUPERVISED PRETRAINING PHASE")
    print("="*70)
    
    # Create SSL model
    ssl_model = EnhancedMMNARModelWithSSL(ssl_mode=True).to(device)
    
    # SSL optimizer (different from main training)
    ssl_optimizer = optim.AdamW(
        ssl_model.parameters(),
        lr=config.PRETRAIN_LR,
        weight_decay=config.PRETRAIN_WEIGHT_DECAY
    )
    
    # SSL loss function
    ssl_loss_fn = SelfSupervisedLoss(
        reconstruction_weight=1.0,
        pattern_weight=0.5,
        contrastive_weight=0.3
    )
    
    # SSL data loader
    ssl_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Pretraining for {config.PRETRAIN_EPOCHS} epochs on {len(train_indices)} samples")
    
    # Pretraining loop
    for epoch in range(1, config.PRETRAIN_EPOCHS + 1):
        ssl_model.train()
        
        epoch_ssl_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_pattern_loss = 0.0
        epoch_contrastive_loss = 0.0
        
        progress_bar = tqdm(ssl_loader, desc=f"SSL Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            x_struct = batch['x_struct'].to(device)
            x_img = batch['x_img'].to(device)
            x_text = batch['x_text'].to(device)
            x_rad = batch['x_rad'].to(device)
            missing_flags = batch['missing_flags'].to(device)
            missing_pattern = batch['missing_pattern'].to(device)
            
            # Clear gradients
            ssl_optimizer.zero_grad()
            
            # Forward pass (SSL mode)
            ssl_outputs = ssl_model(x_struct, x_img, x_text, x_rad, missing_flags)
            
            # Original modality features for loss calculation
            original_modalities = [x_struct, x_img, x_text, x_rad]
            
            # Calculate SSL loss
            ssl_loss, ssl_loss_dict = ssl_loss_fn(
                ssl_outputs, original_modalities, missing_flags, missing_pattern
            )
            
            # Check for NaN loss
            if torch.isnan(ssl_loss).any():
                print(f"WARNING: NaN SSL loss detected in batch {batch_idx}, skipping")
                continue
            
            # Backward and optimize
            ssl_loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
            ssl_optimizer.step()
            
            # Accumulate losses
            batch_size = x_struct.size(0)
            epoch_ssl_loss += ssl_loss.item() * batch_size
            epoch_recon_loss += ssl_loss_dict['reconstruction'].item() * batch_size
            epoch_pattern_loss += ssl_loss_dict['pattern'].item() * batch_size
            epoch_contrastive_loss += ssl_loss_dict['contrastive'].item() * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'SSL_Loss': ssl_loss.item(),
                'Recon': ssl_loss_dict['reconstruction'].item(),
                'Pattern': ssl_loss_dict['pattern'].item(),
                'Contrast': ssl_loss_dict['contrastive'].item()
            })
        
        # Calculate average losses
        n_samples = len(train_indices)
        avg_ssl_loss = epoch_ssl_loss / n_samples
        avg_recon_loss = epoch_recon_loss / n_samples
        avg_pattern_loss = epoch_pattern_loss / n_samples
        avg_contrastive_loss = epoch_contrastive_loss / n_samples
        
        print(f"\nSSL Epoch {epoch}/{config.PRETRAIN_EPOCHS}:")
        print(f"  Total Loss: {avg_ssl_loss:.4f}")
        print(f"  Reconstruction: {avg_recon_loss:.4f}")
        print(f"  Pattern: {avg_pattern_loss:.4f}")
        print(f"  Contrastive: {avg_contrastive_loss:.4f}")
    
    # Save pretrained weights
    pretrained_path = os.path.join(config.MODEL_DIR, 'ssl_pretrained_weights.pt')
    torch.save(ssl_model.state_dict(), pretrained_path)
    print(f"\nPretrained weights saved to {pretrained_path}")
    
    return ssl_model

#########################################
# 11. Enhanced Training Functions
#########################################
def train_epoch_enhanced_with_curriculum(model, dataset, train_indices, optimizer, device, 
                                        losses, epoch, curriculum_scheduler, advanced_sampler):
    """Enhanced training epoch with curriculum learning and adaptive sampling"""
    model.train()
    
    # Get curriculum settings
    curriculum_indices = curriculum_scheduler.get_curriculum_indices(epoch)
    task_weights = curriculum_scheduler.get_adaptive_task_weights(epoch)
    active_tasks = curriculum_scheduler.get_active_tasks(epoch)
    stage = curriculum_scheduler.get_curriculum_stage(epoch)
    
    print(f"\nEpoch {epoch}: Stage={stage}, Samples={len(curriculum_indices)}, Active_tasks={active_tasks}")
    print(f"Task weights: {task_weights}")
    
    # Create curriculum-aware sampler
    if len(curriculum_indices) < len(train_indices):
        # Use curriculum filtering
        allowed_patterns = curriculum_scheduler.get_pattern_difficulty_filter(epoch)
        curriculum_sampler = advanced_sampler.create_curriculum_aware_sampler(
            allowed_patterns=allowed_patterns,
            focus_task='readmission'  # Focus on hardest task
        )
        current_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=curriculum_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        # Use all data with advanced sampling
        weighted_sampler = advanced_sampler.create_curriculum_aware_sampler(
            focus_task='readmission'
        )
        current_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            sampler=weighted_sampler,
            num_workers=4,
            pin_memory=True
        )
    
    # Initialize metrics
    epoch_loss = 0.0
    epoch_readm_loss = 0.0
    epoch_icu_loss = 0.0
    epoch_mortality_loss = 0.0
    
    # Unpack loss functions
    compound_loss_fn = losses['compound_loss']
    
    # Progress bar
    progress_bar = tqdm(current_loader, desc=f"Epoch {epoch} ({stage})")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Extract batch data
        x_struct = batch['x_struct'].to(device)
        x_img = batch['x_img'].to(device)
        x_text = batch['x_text'].to(device)
        x_rad = batch['x_rad'].to(device)
        y_readm = batch['y_readm'].to(device)
        y_icu = batch['y_icu'].to(device)
        y_mortality = batch['y_mortality'].to(device)
        missing_flags = batch['missing_flags'].to(device)
        missing_pattern = batch['missing_pattern'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_struct, x_img, x_text, x_rad, missing_flags)
        
        # Prepare targets
        targets = {
            'y_readm': y_readm,
            'y_icu': y_icu,
            'y_mortality': y_mortality,
            'missing_flags': missing_flags,
            'missing_pattern': missing_pattern
        }
        
        # Calculate epoch ratio for focal loss
        epoch_ratio = epoch / config.NUM_EPOCHS
        
        # Calculate compound loss with curriculum weights
        total_loss, loss_metrics = compound_loss_fn(
            outputs, targets, epoch_ratio=epoch_ratio, task_weights=task_weights
        )
        
        # Check for NaN loss
        if torch.isnan(total_loss).any():
            print(f"WARNING: NaN loss detected in batch {batch_idx}, skipping")
            continue
            
        # Backward and optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        batch_size = x_struct.size(0)
        epoch_loss += total_loss.item() * batch_size
        epoch_readm_loss += loss_metrics['readm_loss'] * batch_size
        epoch_icu_loss += loss_metrics['icu_loss'] * batch_size
        epoch_mortality_loss += loss_metrics['mortality_loss'] * batch_size
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss.item(),
            'readm': loss_metrics['readm_loss'],
            'icu': loss_metrics['icu_loss'],
            'mort': loss_metrics['mortality_loss'],
            'stage': stage
        })
    
    # Calculate average losses
    n_samples = len(curriculum_indices) if len(curriculum_indices) < len(train_indices) else len(train_indices)
    return {
        'loss': epoch_loss / n_samples,
        'readm_loss': epoch_readm_loss / n_samples,
        'icu_loss': epoch_icu_loss / n_samples,
        'mortality_loss': epoch_mortality_loss / n_samples,
        'samples_used': n_samples,
        'stage': stage,
        'task_weights': task_weights
    }

def evaluate_with_tta(model, loader, device, tta_augmenter=None):
    """Enhanced evaluation with test time augmentation"""
    model.eval()
    
    # Predictions and targets
    all_readm_preds = []
    all_readm_targets = []
    all_icu_preds = []
    all_icu_targets = []
    all_mortality_preds = []
    all_mortality_targets = []
    all_missing_flags = []
    all_subject_ids = []
    
    # Loss accumulators
    total_readm_loss = 0.0
    total_icu_loss = 0.0
    total_mortality_loss = 0.0
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mortality_loss_fn = nn.BCEWithLogitsLoss(pos_weight=config.CLASS_WEIGHTS['mortality'][1])
    
    # Use TTA if enabled and augmenter provided
    use_tta = config.TTA_ENABLED and tta_augmenter is not None
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating" + (" (with TTA)" if use_tta else "")):
            # Extract batch data
            x_struct = batch['x_struct']
            x_img = batch['x_img']
            x_text = batch['x_text']
            x_rad = batch['x_rad']
            y_readm = batch['y_readm'].to(device)
            y_icu = batch['y_icu'].to(device)
            y_mortality = batch['y_mortality'].to(device)
            missing_flags = batch['missing_flags']
            subject_ids = batch['subject_id']
            
            # Make predictions (with or without TTA)
            if use_tta:
                outputs = tta_augmenter.predict_with_tta(model, batch, device)
            else:
                outputs = model(
                    x_struct.to(device), x_img.to(device), 
                    x_text.to(device), x_rad.to(device),
                    missing_flags.to(device)
                )
            
            # Calculate losses
            readm_loss = bce_loss(outputs['readmission'], y_readm)
            icu_loss = bce_loss(outputs['icu'], y_icu)
            mortality_loss = mortality_loss_fn(outputs['mortality'], y_mortality)
            
            # Accumulate losses
            batch_size = x_struct.size(0)
            total_readm_loss += readm_loss.item() * batch_size
            total_icu_loss += icu_loss.item() * batch_size
            total_mortality_loss += mortality_loss.item() * batch_size
            
            # Get predictions
            readm_preds = torch.sigmoid(outputs['readmission']).cpu().numpy()
            icu_preds = torch.sigmoid(outputs['icu']).cpu().numpy()
            mortality_preds = torch.sigmoid(outputs['mortality']).cpu().numpy()
            
            # Collect predictions and targets
            all_readm_preds.extend(readm_preds.flatten().tolist())
            all_readm_targets.extend(y_readm.cpu().numpy().flatten().tolist())
            all_icu_preds.extend(icu_preds.flatten().tolist())
            all_icu_targets.extend(y_icu.cpu().numpy().flatten().tolist())
            all_mortality_preds.extend(mortality_preds.flatten().tolist())
            all_mortality_targets.extend(y_mortality.cpu().numpy().flatten().tolist())
            all_missing_flags.extend(missing_flags.cpu().numpy().tolist())
            all_subject_ids.extend(subject_ids)
    
    # Convert to numpy arrays for metric calculation
    readm_targets = np.array(all_readm_targets)
    readm_preds = np.array(all_readm_preds)
    icu_targets = np.array(all_icu_targets)
    icu_preds = np.array(all_icu_preds)
    mortality_targets = np.array(all_mortality_targets)
    mortality_preds = np.array(all_mortality_preds)
    
    # Calculate metrics
    readm_auc = roc_auc_score(readm_targets, readm_preds)
    icu_auc = roc_auc_score(icu_targets, icu_preds)
    mortality_auc = roc_auc_score(mortality_targets, mortality_preds)
    
    readm_apr = average_precision_score(readm_targets, readm_preds)
    icu_apr = average_precision_score(icu_targets, icu_preds)
    mortality_apr = average_precision_score(mortality_targets, mortality_preds)
    
    readm_brier = brier_score_loss(readm_targets, readm_preds)
    icu_brier = brier_score_loss(icu_targets, icu_preds)
    mortality_brier = brier_score_loss(mortality_targets, mortality_preds)
    
    # Average losses
    n_samples = len(loader.dataset)
    avg_readm_loss = total_readm_loss / n_samples
    avg_icu_loss = total_icu_loss / n_samples
    avg_mortality_loss = total_mortality_loss / n_samples
    
    # Compile results
    results = {
        'readm_loss': avg_readm_loss,
        'icu_loss': avg_icu_loss,
        'mortality_loss': avg_mortality_loss,
        'readm_auc': readm_auc,
        'icu_auc': icu_auc,
        'mortality_auc': mortality_auc,
        'readm_apr': readm_apr,
        'icu_apr': icu_apr,
        'mortality_apr': mortality_apr,
        'readm_brier': readm_brier,
        'icu_brier': icu_brier,
        'mortality_brier': mortality_brier
    }
    
    # Prepare predictions for further analysis
    predictions = {
        'subject_id': all_subject_ids,
        'readm_preds': all_readm_preds,
        'readm_targets': all_readm_targets,
        'icu_preds': all_icu_preds,
        'icu_targets': all_icu_targets,
        'mortality_preds': all_mortality_preds,
        'mortality_targets': all_mortality_targets,
        'missing_flags': all_missing_flags
    }
    
    return results, predictions

#########################################
# 12. Main Training Function
#########################################
def train_and_evaluate_enhanced_complete(train_loader, val_loader, test_loader=None):
    """Complete enhanced training with all improvements"""
    print("="*80)
    print("ENHANCED MMNAR MODEL WITH ALL IMPROVEMENTS")
    print("Features: SSL Pretraining + Focal Loss + Curriculum Learning + TTA")
    print("="*80)
    
    # Extract dataset and indices
    dataset = train_loader.dataset
    train_indices = train_loader.sampler.indices
    
    # Step 1: Self-supervised pretraining (if enabled)
    if config.ENABLE_PRETRAINING:
        ssl_model = self_supervised_pretraining(dataset, train_indices, config.DEVICE)
        pretrained_weights = ssl_model.state_dict()
    else:
        print("Self-supervised pretraining disabled, proceeding to main training...")
        pretrained_weights = None
    
    # Step 2: Initialize main model
    print("\nInitializing main model for downstream tasks...")
    model = EnhancedMMNARModelWithSSL(ssl_mode=False).to(config.DEVICE)
    
    # Load pretrained weights if available
    if pretrained_weights is not None:
        print("Loading pretrained weights...")
        # Filter out task-specific heads that don't exist in SSL model
        filtered_weights = {k: v for k, v in pretrained_weights.items() 
                          if k in model.state_dict() and 
                          not any(head in k for head in ['readm_head', 'icu_head', 'mortality_head'])}
        model.load_state_dict(filtered_weights, strict=False)
        print(f"Loaded {len(filtered_weights)} pretrained weight tensors")
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Step 3: Initialize enhanced components
    
    # Enhanced loss functions
    losses = {
        'compound_loss': EnhancedCompoundLoss()
    }
    
    # Enhanced curriculum scheduler
    curriculum_scheduler = EnhancedCurriculumScheduler(dataset, train_indices, config.NUM_EPOCHS)
    
    # Advanced sampler
    advanced_sampler = AdvancedSampler(dataset, train_indices)
    
    # Test time augmentation
    tta_augmenter = TestTimeAugmentation(
        num_augments=config.TTA_SAMPLES,
        noise_level=config.TTA_NOISE_LEVEL
    )
    
    # Step 4: Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Use OneCycleLR instead of ReduceLROnPlateau
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE * 2,
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Step 5: Training tracking
    best_val_score = 0.0
    best_model_state = None
    no_improvement = 0
    
    print("\n" + "="*70)
    print("STARTING ENHANCED TRAINING WITH CURRICULUM LEARNING")
    print("="*70)
    
    # Step 6: Main training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Enhanced training with curriculum
        train_metrics = train_epoch_enhanced_with_curriculum(
            model, dataset, train_indices, optimizer, config.DEVICE,
            losses, epoch, curriculum_scheduler, advanced_sampler
        )
        
        # Update scheduler
        scheduler.step()
        
        # Evaluate on validation set (with TTA if enabled)
        val_metrics, _ = evaluate_with_tta(model, val_loader, config.DEVICE, tta_augmenter)
        
        # Update curriculum scheduler with performance
        curriculum_scheduler.update_performance_history(val_metrics)
        
        # Calculate combined validation score (weighted by task importance)
        val_score = (0.45 * val_metrics['readm_auc'] + 
                    0.35 * val_metrics['mortality_auc'] + 
                    0.20 * val_metrics['icu_auc'])
        
        # Print detailed progress
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} - Stage: {train_metrics['stage']}")
        print(f"Training: Loss={train_metrics['loss']:.4f}, Samples={train_metrics['samples_used']}")
        print(f"Task Weights: R={train_metrics['task_weights']['readmission']:.2f}, " +
              f"I={train_metrics['task_weights']['icu']:.2f}, M={train_metrics['task_weights']['mortality']:.2f}")
        print(f"Validation Metrics:")
        print(f"  Readmission - AUC: {val_metrics['readm_auc']:.4f}, APR: {val_metrics['readm_apr']:.4f}")
        print(f"  ICU         - AUC: {val_metrics['icu_auc']:.4f}, APR: {val_metrics['icu_apr']:.4f}")
        print(f"  Mortality   - AUC: {val_metrics['mortality_auc']:.4f}, APR: {val_metrics['mortality_apr']:.4f}")
        print(f"Combined Score: {val_score:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Check for improvement
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = model.state_dict()
            no_improvement = 0
            
            # Save best model
            torch.save(best_model_state, 
                      os.path.join(config.MODEL_DIR, 'best_enhanced_complete_model.pt'))
            
            print(f"New best model saved! (Score: {val_score:.4f})")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs (best: {best_val_score:.4f})")
            
            # Early stopping
            if no_improvement >= config.EARLY_STOPPING:
                print(f"Early stopping after {epoch} epochs")
                break
    
    # Step 7: Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'best_enhanced_complete_model.pt')))
    
    # Step 8: Final evaluation on test set
    if test_loader is not None:
        print("\n" + "="*80)
        print("FINAL EVALUATION ON TEST SET")
        print("="*80)
        
        test_metrics, test_predictions = evaluate_with_tta(
            model, test_loader, config.DEVICE, tta_augmenter
        )
        
        print(f"\n{'='*80}")
        print("FINAL ENHANCED MMNAR MODEL RESULTS")
        print(f"{'='*80}")
        print(f"{'Task':<15} {'AUC':<10} {'APR':<10} {'Brier':<10}")
        print("-"*50)
        print(f"{'Readmission':<15} {test_metrics['readm_auc']:<10.4f} {test_metrics['readm_apr']:<10.4f} {test_metrics['readm_brier']:<10.4f}")
        print(f"{'ICU':<15} {test_metrics['icu_auc']:<10.4f} {test_metrics['icu_apr']:<10.4f} {test_metrics['icu_brier']:<10.4f}")
        print(f"{'Mortality':<15} {test_metrics['mortality_auc']:<10.4f} {test_metrics['mortality_apr']:<10.4f} {test_metrics['mortality_brier']:<10.4f}")
        print(f"{'='*80}")
        
        # Save final results
        with open(os.path.join(config.RESULTS_DIR, 'enhanced_complete_test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        return model, test_metrics, test_predictions
    
    return model, None, None

#########################################
# 13. Visualization Functions (Enhanced)
#########################################
def save_predictions_enhanced(predictions, save_path):
    """Save enhanced predictions with additional analysis"""
    # Create DataFrame
    df = pd.DataFrame({
        'subject_id': predictions['subject_id'],
        'readmission_target': predictions['readm_targets'],
        'readmission_prediction': predictions['readm_preds'],
        'icu_target': predictions['icu_targets'],
        'icu_prediction': predictions['icu_preds'],
        'mortality_target': predictions['mortality_targets'],
        'mortality_prediction': predictions['mortality_preds']
    })
    
    # Add missing flags
    missing_flags = np.array(predictions['missing_flags'])
    df['has_struct'] = missing_flags[:, 0]
    df['has_img'] = missing_flags[:, 1]
    df['has_text'] = missing_flags[:, 2]
    df['has_rad'] = missing_flags[:, 3]
    
    # Add modality count and pattern
    df['modality_count'] = df[['has_struct', 'has_img', 'has_text', 'has_rad']].sum(axis=1)
    df['missing_pattern'] = df.apply(
        lambda row: f"{int(row['has_struct'])}{int(row['has_img'])}{int(row['has_text'])}{int(row['has_rad'])}", 
        axis=1
    )
    
    # Add prediction errors and calibration
    df['readm_abs_error'] = np.abs(df['readmission_prediction'] - df['readmission_target'])
    df['icu_abs_error'] = np.abs(df['icu_prediction'] - df['icu_target'])
    df['mortality_abs_error'] = np.abs(df['mortality_prediction'] - df['mortality_target'])
    
    # Prediction correctness (using 0.5 threshold)
    df['readm_correct'] = ((df['readmission_prediction'] >= 0.5) == (df['readmission_target'] == 1)).astype(int)
    df['icu_correct'] = ((df['icu_prediction'] >= 0.5) == (df['icu_target'] == 1)).astype(int)
    df['mortality_correct'] = ((df['mortality_prediction'] >= 0.5) == (df['mortality_target'] == 1)).astype(int)
    
    # Confidence levels
    df['readm_confidence'] = np.abs(df['readmission_prediction'] - 0.5) * 2  # 0 to 1
    df['icu_confidence'] = np.abs(df['icu_prediction'] - 0.5) * 2
    df['mortality_confidence'] = np.abs(df['mortality_prediction'] - 0.5) * 2
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    # Print summary statistics
    print(f"\n=== Enhanced Prediction Analysis ===")
    print(f"Total predictions: {len(df)}")
    print(f"Accuracy by task:")
    print(f"  Readmission: {df['readm_correct'].mean():.3f}")
    print(f"  ICU: {df['icu_correct'].mean():.3f}")
    print(f"  Mortality: {df['mortality_correct'].mean():.3f}")
    
    print(f"\nAccuracy by modality count:")
    for count in sorted(df['modality_count'].unique()):
        subset = df[df['modality_count'] == count]
        print(f"  {count} modalities ({len(subset)} samples):")
        print(f"    Readmission: {subset['readm_correct'].mean():.3f}")
        print(f"    ICU: {subset['icu_correct'].mean():.3f}")
        print(f"    Mortality: {subset['mortality_correct'].mean():.3f}")
    
    return df

def plot_enhanced_curves(targets, preds, task_name, save_path, confidence=None):
    """Enhanced curve plotting with confidence intervals"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, preds)
    auc_score = roc_auc_score(targets, preds)
    
    ax1.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC (AUC = {auc_score:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'ROC Curve - {task_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(targets, preds)
    ap_score = average_precision_score(targets, preds)
    
    ax2.plot(recall, precision, 'r-', linewidth=2.5, label=f'PR (AP = {ap_score:.3f})')
    
    # Add baseline
    prevalence = np.mean(targets)
    ax2.axhline(y=prevalence, color='k', linestyle='--', alpha=0.3, linewidth=1.5, 
               label=f'Baseline (Prevalence = {prevalence:.3f})')
    
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'Precision-Recall Curve - {task_name}', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

#########################################
# 14. Main Function
#########################################
def main():
    """Main function with all enhancements"""
    print(f"Using device: {config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*80)
    print("ENHANCED MMNAR MULTIMODAL MODEL")
    print("Self-Supervised + Focal Loss + Curriculum + TTA")
    print("="*80)
    
    # Create output directories
    for directory in [config.OUTPUT_DIR, config.FIGURE_DIR, config.MODEL_DIR, config.RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MultiModalDataset(
        struct_path=config.FEATURES_PATH,
        img_path=config.CXR_PATH,  
        text_path=config.TEXT_PATH,
        normalize=True,
        train_mode=True
    )
    
    # Enhanced train/val/test split
    print("\nPerforming enhanced stratified split...")
    indices = list(range(len(dataset)))
    
    # Multi-criteria stratification
    y_mortality = np.array([dataset[i]['y_mortality'].item() for i in range(len(dataset))])
    missing_patterns = np.array([dataset[i]['missing_pattern'].item() for i in range(len(dataset))])
    y_readm = np.array([dataset[i]['y_readm'].item() for i in range(len(dataset))])
    
    # Create combined stratification variable
    y_combined = (y_mortality * 4 + y_readm * 2 + (missing_patterns == 15).astype(int))
    
    train_idx, test_idx = sklearn.model_selection.train_test_split(
        indices, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y_combined
    )
    
    y_train_combined = y_combined[train_idx]
    train_idx, val_idx = sklearn.model_selection.train_test_split(  
        train_idx, test_size=0.15, random_state=config.RANDOM_SEED, stratify=y_train_combined
    )
    
    print(f"Enhanced split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Print enhanced distribution analysis
    print("\nEnhanced distribution analysis:")
    for split_name, split_idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        mortality_rate = np.mean([dataset[i]['y_mortality'].item() for i in split_idx])
        readm_rate = np.mean([dataset[i]['y_readm'].item() for i in split_idx])
        icu_rate = np.mean([dataset[i]['y_icu'].item() for i in split_idx])
        complete_rate = np.mean([dataset[i]['missing_pattern'].item() == 15 for i in split_idx])
        print(f"{split_name}: Mortality={mortality_rate:.3f}, Readmission={readm_rate:.3f}, "
              f"ICU={icu_rate:.3f}, Complete={complete_rate:.3f}")
    
    # Normalize with enhanced statistics
    dataset.normalize_with_training_stats(train_indices=train_idx)
    
    # Create enhanced data loaders
    train_loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE,
        sampler=SubsetRandomSampler(test_idx),
        num_workers=4, pin_memory=True
    )
    
    # Run enhanced training pipeline
    print("\n" + "="*80)
    print("LAUNCHING ENHANCED TRAINING PIPELINE")
    print("="*80)
    
    _, test_metrics, test_predictions = train_and_evaluate_enhanced_complete(
        train_loader, val_loader, test_loader
    )
    
    # Enhanced result analysis and visualization
    if test_predictions:
        print("\n" + "="*50)
        print("SAVING ENHANCED RESULTS AND VISUALIZATIONS")
        print("="*50)
        
        # Save enhanced predictions
        pred_df = save_predictions_enhanced(
            test_predictions, 
            os.path.join(config.RESULTS_DIR, 'enhanced_complete_predictions.csv')
        )
        
        # Create enhanced visualizations
        tasks = [('Readmission', 'readm'), ('ICU', 'icu'), ('Mortality', 'mortality')]
        
        for task_name, task_key in tasks:
            # Enhanced ROC and PR curves
            plot_enhanced_curves(
                test_predictions[f'{task_key}_targets'],
                test_predictions[f'{task_key}_preds'],
                task_name,
                os.path.join(config.FIGURE_DIR, f'enhanced_complete_{task_key}_curves.png')
            )
        
        # Save configuration for reproducibility
        config_dict = {k: v for k, v in config.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        with open(os.path.join(config.RESULTS_DIR, 'enhanced_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
    
    print("\n" + "="*80)
    print("ENHANCED MMNAR MODEL TRAINING COMPLETED!")
    print("="*80)
    print(f"Results saved in: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
