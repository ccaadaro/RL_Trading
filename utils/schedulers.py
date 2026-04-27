"""
Learning rate and entropy scheduling functions for RL model training.

This module provides various schedules for learning rate and entropy coefficient
decay during training, including cosine annealing and custom schedules.
"""

import numpy as np
from typing import Callable, Union


def cosine_decay_lr(progress_remaining: float,
                    base_lr: float = 5e-5,
                    min_lr: float = 1e-6,
                    plateau_frac: float = 0.15) -> float:
    """
    Cosine decay learning rate schedule with initial plateau.
    
    Args:
        progress_remaining: Training progress from 1.0 (start) to 0.0 (end)
        base_lr: Starting learning rate
        min_lr: Minimum learning rate
        plateau_frac: Fraction of training with constant LR at the beginning
        
    Returns:
        Current learning rate
    """
    if progress_remaining < plateau_frac:
        # Map 1→0 (start) to 0→1 for the cosine
        t = 1.0 - (progress_remaining - plateau_frac) / (1.0 - plateau_frac)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))
    return base_lr


def cosine_lr(progress: float, base: float = 3e-5, min_lr: float = 5e-6) -> float:
    """
    Simple cosine decay learning rate schedule.
    
    Args:
        progress: Training progress from 1.0 (start) to 0.0 (end)
        base: Starting learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Current learning rate
    """
    # progress = 1 → 0
    cos_inner = np.pi * (1.0 - progress)          # 0→π
    return min_lr + 0.5 * (base - min_lr) * (1 + np.cos(cos_inner))


def ent_schedule(progress: float) -> float:
    """
    Entropy coefficient schedule for exploration control.
    
    Higher values encourage more exploration, gradually decays to encourage
    exploitation later in training.
    
    Args:
        progress: Training progress from 1.0 (start) to 0.0 (end)
        
    Returns:
        Current entropy coefficient
    """
    # Higher starting entropy for stronger exploration
    start_ent = 0.1
    end_ent = 0.001
    
    # Slower decay with square root falloff
    decay = progress**0.5
    
    entropy_value = end_ent + (start_ent - end_ent) * decay
    
    # Add minimum entropy clipping to ensure sufficient exploration
    return max(entropy_value, 1e-3)


def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Create a function that returns a linear schedule.
    
    Args:
        initial_value: Initial parameter value
        final_value: Final parameter value
        
    Returns:
        Function that takes progress (1.0 to 0.0) and returns current value
    """
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    
    return func


def lr_schedule(progress: float) -> float:
    """
    Custom learning rate schedule with early high LR followed by decay.
    
    Args:
        progress: Training progress from 1.0 (start) to 0.0 (end)
        
    Returns:
        Current learning rate
    """
    base_lr = 5e-5
    if progress < 0.3:
        return base_lr * (1 + 2 * (0.3 - progress))  # High LR early
    else:
        return base_lr * (1 - (progress - 0.3) / 0.7)  # Decay LR later