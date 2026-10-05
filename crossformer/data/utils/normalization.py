"""Lightweight normalization types shared by training and inference."""

from enum import Enum


class NormalizationType(str, Enum):
    """Defines supported normalization schemes for action and proprio."""

    NORMAL = "normal"
    BOUNDS = "bounds"

