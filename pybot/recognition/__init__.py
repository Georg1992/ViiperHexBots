"""Mob recognition pipeline for ViiperHexBots."""

from pybot.recognition.capture import capture_region
from pybot.recognition.rules import (
    HUNT_OBJECT_RADIUS,
    HUNT_TRACK_LOST_LIMIT,
    DiscoveryDetection,
    MobTrack,
    ReconcileSummary,
    apply_attack_event,
    apply_track_observation,
    cluster_living_detections,
    detection_matches_existing,
    is_alive,
    is_track_lost,
    select_target_id,
)
from pybot.recognition.simple.detector import SimpleMobDetector, load_simple_config

__all__ = [
    "HUNT_OBJECT_RADIUS",
    "HUNT_TRACK_LOST_LIMIT",
    "DiscoveryDetection",
    "MobTrack",
    "ReconcileSummary",
    "SimpleMobDetector",
    "apply_attack_event",
    "apply_track_observation",
    "capture_region",
    "cluster_living_detections",
    "detection_matches_existing",
    "is_alive",
    "is_track_lost",
    "load_simple_config",
    "select_target_id",
]
