# src/position_estimator.py
"""
Position estimator: Calculates relative distance and angle to detected objects.
Supports monocular (pinhole) and stereo methods with filtering.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class PositionEstimator:
    """Estimate relative position (distance, angle) using monocular or stereo vision."""
    
    def __init__(self, config: Dict):
        """
        Initialize position estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.camera_config = config['camera']
        self.position_config = config['position']
        
        # Camera parameters
        self.focal_length_px = self.camera_config['focal_length']
        self.frame_width = self.camera_config['resolution'][0]
        self.frame_height = self.camera_config['resolution'][1]
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Object parameters
        self.known_width_cm = config['detection'].get('known_width_cm', 10.0)
        
        # Filter parameters
        self.alpha = self.position_config.get('distance_filter_alpha', 0.8)
        self.filtered_distance = None
        self.filtered_angle = None
        
        # Measurement history
        self.max_history = self.position_config.get('history_length', 10)
        self.distance_history = deque(maxlen=self.max_history)
        self.angle_history = deque(maxlen=self.max_history)
        
        # Camera matrix (for stereo/projection)
        self.camera_matrix = self._build_camera_matrix()
        
        # Stereo configuration
        self.stereo_baseline_cm = self.config.get('stereo', {}).get('baseline_cm', 6.0)
        self.method = self.position_config.get('method', 'monocular')
        
        print(f"[PositionEstimator] Initialized ({self.method} mode, "
              f"f={self.focal_length_px}px, known_width={self.known_width_cm}cm)")
    
    def _build_camera_matrix(self) -> np.ndarray:
        """Build camera intrinsic matrix from config or defaults."""
        cfg = self.camera_config
        fx = cfg.get('fx', self.focal_length_px)
        fy = cfg.get('fy', self.focal_length_px)
        cx = cfg.get('cx', self.frame_center_x)
        cy = cfg.get('cy', self.frame_center_y)
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def estimate_position(self, object_width_px: Optional[float],
                          centroid: Optional[Tuple[int, int]],
                          disparity: Optional[float] = None) -> Tuple[float, float]:
        """
        Estimate distance and angle to target.
        
        Args:
            object_width_px: Width of detected object in pixels
            centroid: Object center (cx, cy) in pixels
            disparity: Stereo disparity (if available)
            
        Returns:
            distance_cm: Estimated distance in centimeters
            angle_deg: Relative horizontal angle in degrees (- = left, + = right)
        """
        if object_width_px is None or centroid is None or object_width_px <= 0:
            return self.filtered_distance or 555.0, self.filtered_angle or 0.0
        
        # Estimate distance
        if self.method == "stereo" and disparity is not None:
            distance_cm = self._distance_from_stereo(disparity)
        else:
            distance_cm = self._distance_from_width(object_width_px)
        
        # Estimate angle
        angle_deg = self._angle_from_centroid(centroid)
        
        # Validate and clamp
        min_dist = self.position_config.get('min_distance_cm', 5.0)
        max_dist = self.position_config.get('max_distance_cm', 500.0)
        distance_cm = np.clip(distance_cm, min_dist, max_dist)
        
        # Apply filter
        self._apply_filter(distance_cm, angle_deg)
        
        return self.filtered_distance, self.filtered_angle
    
    def _distance_from_width(self, object_width_px: float) -> float:
        """
        Monocular distance: D = (W_real * f) / w_px
        
        Based on pinhole camera model.
        """
        if object_width_px <= 0:
            return 999.0
        
        distance_cm = (self.known_width_cm * self.focal_length_px) / object_width_px
        return distance_cm
    
    def _distance_from_stereo(self, disparity_px: float) -> float:
        """
        Stereo distance: D = (f * baseline) / disparity
        
        Args:
            disparity_px: Disparity in pixels
        """
        if disparity_px <= 0:
            return 999.0
        
        distance_cm = (self.focal_length_px * self.stereo_baseline_cm) / disparity_px
        return distance_cm
    
    def _angle_from_centroid(self, centroid: Tuple[int, int]) -> float:
        """
        Calculate horizontal angle from frame center.
        
        angle = arctan((cx - center_x) / f)
        """
        cx, cy = centroid
        offset_x = cx - self.frame_center_x
        
        if self.focal_length_px > 0:
            angle_rad = np.arctan2(offset_x, self.focal_length_px)
            angle_deg = np.degrees(angle_rad)
        else:
            angle_deg = 0.0
        
        return angle_deg
    
    def _apply_filter(self, distance_cm: float, angle_deg: float):
        """
        Apply exponential moving average (EMA) filter for smooth estimates.
        
        filtered = alpha * filtered + (1 - alpha) * new_measurement
        """
        if self.filtered_distance is None:
            self.filtered_distance = distance_cm
            self.filtered_angle = angle_deg
        else:
            self.filtered_distance = (self.alpha * self.filtered_distance +
                                      (1 - self.alpha) * distance_cm)
            self.filtered_angle = (self.alpha * self.filtered_angle +
                                   (1 - self.alpha) * angle_deg)
        
        # Update history
        self.distance_history.append(self.filtered_distance)
        self.angle_history.append(self.filtered_angle)
    
    def get_median_estimate(self) -> Tuple[float, float]:
        """
        Get median of recent measurements for outlier robustness.
        
        Returns:
            (median_distance_cm, median_angle_deg)
        """
        if not self.distance_history:
            return self.filtered_distance or 555.0, self.filtered_angle or 0.0
        
        median_dist = np.median(list(self.distance_history))
        median_angle = np.median(list(self.angle_history))
        return float(median_dist), float(median_angle)
    
    def is_measurement_valid(self, distance_cm: float, angle_deg: float) -> bool:
        """Check if measurement falls within expected bounds."""
        min_dist = self.position_config.get('min_distance_cm', 5.0)
        max_dist = self.position_config.get('max_distance_cm', 500.0)
        max_angle = self.config.get('navigation', {}).get('max_approach_angle_deg', 45.0)
        
        return (min_dist <= distance_cm <= max_dist and
                abs(angle_deg) <= max_angle)
    
    def update_focal_length(self, new_focal: float):
        """Update focal length after calibration."""
        self.focal_length_px = new_focal
        self.camera_matrix[0, 0] = new_focal
        self.camera_matrix[1, 1] = new_focal
        print(f"[PositionEstimator] Focal length updated: {new_focal:.1f} px")
