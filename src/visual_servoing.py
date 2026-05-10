# src/visual_servoing.py
"""
Visual servoing controller: Generates velocity commands for autonomous approach.
Implements proportional control with configurable gains and thresholds.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum


class SystemMode(Enum):
    """System operational modes."""
    SCANNING = "scanning"
    TRACKING = "tracking"
    GRASPING = "grasping"
    ORIENTING = "orienting"
    HOLDING = "holding"
    ERROR = "error"


class VisualServoingController:
    """
    Closed-loop visual servoing controller.
    
    Generates linear and angular velocity commands based on position error
    relative to target. Implements proportional control with deadband.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize controller.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        nav_config = config['navigation']
        
        # Control gains (tunable)
        self.Kp_linear = nav_config.get('Kp_linear', 0.5)
        self.Kp_angular = nav_config.get('Kp_angular', 0.3)
        self.Ki_linear = nav_config.get('Ki_linear', 0.1)
        self.Kd_linear = nav_config.get('Kd_linear', 0.05)
        
        # Thresholds
        self.grasping_distance_cm = nav_config.get('grasping_distance_cm', 15.0)
        self.safety_margin_cm = nav_config.get('safety_margin_cm', 5.0)
        self.alignment_threshold_deg = nav_config.get('alignment_threshold_deg', 2.0)
        self.max_approach_speed = nav_config.get('approach_speed', 0.5)
        
        # PID error accumulation
        self.distance_error_integral = 0.0
        self.last_distance_error = 0.0
        self.max_integral = 10.0
        
        # State
        self.mode = SystemMode.SCANNING
        self.target_aligned = False
        self.at_grasp_distance = False
        self.target_lost_counter = 0
        self.max_target_lost = nav_config.get('max_frames_lost', 30)
        
        print(f"[VisualServoing] Initialized (grasp_dist={self.grasping_distance_cm}cm, "
              f"Kp_linear={self.Kp_linear}, Kp_angular={self.Kp_angular})")
    
    def compute_command(self, distance_cm: float, angle_deg: float,
                        target_detected: bool) -> Tuple[float, float, SystemMode]:
        """
        Compute velocity commands for approach.
        
        Args:
            distance_cm: Distance to target (cm)
            angle_deg: Relative horizontal angle (degrees)
            target_detected: Whether target is currently visible
            
        Returns:
            linear_velocity_mps: Forward velocity (m/s, positive = forward)
            angular_velocity_rps: Angular velocity (rad/s, positive = CCW from above)
            mode: Current system mode
        """
        # Safety default
        if not target_detected:
            self.target_lost_counter += 1
            if self.target_lost_counter > self.max_target_lost:
                self.mode = SystemMode.SCANNING
            return 0.0, 0.0, self.mode
        
        self.target_lost_counter = 0
        
        # Determine mode
        if distance_cm <= self.grasping_distance_cm:
            self.mode = SystemMode.GRASPING
        elif distance_cm > self.grasping_distance_cm + self.safety_margin_cm:
            self.mode = SystemMode.TRACKING
        else:
            self.mode = SystemMode.TRACKING  # In transition zone
        
        # --- Angular control (rotate to align) ---
        angular_vel = self._compute_angular_velocity(angle_deg)
        
        # --- Linear control (approach) ---
        linear_vel = self._compute_linear_velocity(distance_cm, angle_deg)
        
        self.at_grasp_distance = (distance_cm <=
                                  self.grasping_distance_cm + self.safety_margin_cm)
        
        return linear_vel, angular_vel, self.mode
    
    def _compute_angular_velocity(self, angle_deg: float) -> float:
        """
        Proportional angular control.
        
        Angular velocity proportional to error, with saturation.
        """
        if abs(angle_deg) <= self.alignment_threshold_deg:
            self.target_aligned = True
            return 0.0
        
        self.target_aligned = False
        
        # Proportional: omega = Kp_angular * error
        error_normalized = angle_deg / 30.0  # Normalize by max expected offset
        omega = self.Kp_angular * np.clip(error_normalized, -1.0, 1.0)
        
        return omega
    
    def _compute_linear_velocity(self, distance_cm: float, angle_deg: float) -> float:
        """
        Compute linear velocity based on distance error.
        
        Only approach when target is centered (aligned).
        Includes PID control for smooth deceleration.
        """
        # Don't approach unless aligned
        if not self.target_aligned:
            return 0.0
        
        # Distance error (positive = too far)
        distance_error_cm = distance_cm - self.grasping_distance_cm
        
        # PID terms
        P = self.Kp_linear * distance_error_cm / 100.0  # Convert cm -> m
        
        self.distance_error_integral += distance_error_cm * 0.1  # dt ≈ 0.1s
        self.distance_error_integral = np.clip(
            self.distance_error_integral, -self.max_integral, self.max_integral
        )
        I = self.Ki_linear * self.distance_error_integral / 100.0
        
        D = self.Kd_linear * (distance_error_cm - self.last_distance_error) / 100.0
        self.last_distance_error = distance_error_cm
        
        # Combine PID
        linear_vel = P + I + D
        
        # Clamp
        linear_vel = np.clip(linear_vel, 0.0, self.max_approach_speed)
        
        # Slow down near target
        if distance_cm < self.grasping_distance_cm * 2:
            linear_vel *= 0.5
        
        return linear_vel
    
    def should_grasp(self, distance_cm: float, angle_deg: float) -> bool:
        """Check if grasping conditions are met."""
        return (self.target_aligned and
                distance_cm <= self.grasping_distance_cm + 3.0 and
                abs(angle_deg) <= self.alignment_threshold_deg)
    
    def get_status_string(self) -> str:
        """Return human-readable status string."""
        status = {
            SystemMode.SCANNING: "SCANNING - Searching",
            SystemMode.TRACKING: "TRACKING - Approaching",
            SystemMode.GRASPING: "GRASPING - Final",
            SystemMode.ORIENTING: "ORIENTING - Star Track",
            SystemMode.HOLDING: "HOLDING - Station",
            SystemMode.ERROR: "ERROR - Check System",
        }
        return status.get(self.mode, "UNKNOWN")
    
    def get_telemetry(self, distance_cm: float, angle_deg: float,
                      linear_vel: float, angular_vel: float) -> Dict:
        """Package telemetry data for external transmission."""
        return {
            'mode': self.mode.value,
            'distance_cm': round(distance_cm, 1),
            'angle_deg': round(angle_deg, 1),
            'linear_vel_mps': round(linear_vel, 3),
            'angular_vel_rps': round(angular_vel, 3),
            'target_aligned': self.target_aligned,
            'at_grasp_distance': self.at_grasp_distance,
            'ready_to_grasp': self.should_grasp(distance_cm, angle_deg),
        }
