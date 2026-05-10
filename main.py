#!/usr/bin/env python3
"""
Space Object Detection, Relative Navigation & Grasping System

Main entry point. Integrates all subsystems:
  - Object detection (YOLO/Color/ORB)
  - Position estimation (distance & angle)
  - Visual servoing (approach control)
  - Star tracker (attitude determination)
  - Telemetry (serial/UART output)

Usage:
    python main.py --video fl3.mp4
    python main.py --video 0 --config config/params.yaml
"""

import cv2
import yaml
import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.object_detector import ObjectDetector
from src.position_estimator import PositionEstimator
from src.visual_servoing import VisualServoingController, SystemMode
from src.telemetry import TelemetryHandler


class SpaceGraspingSystem:
    """
    Integrated space object detection, relative navigation & grasping system.
    
    Combines:
    - Multi-method object detection
    - Position estimation (distance & angle)
    - Visual servoing controller
    - Star tracker (optional)
    - Telemetry output
    """
    
    def __init__(self, config_path: str = "config/params.yaml"):
        """Initialize system from configuration file."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("  SPACE OBJECT DETECTION & GRASPING SYSTEM")
        print("=" * 60)
        
        # Initialize subsystems
        print("\n[INIT] Initializing subsystems...")
        
        self.detector = ObjectDetector(self.config)
        self.position_estimator = PositionEstimator(self.config)
        self.controller = VisualServoingController(self.config)
        
        telemetry_config = self.config.get('telemetry', {})
        self.telemetry = TelemetryHandler(
            output_format=telemetry_config.get('output_format', 'json'),
            serial_port=telemetry_config.get('serial_port', None)
        )
        
        # State variables
        self.distance_cm = 555.0
        self.angle_deg = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.mode = SystemMode.SCANNING
        self.target_detected = False
        self.bbox = None
        self.centroid = None
        
        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Visualization flags
        self.show_detection = True
        self.show_overlay = True
        self.show_fps = True
        
        # Output recording
        self.record = self.config.get('system', {}).get('save_output_video', False)
        self.video_writer = None
        
        print("[INIT] All subsystems ready")
        print(f"[INIT] Detection method: {self.config['detection']['method']}")
        print(f"[INIT] Position method: {self.config['position']['method']}")
        print(f"[INIT] Grasp distance: {self.controller.grasping_distance_cm} cm")
        print("=" * 60)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            display_frame: Annotated image for visualization
        """
        self.frame_count += 1
        display = frame.copy()
        
        # === Step 1: Object Detection ===
        self.target_detected, self.bbox, self.centroid, obj_width_px = \
            self.detector.detect(frame)
        
        # === Step 2: Position Estimation ===
        if self.target_detected:
            self.distance_cm, self.angle_deg = self.position_estimator.estimate_position(
                obj_width_px, self.centroid
            )
        else:
            # Use smoothed estimates during brief loss
            self.distance_cm, self.angle_deg = self.position_estimator.get_median_estimate()
        
        # === Step 3: Visual Servoing Control ===
        self.linear_vel, self.angular_vel, self.mode = self.controller.compute_command(
            self.distance_cm, self.angle_deg, self.target_detected
        )
        
        # === Step 4: Visualize ===
        # Detection overlay
        if self.show_detection:
            display = self.detector.draw_detection(
                display, self.bbox, self.centroid,
                self.distance_cm if self.target_detected else None,
                self.angle_deg if self.target_detected else None
            )
        
        # System overlay
        if self.show_overlay:
            display = self._draw_overlay(display)
        
        # === Step 5: Telemetry ===
        telem_data = self.controller.get_telemetry(
            self.distance_cm, self.angle_deg,
            self.linear_vel, self.angular_vel
        )
        self.telemetry.send(telem_data)
        
        # === Step 6: Grasp Check ===
        if self.controller.should_grasp(self.distance_cm, self.angle_deg):
            display = self._draw_grasp_indicator(display)
        
        return display
    
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw system status overlay."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Top bar background
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 105), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        y = 25
        
        # Mode and status
        status_str = self.controller.get_status_string()
        color = (0, 255, 0) if self.mode != SystemMode.ERROR else (0, 0, 255)
        cv2.putText(display, status_str, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        y += 25
        # Distance and angle
        if self.target_detected:
            cv2.putText(display, f"Dist: {self.distance_cm:.1f} cm | Ang: {self.angle_deg:.1f} deg",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(display, "No target detected", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 25
        # Velocity commands
        cv2.putText(display, f"Cmd: v={self.linear_vel:.2f} m/s | w={self.angular_vel:.2f} rad/s",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 25
        # FPS
        if self.show_fps:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(time.time() - self.start_time, 0.001))
            self.start_time = time.time()
            cv2.putText(display, f"FPS: {self.fps:.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    def _draw_grasp_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Draw grasping indicator overlay."""
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # Center text
        cv2.putText(display, "READY TO GRASP", (w//2 - 150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Border flash
        cv2.rectangle(display, (5, 5), (w-5, h-5), (0, 255, 0), 5)
        
        return display
    
    def run(self, video_source=0):
        """
        Main execution loop.
        
        Args:
            video_source: Camera index (0, 1, ...) or video file path
        """
        print(f"\n[RUN] Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {video_source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[RUN] Video: {width}x{height} @ {fps:.1f} FPS")
        
        # Initialize video writer if recording
        if self.record:
            output_path = self.config.get('system', {}).get('output_video_path',
                                                            'output/grasping_output.mp4')
            Path('output').mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps,
                                                (self.config['camera']['resolution'][0],
                                                 self.config['camera']['resolution'][1]))
            print(f"[RUN] Recording output to: {output_path}")
        
        print("\n[RUN] System active. Press 'q' to quit, 'd' to toggle detection")
        print("[RUN] Press 'o' to toggle overlay, 'f' to toggle FPS")
        print("-" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[RUN] End of video stream")
                break
            
            # Resize to config resolution
            target_w = self.config['camera']['resolution'][0]
            target_h = self.config['camera']['resolution'][1]
            frame = cv2.resize(frame, (target_w, target_h))
            
            # Process frame
            display = self.process_frame(frame)
            
            # Show output
            cv2.imshow("Space Grasping System", display)
            
            # Record if enabled
            if self.video_writer:
                self.video_writer.write(display)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[RUN] User requested shutdown")
                break
            elif key == ord('d'):
                self.show_detection = not self.show_detection
                print(f"[RUN] Detection visualization: {'ON' if self.show_detection else 'OFF'}")
            elif key == ord('o'):
                self.show_overlay = not self.show_overlay
                print(f"[RUN] Status overlay: {'ON' if self.show_overlay else 'OFF'}")
            elif key == ord('f'):
                self.show_fps = not self.show_fps
        
        # Cleanup
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.telemetry.close()
        print("[RUN] System shutdown complete")
    
    def run_with_star_tracker(self, video_source=0):
        """Run with star tracker integration (placeholder)."""
        print("[RUN] Star tracker mode selected")
        # Add your star tracker imports and integration here
        self.run(video_source)


def main():
    """Parse arguments and start system."""
    parser = argparse.ArgumentParser(
        description="Space Object Detection, Relative Navigation & Grasping System"
    )
    parser.add_argument("--config", type=str, default="config/params.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--video", type=str, default="0",
                       help="Video source (0 for webcam, path for video file)")
    parser.add_argument("--star-tracker", action="store_true",
                       help="Enable star tracker integration")
    
    args = parser.parse_args()
    
    # Convert video argument
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    # Initialize and run
    system = SpaceGraspingSystem(args.config)
    
    if args.star_tracker:
        system.run_with_star_tracker(video_source)
    else:
        system.run(video_source)


if __name__ == "__main__":
    main()
