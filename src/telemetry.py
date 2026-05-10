# src/telemetry.py
"""
Telemetry handler: Formats and transmits system state for external communication.
Supports serial/UART output with checksum validation.
"""

import json
import time
from typing import Dict, Optional
from datetime import datetime


class TelemetryHandler:
    """Formats and outputs telemetry data for downstream systems."""
    
    def __init__(self, output_format: str = "serial", serial_port: Optional[str] = None):
        """
        Initialize telemetry handler.
        
        Args:
            output_format: "serial", "json", or "both"
            serial_port: Serial port path (e.g., '/dev/ttyUSB0')
        """
        self.output_format = output_format
        self.serial_port = serial_port
        self.sequence_number = 0
        
        # Initialize serial if needed
        self.serial_conn = None
        if serial_port and output_format in ("serial", "both"):
            self._init_serial()
        
        # Statistics
        self.packets_sent = 0
        self.errors = 0
        self.last_send_time = time.time()
    
    def _init_serial(self):
        """Initialize serial connection."""
        try:
            import serial
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=115200,
                timeout=1
            )
            print(f"[Telemetry] Serial port {self.serial_port} opened at 115200 baud")
        except ImportError:
            print("[Telemetry] ⚠️ pyserial not installed. Install with: pip install pyserial")
            self.serial_conn = None
        except Exception as e:
            print(f"[Telemetry] ⚠️ Could not open serial port: {e}")
            self.serial_conn = None
    
    def format_serial(self, data: Dict) -> bytes:
        """
        Format telemetry as serial string with checksum.
        
        Format: dang,<angle>,dist,<distance>,scam,<star_info>,x,<checksum>
        """
        angle_str = f"dang,{data.get('angle_deg', 555)},"
        dist_str = f"dist,{data.get('distance_cm', 555)},"
        mode_str = f"mode,{data.get('mode', 'unknown')},"
        
        star_info = data.get('orientation', {})
        if isinstance(star_info, dict):
            star_str = f"scam,{json.dumps(star_info)}"
        else:
            star_str = f"scam,{str(star_info)}"
        
        # Build message
        message = (angle_str + dist_str + mode_str + star_str).encode('utf-8')
        
        # Add checksum
        checksum = len(message) + len(str(len(message))) + 3
        full_message = message + f",x,{checksum}".encode('utf-8')
        
        return full_message
    
    def format_json(self, data: Dict) -> str:
        """Format telemetry as JSON string."""
        packet = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'sequence': self.sequence_number,
            'data': data
        }
        return json.dumps(packet)
    
    def send(self, data: Dict) -> bool:
        """
        Send telemetry packet.
        
        Args:
            data: Dictionary of telemetry values
            
        Returns:
            success: Whether send was successful
        """
        self.sequence_number += 1
        success = True
        
        try:
            if self.output_format in ("serial", "both") and self.serial_conn:
                serial_msg = self.format_serial(data)
                self.serial_conn.write(serial_msg)
                self.packets_sent += 1
            
            if self.output_format in ("json", "both"):
                json_msg = self.format_json(data)
                print(f"[Telemetry] {json_msg}")
                self.packets_sent += 1
            
            self.last_send_time = time.time()
            
        except Exception as e:
            self.errors += 1
            if self.errors % 10 == 0:  # Suppress frequent error messages
                print(f"[Telemetry] ⚠️ Error: {e}")
            success = False
        
        return success
    
    def close(self):
        """Clean shutdown."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[Telemetry] Serial port closed")
