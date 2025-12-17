"""
Video Processing Module
Extracts frames and audio from video files for analysis
"""
import cv2
import os
import base64
import subprocess
from typing import List, Tuple, Optional
import numpy as np

class VideoProcessor:
    """Process videos to extract frames and audio"""
    
    def __init__(self, max_frames: int = 30, fps_sample_rate: float = 1.0):
        """
        Initialize video processor
        
        Args:
            max_frames: Maximum number of frames to extract
            fps_sample_rate: Sample rate in frames per second (1.0 = 1 frame/sec, 2.0 = 2 frames/sec)
        """
        self.max_frames = max_frames
        self.fps_sample_rate = fps_sample_rate
    
    def extract_frames(self, video_path: str, output_format: str = 'base64') -> List[str]:
        """
        Extract frames from video at specified sample rate
        
        Args:
            video_path: Path to video file
            output_format: 'base64' or 'bytes'
            
        Returns:
            List of frames (base64 encoded strings or bytes)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            print(f"Video properties: {total_frames} frames, {video_fps:.2f} FPS, {duration:.2f}s duration")
            
            # Calculate frame interval based on sample rate
            frame_interval = int(video_fps / self.fps_sample_rate) if video_fps > 0 else 1
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frame at specified interval
                if frame_count % frame_interval == 0:
                    # Resize frame to reduce size (max width 1024px)
                    frame = self._resize_frame(frame, max_width=1024)
                    
                    # Encode frame
                    if output_format == 'base64':
                        encoded = self._encode_frame_base64(frame)
                        frames.append(encoded)
                    else:
                        encoded = self._encode_frame_bytes(frame)
                        frames.append(encoded)
                    
                    extracted_count += 1
                
                frame_count += 1
            
            print(f"✓ Extracted {len(frames)} frames from video (sampled at {self.fps_sample_rate} FPS)")
            return frames
            
        finally:
            cap.release()
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio track from video using ffmpeg
        
        Args:
            video_path: Path to video file
            output_path: Output path for audio file (default: /tmp/audio_<timestamp>.mp3)
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate output path if not provided
        if output_path is None:
            import time
            timestamp = int(time.time())
            output_path = f"/tmp/audio_{timestamp}.mp3"
        
        try:
            # Use ffmpeg to extract audio
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'libmp3lame',  # MP3 codec
                '-ar', '16000',  # 16kHz sample rate (good for speech)
                '-ac', '1',  # Mono
                '-b:a', '64k',  # 64kbps bitrate
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                # Check if video has no audio track
                if 'no audio' in result.stderr.lower() or 'stream not found' in result.stderr.lower():
                    print("⚠ Video has no audio track")
                    return None
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            if not os.path.exists(output_path):
                raise RuntimeError("Audio extraction failed - output file not created")
            
            file_size = os.path.getsize(output_path)
            print(f"✓ Extracted audio: {output_path} ({file_size / 1024:.1f} KB)")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install ffmpeg: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about a video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'file_size_bytes': os.path.getsize(video_path)
            }
        finally:
            cap.release()
    
    def _resize_frame(self, frame: np.ndarray, max_width: int = 1024) -> np.ndarray:
        """Resize frame to reduce size while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        
        if width <= max_width:
            return frame
        
        # Calculate new dimensions
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        
        # Resize
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    def _encode_frame_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """Encode frame as base64 JPEG string"""
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}"
    
    def _encode_frame_bytes(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """Encode frame as JPEG bytes"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()


def process_video_for_analysis(video_path: str) -> Tuple[List[str], Optional[str], dict]:
    """
    Process a video for fair-use analysis
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (frames, audio_path, video_info)
    """
    processor = VideoProcessor(max_frames=30, fps_sample_rate=1.0)
    
    # Get video info
    video_info = processor.get_video_info(video_path)
    print(f"Processing video: {video_info['duration_seconds']:.1f}s, {video_info['width']}x{video_info['height']}")
    
    # Extract frames
    frames = processor.extract_frames(video_path, output_format='base64')
    
    # Extract audio (may return None if no audio track)
    try:
        audio_path = processor.extract_audio(video_path)
    except Exception as e:
        print(f"⚠ Audio extraction failed: {e}")
        audio_path = None
    
    return frames, audio_path, video_info

