"""
Cloud Storage Integration for Video Uploads
Supports AWS S3 with presigned URLs for secure client-side uploads
"""
import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import uuid
from typing import Optional, Dict

class CloudStorage:
    """Handle S3 operations for video storage"""
    
    def __init__(self):
        self.storage_type = os.getenv('CLOUD_STORAGE_TYPE', 's3')
        self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Validate configuration
        if not self.bucket_name:
            raise ValueError("AWS_S3_BUCKET_NAME environment variable is required")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=self.region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        print(f"✓ Cloud storage initialized: S3 bucket '{self.bucket_name}' in {self.region}")
    
    def generate_video_key(self, file_extension: str = 'mp4') -> str:
        """Generate a unique key for a video file"""
        timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"videos/{timestamp}-{unique_id}.{file_extension}"
    
    def get_presigned_upload_url(self, video_key: str, expiration: int = 600) -> Dict[str, str]:
        """
        Generate a presigned URL for uploading a video directly to S3
        
        Args:
            video_key: The S3 key where the video will be stored
            expiration: URL expiration time in seconds (default: 10 minutes)
            
        Returns:
            Dict with 'upload_url' and 'video_key'
        """
        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': video_key,
                    'ContentType': 'video/mp4'
                },
                ExpiresIn=expiration
            )
            
            return {
                'upload_url': presigned_url,
                'video_key': video_key,
                'bucket': self.bucket_name
            }
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")
            raise
    
    def download_video(self, video_key: str, local_path: str) -> bool:
        """
        Download a video from S3 to local filesystem
        
        Args:
            video_key: The S3 key of the video
            local_path: Local file path where video will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.download_file(self.bucket_name, video_key, local_path)
            print(f"✓ Downloaded video from S3: {video_key} -> {local_path}")
            return True
        except ClientError as e:
            print(f"Error downloading video from S3: {e}")
            return False
    
    def delete_video(self, video_key: str) -> bool:
        """
        Delete a video from S3
        
        Args:
            video_key: The S3 key of the video to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=video_key)
            print(f"✓ Deleted video from S3: {video_key}")
            return True
        except ClientError as e:
            print(f"Error deleting video from S3: {e}")
            return False
    
    def video_exists(self, video_key: str) -> bool:
        """
        Check if a video exists in S3
        
        Args:
            video_key: The S3 key to check
            
        Returns:
            True if video exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=video_key)
            return True
        except ClientError:
            return False
    
    def get_video_metadata(self, video_key: str) -> Optional[Dict]:
        """
        Get metadata about a video in S3
        
        Args:
            video_key: The S3 key of the video
            
        Returns:
            Dict with metadata or None if video doesn't exist
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=video_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown')
            }
        except ClientError as e:
            print(f"Error getting video metadata: {e}")
            return None


# Global instance
_storage = None

def get_storage() -> CloudStorage:
    """Get or create the global storage instance"""
    global _storage
    if _storage is None:
        _storage = CloudStorage()
    return _storage

