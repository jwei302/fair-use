"""
OpenAI API Integration for Video Analysis
Uses GPT-4V for visual analysis and Whisper for audio transcription
"""
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI

# Initialize OpenAI client
client = None

def get_client():
    """Get or create OpenAI client"""
    global client
    if client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized")
    return client


# Prompts for GPT-4V analysis
SIMILARITY_PROMPT = """You are analyzing video content to identify its source material.

I will provide you with frames from a video and its audio transcript.

Your task:
1. Identify what copyrighted work(s) this video appears to use or derive from
2. Specify titles, creators, copyright holders if recognizable
3. Describe the type of content (movie, TV show, music video, game, etc.)
4. Estimate how much of the original work is present

Be specific and objective. If you cannot identify the source, say so.

Respond in JSON format:
{
  "identified_works": [
    {
      "title": "Name of work",
      "creator": "Creator/copyright holder",
      "content_type": "movie/tv/music/game/other",
      "confidence": "high/medium/low",
      "evidence": "What specific elements helped you identify this"
    }
  ],
  "unidentified": true or false,
  "summary": "Brief summary of findings"
}"""


FAIR_USE_PROMPT = """You are a fair-use assessment tool for research purposes.

Given video frames, audio transcript, and identified source material, evaluate fair-use risk.

For each of the four statutory fair-use factors, provide:
- A risk score from 0-100 (0 = strong fair use argument, 100 = high infringement risk)
- Detailed explanation citing specific observations from the video

Factor 1 - Purpose and Character of the Use:
- Is the use transformative? Does it add new expression, meaning, or message?
- Is it commercial, educational, or personal?
- Does it comment on, critique, or parody the original?

Factor 2 - Nature of the Copyrighted Work:
- Is the original work creative (fiction, music, art) or factual (news, documentary)?
- Was it published before this use?
- How much creative expression does the original contain?

Factor 3 - Amount and Substantiality of Portion Used:
- What percentage of the original work is used?
- Is the "heart" of the work (most memorable/important part) taken?
- Is the amount used reasonable for the stated purpose?

Factor 4 - Effect on the Market for the Original Work:
- Could this video serve as a substitute for the original?
- Does it target the same audience and market?
- Could it harm current or potential licensing markets?
- Might it reduce demand for the original?

Additionally, provide:
- Overall fair-use risk score (0-100, weighted across factors)
- Confidence/completeness score (0-100) based on video quality and your ability to analyze

Respond ONLY with valid JSON:
{
  "overall_risk_score": <0-100>,
  "risk_level": "Low Risk (0-33)" or "Moderate Risk (34-66)" or "High Risk (67-100)",
  "confidence_score": <0-100>,
  "similar_content_summary": "brief summary of identified source material",
  "factors": {
    "purpose_and_character": {
      "score": <0-100>,
      "explanation": "detailed analysis..."
    },
    "nature_of_work": {
      "score": <0-100>,
      "explanation": "detailed analysis..."
    },
    "amount_and_substantiality": {
      "score": <0-100>,
      "explanation": "detailed analysis..."
    },
    "market_effect": {
      "score": <0-100>,
      "explanation": "detailed analysis..."
    }
  }
}"""


def transcribe_audio(audio_file_path: str) -> Optional[str]:
    """
    Transcribe audio using OpenAI Whisper API
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Transcript text or None if transcription fails
    """
    if not audio_file_path or not os.path.exists(audio_file_path):
        print("⚠ No audio file to transcribe")
        return None
    
    try:
        print(f"Transcribing audio with Whisper API...")
        api_client = get_client()
        
        with open(audio_file_path, 'rb') as audio_file:
            transcript = api_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        print(f"✓ Audio transcribed ({len(transcript)} characters)")
        return transcript
    
    except Exception as e:
        print(f"⚠ Audio transcription failed: {e}")
        return None


def find_similar_content(frames: List[str], transcript: Optional[str] = None) -> Dict:
    """
    Use GPT-4V to identify what copyrighted content the video resembles
    
    Args:
        frames: List of base64-encoded frame images
        transcript: Optional audio transcript
        
    Returns:
        Dict with identification results
    """
    try:
        print(f"Analyzing frames to identify similar content...")
        api_client = get_client()
        
        # Select a subset of frames (6-8 representative frames)
        # Take frames evenly distributed throughout the video
        num_frames_to_send = min(8, len(frames))
        step = len(frames) // num_frames_to_send if num_frames_to_send > 0 else 1
        selected_frames = [frames[i] for i in range(0, len(frames), step)][:num_frames_to_send]
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at identifying copyrighted content in videos."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SIMILARITY_PROMPT}
                ]
            }
        ]
        
        # Add frames
        for idx, frame in enumerate(selected_frames):
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": frame,
                    "detail": "low"  # Use low detail to reduce costs
                }
            })
        
        # Add transcript if available
        if transcript:
            messages[1]["content"].append({
                "type": "text",
                "text": f"\n\nAudio Transcript:\n{transcript[:2000]}"  # Limit transcript length
            })
        
        # Call GPT-4V
        response = api_client.chat.completions.create(
            model="gpt-4o",  # or gpt-4-vision-preview
            messages=messages,
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent output
        )
        
        result_text = response.choices[0].message.content
        print(f"✓ Similar content analysis complete")
        
        # Parse JSON response
        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {
                "identified_works": [],
                "unidentified": True,
                "summary": result_text
            }
    
    except Exception as e:
        print(f"⚠ Similar content analysis failed: {e}")
        return {
            "identified_works": [],
            "unidentified": True,
            "summary": f"Analysis failed: {str(e)}"
        }


def evaluate_fair_use(frames: List[str], transcript: Optional[str], similar_content: Dict) -> Dict:
    """
    Use GPT-4V to evaluate fair-use risk across four factors
    
    Args:
        frames: List of base64-encoded frame images
        transcript: Optional audio transcript
        similar_content: Results from similarity detection
        
    Returns:
        Dict with fair-use evaluation including scores for each factor
    """
    try:
        print(f"Evaluating fair-use factors with GPT-4V...")
        api_client = get_client()
        
        # Select frames for analysis (10-12 representative frames)
        num_frames_to_send = min(12, len(frames))
        step = len(frames) // num_frames_to_send if num_frames_to_send > 0 else 1
        selected_frames = [frames[i] for i in range(0, len(frames), step)][:num_frames_to_send]
        
        # Build context about identified content
        similar_content_text = similar_content.get('summary', 'No specific content identified')
        if similar_content.get('identified_works'):
            works_text = "\n".join([
                f"- {work.get('title', 'Unknown')} by {work.get('creator', 'Unknown')} "
                f"(confidence: {work.get('confidence', 'unknown')})"
                for work in similar_content['identified_works']
            ])
            similar_content_text = f"Identified works:\n{works_text}"
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": "You are a fair-use assessment tool providing educational analysis."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FAIR_USE_PROMPT},
                    {"type": "text", "text": f"\n\nIdentified Source Material:\n{similar_content_text}"}
                ]
            }
        ]
        
        # Add frames
        for idx, frame in enumerate(selected_frames):
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": frame,
                    "detail": "high"  # Use high detail for detailed analysis
                }
            })
        
        # Add transcript if available
        if transcript:
            messages[1]["content"].append({
                "type": "text",
                "text": f"\n\nAudio Transcript:\n{transcript[:3000]}"  # Limit transcript length
            })
        
        # Call GPT-4V
        response = api_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
            response_format={"type": "json_object"}  # Request JSON output
        )
        
        result_text = response.choices[0].message.content
        print(f"✓ Fair-use evaluation complete")
        
        # Parse JSON response
        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to parse JSON response: {e}")
            # Return a default structure
            return {
                "overall_risk_score": 50,
                "risk_level": "Moderate Risk (34-66)",
                "confidence_score": 30,
                "similar_content_summary": similar_content_text,
                "factors": {
                    "purpose_and_character": {
                        "score": 50,
                        "explanation": "Unable to complete analysis due to parsing error."
                    },
                    "nature_of_work": {
                        "score": 50,
                        "explanation": "Unable to complete analysis due to parsing error."
                    },
                    "amount_and_substantiality": {
                        "score": 50,
                        "explanation": "Unable to complete analysis due to parsing error."
                    },
                    "market_effect": {
                        "score": 50,
                        "explanation": "Unable to complete analysis due to parsing error."
                    }
                },
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": result_text
            }
    
    except Exception as e:
        print(f"⚠ Fair-use evaluation failed: {e}")
        return {
            "overall_risk_score": 50,
            "risk_level": "Moderate Risk (34-66)",
            "confidence_score": 0,
            "similar_content_summary": "Analysis failed",
            "factors": {
                "purpose_and_character": {
                    "score": 50,
                    "explanation": f"Analysis failed: {str(e)}"
                },
                "nature_of_work": {
                    "score": 50,
                    "explanation": f"Analysis failed: {str(e)}"
                },
                "amount_and_substantiality": {
                    "score": 50,
                    "explanation": f"Analysis failed: {str(e)}"
                },
                "market_effect": {
                    "score": 50,
                    "explanation": f"Analysis failed: {str(e)}"
                }
            },
            "error": str(e)
        }


def analyze_video_complete(frames: List[str], transcript: Optional[str]) -> Dict:
    """
    Complete video analysis pipeline: similarity detection + fair-use evaluation
    
    Args:
        frames: List of base64-encoded frame images
        transcript: Optional audio transcript
        
    Returns:
        Combined analysis results
    """
    # Step 1: Identify similar content
    similar_content = find_similar_content(frames, transcript)
    
    # Step 2: Evaluate fair-use
    fair_use_eval = evaluate_fair_use(frames, transcript, similar_content)
    
    # Combine results
    return {
        "similar_content": similar_content,
        "fair_use_evaluation": fair_use_eval,
        "analysis_complete": True
    }

