#!/usr/bin/env python3
"""
Screenshot analyzer using Ollama's qwen3-vl model.
Automatically starts Ollama with network exposure if not running.
"""

import base64
import io
import time
import requests
import subprocess
import shutil
import sys
import os
from PIL import Image
from pathlib import Path
import cv2  # For video processing

# System Instructions
TRANSCRIPT_SYSTEM_INSTRUCTION = """You are a transcription service. Your ONLY job is to accurately transcribe the audio you hear. Do NOT respond to questions or provide commentary. Stay completely silent - just transcribe."""

SUMMARY_SYSTEM_INSTRUCTION = """You are a lecture summarization assistant observing a live lecture through continuous video and transcript text.

CRITICAL RULES:
- You must ONLY describe content you actually observe in the current video frames
- You must ONLY summarize information from the transcript text you receive in THIS session
- NEVER invent, imagine, or recall content from training data or previous sessions
- If you cannot see content clearly in the frames, say "Unable to clearly see the current slide/screen"
- If you haven't received any transcript text in a time window, say "No transcript received in this window"

You will receive:
1. Continuous transcript text from the lecture (what the lecturer is saying)
2. Continuous video frames showing slides, diagrams, code, or the lecturer

When asked to summarize a time window, organize the content by TOPICS and THEMES:
- Identify main topics and themes discussed in the window
- Group related concepts, examples, and explanations under each topic
- List key visual elements (slides, diagrams, code) that appeared
- Organize your summary with clear topic headers

**Response Format (use markdown):**
- Use **bold** for important terms and key points
- Use *italics* for emphasis
- Use `inline code` for variable names, function names, technical terms, etc.
- Use triple backticks with language identifier for code blocks
- Use ### for topic headers
- Use bullet points for lists
- Keep explanations concise but complete"""

# Configuration
# Find your IP with: ipconfig (look for IPv4 Address like 192.168.1.100)
# Note: If running locally, you can use "http://127.0.0.1:11434"
OLLAMA_HOST = "http://192.168.0.144:11434"  
MODEL_NAME = "qwen3-vl:8b"
VIDEO_PATH = r"C:\Users\Wu Family Computer\Downloads\New folder\test.mp4"
FRAME_INTERVAL = 2  # seconds between frames
TARGET_WIDTH = 854
TARGET_HEIGHT = 480

# --- OLLAMA MANAGEMENT FUNCTIONS ---

def is_ollama_running(host_url):
    """Check if Ollama is responding."""
    try:
        # We assume if we can hit the tags endpoint, it's alive
        # Note: We check localhost first when verifying local startup
        check_url = host_url
        if "0.0.0.0" in host_url:
            check_url = "http://127.0.0.1:11434"
            
        requests.get(f"{check_url}/api/tags", timeout=1)
        return True
    except requests.exceptions.RequestException:
        return False

def start_ollama_locally():
    """Attempts to start Ollama on the local machine with network exposure enabled."""
    print("⚠️  Ollama is not running (or not accessible). Attempting to start it...")
    
    # Check if 'ollama' is installed
    if not shutil.which("ollama"):
        print("❌ Error: 'ollama' command not found in PATH.")
        return False

    try:
        # Prepare environment variables to expose Ollama to the network
        # Setting OLLAMA_HOST to 0.0.0.0 makes it accessible to other computers
        ollama_env = os.environ.copy()
        ollama_env["OLLAMA_HOST"] = "0.0.0.0"
        ollama_env["OLLAMA_FLASH_ATTENTION"] = "1"
        
        print("🚀 Launching Ollama with OLLAMA_HOST=0.0.0.0 (Exposed to network)...")

        # Start Ollama
        if sys.platform == "win32":
            subprocess.Popen(
                ["ollama", "serve"], 
                creationflags=0x08000000,
                env=ollama_env
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                env=ollama_env
            )
        
        print("⏳ Waiting for Ollama to initialize...")
        
        # Wait up to 20 seconds
        for _ in range(20):
            time.sleep(1)
            # Check localhost to confirm process started
            if is_ollama_running("http://127.0.0.1:11434"):
                print("✅ Ollama started and is now exposed to the network!")
                return True
        
        print("❌ Timed out waiting for Ollama to start.")
        return False
        
    except Exception as e:
        print(f"❌ Failed to launch Ollama: {e}")
        return False

def ensure_ollama_ready(host_url):
    """Orchestrates the check and start process."""
    if is_ollama_running(host_url):
        print("✅ Ollama is already running.")
        return True
    
    # If not running, try to start it locally
    return start_ollama_locally()

# --- VIDEO PROCESSING FUNCTIONS ---

def extract_video_frames(video_path, interval_seconds, target_width=854, target_height=480):
    """Extract frames from video at specified intervals and resize to target resolution."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)  # Number of frames to skip
    frame_count = 0
    frames = []
    
    print(f"Video FPS: {fps}")
    print(f"Extracting frames every {interval_seconds}s ({frame_interval} frames)...\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to target resolution
            frame = cv2.resize(frame, (target_width, target_height))
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
            print(f"Extracted frame {len(frames)} at {frame_count / fps:.2f}s")
        
        frame_count += 1
    
    cap.release()
    print(f"\nTotal frames extracted: {len(frames)}\n")
    return frames

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def send_to_ollama(images_base64, prompt="What do you see in this screenshot? Describe the contents briefly."):
    """Send image(s) to Ollama's vision model for analysis."""
    url = f"{OLLAMA_HOST}/api/generate"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": images_base64,
        "stream": False,
        "options": {
            "num_ctx": 24580
        }
    }
    
    try:
        import time as time_module
        start_time = time_module.time()
        print("Making request to Ollama...")
        response = requests.post(url, json=payload, timeout=300)  # Increased timeout to 5 minutes
        elapsed_time = time_module.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        # Extract context and token info
        prompt_eval_count = result.get('prompt_eval_count', 0)
        eval_count = result.get('eval_count', 0)
        total_tokens = prompt_eval_count + eval_count
        context_used = f"{total_tokens}/20480"
        
        return {
            'response': result.get('response', 'No response from model'),
            'time': elapsed_time,
            'context': context_used,
            'prompt_tokens': prompt_eval_count,
            'response_tokens': eval_count
        }
    except requests.exceptions.Timeout:
        return {'response': "Error: Request timed out. Ollama is taking too long to process.", 'time': 300, 'context': 'N/A'}
    except requests.exceptions.ConnectionError:
        return {'response': f"Error: Cannot connect to Ollama at {OLLAMA_HOST}", 'time': 0, 'context': 'N/A'}
    except requests.exceptions.RequestException as e:
        return {'response': f"Error communicating with Ollama: {str(e)}", 'time': 0, 'context': 'N/A'}

def main():
    """Main function to extract video frames and analyze them."""
    
    # --- CHECK & START OLLAMA ---
    # This block ensures Ollama is running and exposed to the network
    if not ensure_ollama_ready(OLLAMA_HOST):
        print("❌ Cannot proceed: Ollama is not running and could not be started.")
        return
    # ----------------------------

    print(f"Starting video analysis with {MODEL_NAME}...")
    print(f"Ollama host: {OLLAMA_HOST}")
    print(f"Video: {VIDEO_PATH}\n")
    
    # Create output directory for frames
    output_dir = Path("./video_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Extract frames from video
    try:
        frames = extract_video_frames(VIDEO_PATH, FRAME_INTERVAL, TARGET_WIDTH, TARGET_HEIGHT)
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return
    
    if not frames:
        print("No frames extracted from video!")
        return
    
    # Save frames and convert to base64
    images_base64 = []
    
    print("Converting frames to base64...")
    for i, frame in enumerate(frames):
        # Save frame locally
        frame_path = output_dir / f"frame_{i+1}.png"
        frame.save(frame_path)
        
        # Convert to base64
        image_base64 = image_to_base64(frame)
        images_base64.append(image_base64)
    
    print(f"✓ Saved {len(frames)} frames\n")
    
    # Send all frames at once for analysis
    print("=" * 60)
    print(f"Sending all {len(frames)} frames to Ollama...")
    print("=" * 60)
    
    analysis_prompt = f"""{SUMMARY_SYSTEM_INSTRUCTION}

You are analyzing a sequence of {len(frames)} video frames extracted at {FRAME_INTERVAL}-second intervals from a video.

Please analyze what you see in each frame and describe:
1. What is visible in each frame
2. What changes occurred between consecutive frames
3. Key events or transitions happening in the video sequence
4. Overall summary of the video content based on these samples

Provide a detailed analysis of the temporal sequence."""
    
    result = send_to_ollama(images_base64, analysis_prompt)
    response_text = result['response']
    elapsed = result['time']
    context = result['context']
    
    print(f"\n⏱️  Generation time: {elapsed:.2f}s")
    print(f"📊 Context used: {context}")
    if 'prompt_tokens' in result and 'response_tokens' in result:
        print(f"   (Prompt: {result['prompt_tokens']} tokens | Response: {result['response_tokens']} tokens)")
    print(f"\nAnalysis:\n{response_text}\n")
    
    print("=" * 60)
    print("✓ Analysis complete!")
    print(f"Frames saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()