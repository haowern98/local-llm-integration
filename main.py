#!/usr/bin/env python3
"""
Professional Lecture Analysis System
Generates comprehensive reports with screenshot citations using Ollama's qwen3-vl model.
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
import json
import argparse
from datetime import datetime
import webbrowser  # For opening HTML in browser

# Import modules (not specific classes) to avoid circular imports
import tools
import report_generator

# System Instructions
SUMMARY_SYSTEM_INSTRUCTION = """You are a lecture summarization assistant observing a complete lecture through continuous video frames.

CRITICAL RULES:
- You must analyze the ENTIRE sequence of frames to understand the full narrative flow
- You must identify how content evolves and transitions between concepts over time
- NEVER invent, imagine, or recall content from training data or previous sessions
- Focus on describing what you actually observe in the video frames
- Identify 3-5 KEY MOMENTS that would benefit from visual citations (screenshots)

You will receive:
1. A complete sequence of video frames showing slides, diagrams, code, or the lecturer
2. The frames represent a continuous lecture with temporal progression

When summarizing the entire video, organize the content by:
- Major topics and themes that emerged throughout the lecture
- How concepts evolved and built upon each other over time
- Key transitions between different sections of the lecture
- Important visual elements (slides, diagrams, code) that appeared
- Critical moments that would benefit from visual evidence (screenshots)

**Response Format (use markdown):**
### 📋 Executive Summary
[2-3 sentence overview of the entire lecture]

### 🎯 Major Topics & Themes
- **Topic 1**: Brief description of the topic and its importance
- **Topic 2**: Brief description of the topic and its importance

### 🔄 Content Evolution & Transitions
[Describe how concepts developed and changed over time, including key transitions between topics]

### 📸 Key Moments Needing Visual Citations
[Identify 3-5 specific moments where seeing the actual content would significantly enhance understanding]
- **Timestamp X.Xs**: [Brief description of why this moment needs visual evidence]
- **Timestamp Y.Ys**: [Brief description of why this moment needs visual evidence]
"""

# Configuration
OLLAMA_HOST = "http://192.168.0.144:11434"  
MODEL_NAME = "qwen3-vl:8b"
VIDEO_PATH = r"C:\Users\Wu Family Computer\Downloads\New folder\test.mp4"
FRAME_INTERVAL = 2  # seconds between frames
TARGET_WIDTH = 854
TARGET_HEIGHT = 480
OUTPUT_DIR = Path("./analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- OLLAMA MANAGEMENT FUNCTIONS ---

def is_ollama_running(host_url):
    """Check if Ollama is responding."""
    try:
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
    
    if not shutil.which("ollama"):
        print("❌ Error: 'ollama' command not found in PATH.")
        return False

    try:
        ollama_env = os.environ.copy()
        ollama_env["OLLAMA_HOST"] = "0.0.0.0"
        ollama_env["OLLAMA_FLASH_ATTENTION"] = "1"
        
        print("🚀 Launching Ollama with OLLAMA_HOST=0.0.0.0 (Exposed to network)...")

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
        
        for _ in range(20):
            time.sleep(1)
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
    
    return start_ollama_locally()

# --- VIDEO PROCESSING FUNCTIONS ---

def extract_video_frames(video_path, interval_seconds, target_width=854, target_height=480):
    """Extract frames from video at specified intervals and resize to target resolution."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    frames = []
    timestamps = []
    
    print(f"🎥 Video FPS: {fps:.2f}")
    print(f"⏱️  Extracting frames every {interval_seconds}s ({frame_interval} frames)...\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target_width, target_height))
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
            timestamp = frame_count / fps
            timestamps.append(timestamp)
            print(f"✅ Extracted frame {len(frames)} at {timestamp:.2f}s")
        
        frame_count += 1
    
    cap.release()
    print(f"\n📊 Total frames extracted: {len(frames)}\n")
    return frames, timestamps

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
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=600)  # 10-minute timeout
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        # Extract context and token info
        prompt_eval_count = result.get('prompt_eval_count', 0)
        eval_count = result.get('eval_count', 0)
        total_tokens = prompt_eval_count + eval_count
        context_used = f"{total_tokens}/24580"
        
        return {
            'response': result.get('response', 'No response from model'),
            'time': elapsed_time,
            'context': context_used,
            'prompt_tokens': prompt_eval_count,
            'response_tokens': eval_count
        }
    except requests.exceptions.Timeout:
        return {'response': "Error: Request timed out. Ollama is taking too long to process.", 'time': 600, 'context': 'N/A'}
    except requests.exceptions.ConnectionError:
        return {'response': f"Error: Cannot connect to Ollama at {OLLAMA_HOST}", 'time': 0, 'context': 'N/A'}
    except requests.exceptions.RequestException as e:
        return {'response': f"Error communicating with Ollama: {str(e)}", 'time': 0, 'context': 'N/A'}

def parse_comprehensive_summary(response_text):
    """Parse the comprehensive summary response into structured data."""
    try:
        # Extract key sections using markdown headers
        sections = {}
        
        # Executive Summary
        if "### 📋 Executive Summary" in response_text:
            exec_summary = response_text.split("### 📋 Executive Summary")[1].split("###")[0].strip()
            sections['executive_summary'] = exec_summary
        else:
            sections['executive_summary'] = response_text[:200] + "..."  # Fallback
        
        # Major Topics
        if "### 🎯 Major Topics & Themes" in response_text:
            topics_section = response_text.split("### 🎯 Major Topics & Themes")[1].split("###")[0].strip()
            sections['major_topics'] = topics_section
        else:
            sections['major_topics'] = "Major topics were discussed throughout the lecture."
        
        # Content Evolution
        if "### 🔄 Content Evolution & Transitions" in response_text:
            evolution_section = response_text.split("### 🔄 Content Evolution & Transitions")[1].split("###")[0].strip()
            sections['content_evolution'] = evolution_section
        else:
            sections['content_evolution'] = "The lecture content evolved logically through different concepts."
        
        # Key Moments
        key_moments = []
        if "### 📸 Key Moments Needing Visual Citations" in response_text:
            moments_section = response_text.split("### 📸 Key Moments Needing Visual Citations")[1].split("###")[0].strip()
            # Parse moments with timestamps
            for line in moments_section.split('\n'):
                if line.strip().startswith('-') and 'Timestamp' in line:
                    try:
                        # Extract timestamp and description
                        timestamp_str = line.split('Timestamp')[1].split(':')[0].strip()
                        timestamp = float(timestamp_str.replace('s', ''))
                        description = line.split(':', 1)[1].strip()
                        key_moments.append({
                            'timestamp': timestamp,
                            'description': description,
                            'reason': 'Identified as key moment needing visual evidence'
                        })
                    except:
                        continue
        
        sections['key_moments'] = key_moments if key_moments else [
            {'timestamp': 30.0, 'description': 'Early lecture content', 'reason': 'Starting point reference'},
            {'timestamp': 60.0, 'description': 'Mid-lecture content', 'reason': 'Middle point reference'},
            {'timestamp': 90.0, 'description': 'Later lecture content', 'reason': 'Ending point reference'}
        ]
        
        return sections
    
    except Exception as e:
        print(f"⚠️  Warning: Failed to parse summary structure: {e}")
        return {
            'executive_summary': response_text[:300] + "...",
            'major_topics': "Topics covered in the lecture",
            'content_evolution': "Content evolved throughout the lecture",
            'key_moments': [
                {'timestamp': 30.0, 'description': 'Default moment 1', 'reason': 'Fallback moment'},
                {'timestamp': 60.0, 'description': 'Default moment 2', 'reason': 'Fallback moment'}
            ]
        }

def generate_comprehensive_video_summary(frames, timestamps):
    """Generate a comprehensive summary of the entire video sequence."""
    print("🧠 Generating comprehensive video summary...")
    
    # Convert all frames to base64
    frames_base64 = [image_to_base64(frame) for frame in frames]
    total_duration = timestamps[-1] if timestamps else len(frames) * FRAME_INTERVAL
    
    comprehensive_prompt = f"""
    {SUMMARY_SYSTEM_INSTRUCTION}
    
    You are analyzing the COMPLETE SEQUENCE of {len(frames)} video frames from a lecture.
    
    **Temporal Context:**
    - Frames extracted every {FRAME_INTERVAL} seconds
    - Total video duration: approximately {total_duration:.1f} seconds
    - Timestamp of first frame: {timestamps[0]:.1f}s
    - Timestamp of last frame: {timestamps[-1]:.1f}s
    - This is a continuous lecture - content builds upon itself over time
    
    **Your Task:**
    1. Analyze the ENTIRE sequence, describing how content evolves over time
    2. Identify major topics, themes, and transitions between concepts
    3. Note key visual elements that appeared (slides, code, diagrams)
    4. Identify 3-5 CRITICAL MOMENTS that would benefit from visual citations
    
    **Critical:** Focus on the temporal flow and narrative progression. How do concepts build upon each other?
    """
    
    start_time = time.time()
    result = send_to_ollama(frames_base64, comprehensive_prompt)
    elapsed_time = time.time() - start_time
    
    print(f"✅ Summary generated in {elapsed_time:.2f}s")
    print(f"📊 Context used: {result['context']}")
    print(f"   (Prompt: {result['prompt_tokens']} tokens | Response: {result['response_tokens']} tokens)")
    
    # Save raw summary for debugging
    raw_summary_path = OUTPUT_DIR / f"raw_summary_{int(time.time())}.txt"
    with open(raw_summary_path, 'w', encoding='utf-8') as f:
        f.write(result['response'])
    print(f"📝 Raw summary saved to: {raw_summary_path}")
    
    # Parse into structured format
    return parse_comprehensive_summary(result['response']), result['response']

def identify_key_moments_for_citations(full_summary, frames, timestamps):
    """Identify specific moments that need screenshot citations."""
    print("\n🎯 Identifying key moments for visual citations...")
    
    key_moments_analyzer = tools.KeyMomentAnalyzer()
    return key_moments_analyzer.identify_key_moments(full_summary, frames, timestamps, OLLAMA_HOST, MODEL_NAME)

def capture_screenshots_for_key_moments(key_moments, frames, timestamps):
    """Capture screenshots for the identified key moments."""
    print("\n📸 Capturing screenshots for key moments...")
    
    screenshot_capturer = tools.ScreenshotCapturer(OUTPUT_DIR / "screenshots")
    return screenshot_capturer.capture_screenshots(key_moments, frames, timestamps, FRAME_INTERVAL)

def generate_markdown_report(full_summary, screenshots, analysis_metadata):
    """Generate the professional markdown report."""
    print("\n📄 Generating professional markdown report...")
    
    report_generator_instance = report_generator.ProfessionalReportGenerator(OUTPUT_DIR / "reports")
    return report_generator_instance.generate_report(full_summary, screenshots, analysis_metadata)

def convert_md_to_html(md_file_path):
    """Convert markdown report to HTML with professional styling and fixed screenshot paths."""
    try:
        # Try to import markdown library
        import markdown
        
        # Fix screenshot paths first
        fixed_md_path = fix_screenshot_paths_for_html(md_file_path)
        
        # Read the (fixed) markdown file
        with open(fixed_md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML with extensions
        html_content = markdown.markdown(
            md_content, 
            extensions=['tables', 'fenced_code', 'nl2br']
        )
        
        # Get current time for footer
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add professional HTML structure with styling
        full_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lecture Analysis Report</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --accent-color: #e74c3c;
            --light-color: #ecf0f1;
            --dark-color: #34495e;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #c0392b;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            padding: 40px;
            margin-top: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid var(--primary-color);
        }
        
        h1 {
            color: var(--accent-color);
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            color: var(--secondary-color);
            font-size: 1.2em;
            margin-bottom: 15px;
        }
        
        .metadata {
            background: var(--light-color);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            font-family: monospace;
            font-size: 0.9em;
            color: var(--dark-color);
        }
        
        h2 {
            color: var(--primary-color);
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 10px;
            margin: 30px 0 20px 0;
            font-size: 1.8em;
        }
        
        h3 {
            color: var(--success-color);
            margin: 25px 0 15px 0;
            font-size: 1.5em;
        }
        
        h4 {
            color: var(--warning-color);
            margin: 20px 0 10px 0;
            font-size: 1.3em;
        }
        
        p {
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        ul, ol {
            margin: 15px 0 15px 30px;
            padding-left: 20px;
        }
        
        li {
            margin-bottom: 8px;
            font-size: 1.05em;
        }
        
        img {
            max-width: 100%;
            height: auto;
            border: 3px solid var(--primary-color);
            border-radius: 10px;
            margin: 25px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        
        .caption {
            text-align: center;
            font-style: italic;
            color: var(--dark-color);
            margin-top: -15px;
            margin-bottom: 25px;
            font-size: 0.95em;
        }
        
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            font-family: 'Consolas', monospace;
            font-size: 0.95em;
            border-left: 4px solid var(--accent-color);
        }
        
        code {
            background: #f1f1f1;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            color: var(--secondary-color);
            font-size: 0.95em;
        }
        
        blockquote {
            border-left: 4px solid var(--primary-color);
            padding: 15px 20px;
            margin: 20px 0;
            background: rgba(52, 152, 219, 0.05);
            border-radius: 0 8px 8px 0;
            font-style: italic;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid var(--light-color);
            color: var(--dark-color);
            font-size: 0.9em;
        }
        
        .emoji {
            font-size: 1.2em;
            margin-right: 5px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
""" + html_content + """
        
        <footer>
            <p>Generated by Professional Lecture Analysis System | """ + current_time + """</p>
            <p>Powered by Ollama's qwen3-vl model</p>
        </footer>
    </div>
</body>
</html>"""
        
        # Save HTML file
        html_file_path = Path(md_file_path).with_suffix('.html')
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        # Clean up temporary fixed markdown file
        try:
            fixed_md_file = Path(fixed_md_path)
            if fixed_md_file.exists() and "_fixed.md" in fixed_md_file.name:
                fixed_md_file.unlink()
        except:
            pass
        
        print(f"🌐 HTML version saved: {html_file_path}")
        return str(html_file_path)
        
    except ImportError:
        print("⚠️  Warning: 'markdown' library not installed. Install with: pip install markdown")
        print("📊 Displaying markdown report path instead:")
        return None
    except Exception as e:
        print(f"❌ Error converting to HTML: {e}")
        return None

def fix_screenshot_paths_for_html(md_file_path):
    """Fix screenshot paths in markdown to work with HTML conversion"""
    try:
        # Read the markdown file
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get the directory where the markdown file is located
        md_dir = Path(md_file_path).parent
        reports_dir = md_dir  # Usually analysis_output/reports/
        screenshots_dir = reports_dir.parent / "screenshots"  # analysis_output/screenshots/
        
        # Fix image paths: change "screenshots/filename.png" to "../screenshots/filename.png"
        import re
        
        # Pattern to match screenshot paths in markdown images
        # ![caption](screenshots/filename.png)
        pattern = r'!\[([^\]]*)\]\((screenshots/[^\)]+)\)'
        
        def replace_path(match):
            caption = match.group(1)
            old_path = match.group(2)  # "screenshots/filename.png"
            # Convert to relative path from reports/ to screenshots/
            new_path = f"../{old_path}"  # "../screenshots/filename.png"
            return f"![{caption}]({new_path})"
        
        fixed_content = re.sub(pattern, replace_path, content)
        
        # Save the fixed markdown (optional - you can save to a temp file instead)
        fixed_md_path = md_file_path.with_name(md_file_path.stem + "_fixed.md")
        with open(fixed_md_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return str(fixed_md_path)
        
    except Exception as e:
        print(f"⚠️  Warning: Could not fix screenshot paths: {e}")
        return str(md_file_path)  # Return original if fixing fails

def main():
    """Main function to orchestrate the entire analysis workflow."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Professional Lecture Analysis System')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--host', type=str, help='Ollama host URL')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--interval', type=float, help='Frame interval in seconds')
    args = parser.parse_args()
    
    # Use command line args or defaults
    video_path = args.video if args.video else VIDEO_PATH
    ollama_host = args.host if args.host else OLLAMA_HOST
    model_name = args.model if args.model else MODEL_NAME
    frame_interval = args.interval if args.interval else FRAME_INTERVAL
    
    print("=" * 60)
    print("🎓 PROFESSIONAL LECTURE ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"🎬 Video: {video_path}")
    print(f"🤖 Model: {model_name}")
    print(f"⏱️  Frame interval: {frame_interval}s")
    print(f"🏠 Ollama host: {ollama_host}")
    print("=" * 60 + "\n")
    
    # --- CHECK & START OLLAMA ---
    if not ensure_ollama_ready(ollama_host):
        print("❌ Cannot proceed: Ollama is not running and could not be started.")
        return
    # ----------------------------
    
    analysis_start_time = time.time()
    
    # Extract frames from video
    try:
        frames, timestamps = extract_video_frames(video_path, frame_interval, TARGET_WIDTH, TARGET_HEIGHT)
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        return
    
    if not frames:
        print("❌ No frames extracted from video!")
        return
    
    # Save frames locally
    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)
    print(f"\n💾 Saving extracted frames to: {frames_dir.absolute()}")
    
    for i, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{i+1:04d}_{timestamps[i]:.1f}s.png"
        frame.save(frame_path)
    
    # Generate comprehensive summary
    full_summary, raw_summary = generate_comprehensive_video_summary(frames, timestamps)
    
    # Identify key moments for citations
    key_moments = identify_key_moments_for_citations(full_summary, frames, timestamps)
    
    # Capture screenshots for key moments
    screenshots = capture_screenshots_for_key_moments(key_moments, frames, timestamps)
    
    # Generate metadata for report
    analysis_metadata = {
        'video_path': video_path,
        'model_name': model_name,
        'ollama_host': ollama_host,
        'frame_interval': frame_interval,
        'target_resolution': f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        'total_frames': len(frames),
        'total_duration': timestamps[-1] if timestamps else len(frames) * frame_interval,
        'analysis_start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'analysis_total_time': time.time() - analysis_start_time,
        'key_moments_count': len(key_moments),
        'screenshots_count': len(screenshots)
    }
    
    # Generate markdown report
    report_path = generate_markdown_report(full_summary, screenshots, analysis_metadata)
    
    # Convert to HTML and open in browser (new functionality)
    print("\n" + "=" * 60)
    print("🌐 CONVERTING TO HTML AND OPENING IN BROWSER")
    print("=" * 60)
    
    html_path = convert_md_to_html(report_path)
    if html_path:
        try:
            # Open HTML file in default browser
            webbrowser.open(f'file://{os.path.abspath(html_path)}')
            print(f"✅ HTML report opened in your default browser!")
            print(f"🔗 HTML file path: {html_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not open browser automatically: {e}")
            print(f"🖱️  Please open this file manually: {html_path}")
    else:
        print("⚠️  HTML conversion failed. Please view the markdown file directly.")
        print(f"📄 Markdown file path: {report_path}")
    
    # Print completion summary
    total_time = time.time() - analysis_start_time
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📄 Report saved to: {report_path}")
    if html_path:
        print(f"🌐 HTML report saved to: {html_path}")
    print(f"🖼️  Screenshots saved to: {OUTPUT_DIR / 'screenshots'}")
    print(f"📊 Total frames analyzed: {len(frames)}")
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"🎯 Key moments identified: {len(key_moments)}")
    print(f"📸 Screenshots captured: {len(screenshots)}")
    print("=" * 60)

if __name__ == "__main__":
    main()