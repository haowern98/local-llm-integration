"""
tools.py - ROI detection and screenshot capture tools for the lecture analysis system.
"""

import base64
import io
import time
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import requests

class KeyMomentAnalyzer:
    """Identifies key moments in the lecture that need visual citations."""
    
    def __init__(self):
        # Use triple braces {{{ and }}} to escape JSON curly braces in format strings
        self.roi_prompt_template = """You are analyzing a frame from a lecture at timestamp {timestamp:.1f}s.

**Context from Comprehensive Summary:**
{summary_context}

**Frame Description:**
{frame_description}

**Your Task:**
Identify the SINGLE MOST IMPORTANT region in this frame that would serve as visual evidence for the lecture content. This region should be directly relevant to the key concepts discussed around this timestamp.

**Region Types (choose one):**
- `code_block`: Programming code with syntax highlighting
- `diagram`: Charts, graphs, flowcharts, system diagrams
- `formula`: Mathematical equations and formulas
- `ui`: User interface elements, buttons, screens
- `text`: Important text blocks, annotations, or bullet points

**Selection Criteria:**
✅ **DO capture regions that:**
- Contain central concepts discussed in the summary
- Show complex content that's hard to describe textually
- Were specifically emphasized or referenced by the lecturer
- Support claims made in the comprehensive summary

❌ **DO NOT capture regions that:**
- Are empty or contain only background elements
- Show generic lecture interface (title bars, menus)
- Are too small to be readable when captured
- Don't add significant value beyond the text summary

**Output Format (STRICT JSON):**
{{
    "coordinates": [x1, y1, x2, y2],
    "region_type": "code_block",
    "region_name": "descriptive_name_for_file",
    "caption": "Professional caption describing what this screenshot shows and why it matters",
    "relevance_score": 0.95
}}

**Coordinate System:**
- Image dimensions: {width}x{height} pixels
- (0,0) is top-left corner, ({width},{height}) is bottom-right
- Coordinates must be within image bounds
- ROI should be large enough to be readable (minimum 200x150 pixels)
"""

    def identify_key_moments(self, full_summary, frames, timestamps, ollama_host, model_name):
        """Identify key moments that need visual citations from the comprehensive summary."""
        
        # Create summary context for the prompt
        summary_context = f"""
### Executive Summary
{full_summary.get('executive_summary', '')}

### Major Topics
{full_summary.get('major_topics', '')}

### Content Evolution
{full_summary.get('content_evolution', '')}
"""
        
        key_moments = full_summary.get('key_moments', [])
        
        if not key_moments:
            print("⚠️  No key moments identified in summary. Using default timestamps.")
            # Fallback to evenly distributed timestamps
            total_duration = timestamps[-1] if timestamps else len(frames) * 2
            key_moments = [
                {'timestamp': total_duration * 0.25, 'description': 'Early lecture content', 'reason': 'Quarter point reference'},
                {'timestamp': total_duration * 0.5, 'description': 'Mid-lecture content', 'reason': 'Midpoint reference'},
                {'timestamp': total_duration * 0.75, 'description': 'Late lecture content', 'reason': 'Three-quarter point reference'}
            ]
        
        identified_moments = []
        
        for moment in key_moments:
            timestamp = moment['timestamp']
            print(f"🔍 Analyzing moment at {timestamp:.1f}s: {moment['description']}")
            
            # Find the closest frame to this timestamp
            frame_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - timestamp))
            frame_timestamp = timestamps[frame_index]
            frame = frames[frame_index]
            
            # Get frame description first
            frame_base64 = self._image_to_base64(frame)
            frame_description_prompt = f"Describe the key visual elements in this lecture frame at {frame_timestamp:.1f}s. Focus on content that would be important for understanding the lecture concepts."
            
            frame_result = self._send_to_ollama([frame_base64], frame_description_prompt, ollama_host, model_name)
            frame_description = frame_result['response']
            
            # Now get ROI coordinates with context
            # Fixed: Use proper escaping for JSON braces and correct format parameters
            try:
                roi_prompt = self.roi_prompt_template.format(
                    timestamp=frame_timestamp,
                    summary_context=summary_context[:1000],  # Limit context length
                    frame_description=frame_description[:500],
                    width=frame.width,
                    height=frame.height
                )
            except KeyError as e:
                print(f"❌ Format error: {e}. Using fallback prompt.")
                # Fallback prompt without complex formatting
                roi_prompt = f"""Analyze frame at {frame_timestamp:.1f}s and identify important region coordinates [x1,y1,x2,y2]. Return JSON with coordinates, region_type, region_name, caption, relevance_score."""
            
            roi_result = self._send_to_ollama([frame_base64], roi_prompt, ollama_host, model_name)
            
            try:
                # Try to parse as JSON, with fallback to extract JSON from text
                response_text = roi_result['response']
                
                # Extract JSON from response if it's wrapped in text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_str = response_text[json_start:json_end]
                    roi_data = json.loads(json_str)
                else:
                    # Fallback JSON if parsing fails
                    roi_data = {
                        "coordinates": [100, 100, 300, 300],
                        "region_type": "text",
                        "region_name": "fallback_roi",
                        "caption": "Fallback ROI due to JSON parsing error",
                        "relevance_score": 0.3
                    }
                
                # Validate coordinates
                coords = roi_data.get('coordinates', [0, 0, frame.width, frame.height])
                if self._validate_coordinates(coords, frame.width, frame.height):
                    identified_moments.append({
                        'timestamp': frame_timestamp,
                        'frame_index': frame_index,
                        'description': moment['description'],
                        'reason': moment.get('reason', 'Key moment from summary'),
                        'roi_data': roi_data,
                        'frame_description': frame_description,
                        'relevance_score': roi_data.get('relevance_score', 0.5)
                    })
                    print(f"✅ ROI identified: {roi_data.get('region_type')} - {roi_data.get('caption')[:50]}...")
                else:
                    print(f"❌ Invalid coordinates: {coords}. Using fallback ROI.")
                    identified_moments.append({
                        'timestamp': frame_timestamp,
                        'frame_index': frame_index,
                        'description': moment['description'],
                        'reason': 'Fallback ROI due to invalid coordinates',
                        'roi_data': self._get_fallback_roi(frame.width, frame.height),
                        'frame_description': frame_description,
                        'relevance_score': 0.3
                    })
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}. Using fallback ROI.")
                identified_moments.append({
                    'timestamp': frame_timestamp,
                    'frame_index': frame_index,
                    'description': moment['description'],
                    'reason': 'Fallback ROI due to JSON error',
                    'roi_data': self._get_fallback_roi(frame.width, frame.height),
                    'frame_description': frame_description,
                    'relevance_score': 0.2
                })
            except Exception as e:
                print(f"❌ Error processing moment: {e}. Using fallback ROI.")
                identified_moments.append({
                    'timestamp': frame_timestamp,
                    'frame_index': frame_index,
                    'description': moment['description'],
                    'reason': f'Fallback ROI due to error: {str(e)}',
                    'roi_data': self._get_fallback_roi(frame.width, frame.height),
                    'frame_description': frame_description,
                    'relevance_score': 0.1
                })
        
        # Sort by relevance score and limit to top 5
        identified_moments.sort(key=lambda x: x['relevance_score'], reverse=True)
        return identified_moments[:5]
    
    def _validate_coordinates(self, coords, width, height):
        """Validate that coordinates are within bounds and reasonable size."""
        if len(coords) != 4:
            return False
        
        x1, y1, x2, y2 = coords
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        
        # Check size (minimum 200x150 pixels)
        if (x2 - x1) < 200 or (y2 - y1) < 150:
            return False
        
        # Check proportions (not too narrow or tall)
        aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        return True
    
    def _get_fallback_roi(self, width, height):
        """Get a reasonable fallback ROI when analysis fails."""
        # Center 60% of the frame
        margin_x = width * 0.2
        margin_y = height * 0.2
        
        return {
            "coordinates": [int(margin_x), int(margin_y), int(width - margin_x), int(height - margin_y)],
            "region_type": "text",
            "region_name": "fallback_content",
            "caption": "Fallback screenshot of lecture content (ROI detection failed)",
            "relevance_score": 0.1
        }
    
    def _image_to_base64(self, image):
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _send_to_ollama(self, images_base64, prompt, ollama_host, model_name):
        """Send image(s) to Ollama's vision model for analysis."""
        url = f"{ollama_host}/api/generate"
        
        payload = {
            "model": model_name,
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
            return {'response': f"Error: Cannot connect to Ollama at {ollama_host}", 'time': 0, 'context': 'N/A'}
        except requests.exceptions.RequestException as e:
            return {'response': f"Error communicating with Ollama: {str(e)}", 'time': 0, 'context': 'N/A'}

class ScreenshotCapturer:
    """Captures screenshots of specific regions at key moments."""
    
    def __init__(self, screenshots_dir):
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True, parents=True)
        print(f"📁 Screenshots directory: {self.screenshots_dir.absolute()}")
    
    def capture_screenshots(self, key_moments, frames, timestamps, frame_interval):
        """Capture screenshots for all key moments."""
        screenshots = []
        timestamp_map = {ts: i for i, ts in enumerate(timestamps)}
        
        for moment in key_moments:
            timestamp = moment['timestamp']
            frame_index = moment['frame_index']
            roi_data = moment['roi_data']
            
            if frame_index >= len(frames):
                print(f"❌ Frame index {frame_index} out of bounds for timestamp {timestamp:.1f}s")
                continue
            
            frame = frames[frame_index]
            coords = roi_data['coordinates']
            region_type = roi_data['region_type']
            region_name = roi_data['region_name']
            caption = roi_data['caption']
            
            print(f"📸 Capturing {region_type} at {timestamp:.1f}s: {caption[:50]}...")
            
            try:
                # Crop the frame to the ROI
                x1, y1, x2, y2 = coords
                cropped_frame = frame.crop((x1, y1, x2, y2))
                
                # Generate filename with timestamp and region type
                safe_region_name = ''.join(c for c in region_name if c.isalnum() or c in ('_', '-')).lower()
                filename = f"{safe_region_name}_{int(timestamp)}s_{int(time.time())}.png"
                filepath = self.screenshots_dir / filename
                
                # Save the cropped screenshot
                cropped_frame.save(filepath, quality=95, optimize=True)
                
                # Convert to base64 for potential analysis (optional)
                with open(filepath, "rb") as img_file:
                    base64_string = base64.b64encode(img_file.read()).decode('utf-8')
                
                screenshots.append({
                    'path': str(filepath.absolute()),
                    'relative_path': str(filepath.relative_to(self.screenshots_dir.parent)),
                    'caption': caption,
                    'timestamp': timestamp,
                    'coordinates': coords,
                    'region_type': region_type,
                    'region_name': region_name,
                    'base64': base64_string[:100] + '...',  # Truncate for storage
                    'moment_description': moment['description'],
                    'relevance_score': moment['relevance_score']
                })
                
                print(f"✅ Screenshot saved: {filepath.name}")
                
            except Exception as e:
                print(f"❌ Error capturing screenshot: {e}")
                # Fallback: save the entire frame
                fallback_filename = f"fallback_{int(timestamp)}s_{int(time.time())}.png"
                fallback_filepath = self.screenshots_dir / fallback_filename
                frame.save(fallback_filepath)
                screenshots.append({
                    'path': str(fallback_filepath.absolute()),
                    'relative_path': str(fallback_filepath.relative_to(self.screenshots_dir.parent)),
                    'caption': f"Fallback screenshot at {timestamp:.1f}s (ROI capture failed)",
                    'timestamp': timestamp,
                    'coordinates': [0, 0, frame.width, frame.height],
                    'region_type': 'fallback',
                    'region_name': 'fallback_full_frame',
                    'base64': '',  # Skip base64 for fallback
                    'moment_description': moment['description'],
                    'relevance_score': moment['relevance_score'] * 0.5
                })
        
        return screenshots
    
    def _enhance_screenshot_quality(self, image):
        """Enhance screenshot quality for better readability."""
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(img_cv, -1, kernel)
        
        # Convert back to PIL Image
        enhanced = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        
        return enhanced