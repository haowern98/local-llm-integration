#!/usr/bin/env python3
"""
Active Vision Agent v3.4 (Stable JSON Edition)
- Model: qwen3-vl:8b (User Specified)
- Context: 23,000 tokens
- Fixes: loop/EOF errors via strict generation parameters.
"""

import base64
import io
import time
import requests
import cv2
import sys
import os
import shutil
from typing import TypedDict, List, Optional
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

# ==========================================
# 🔧 USER CONFIGURATION
# ==========================================
OLLAMA_HOST = "http://192.168.0.144:11434"
MODEL_NAME = "qwen3-vl:8b"  # <--- As requested
VIDEO_PATH = r"C:\Users\Wu Family Computer\Downloads\New folder\test.mp4"

# Performance Tuning
CONTEXT_WINDOW = 23000
FRAME_INTERVAL = 2      # 2s between frames
FRAMES_PER_BATCH = 6    # 12s per batch (Small batch = better attention)

# Output
OUTPUT_DIR = Path("video_analysis_report")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. SIMPLE SCHEMA (Reduces Hallucination)
# ==========================================

class FrameCapture(BaseModel):
    frame_index: int
    bbox: List[int]
    label: str

class SmartAnalysis(BaseModel):
    summary: str
    updated_context: str
    captures: List[FrameCapture]

class AgentState(TypedDict):
    video_cap: cv2.VideoCapture
    batch_images: List[Image.Image]
    batch_timestamps: List[float]
    final_html_segments: List[str]
    running_context: str
    batch_count: int

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def normalize_bbox(bbox, w, h):
    return [int(bbox[0]/1000*w), int(bbox[1]/1000*h), int(bbox[2]/1000*w), int(bbox[3]/1000*h)]

def ensure_ollama_ready():
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
        return True
    except: return False

# ==========================================
# 3. AGENT NODES
# ==========================================

def node_input(state: AgentState):
    """Miner: Collects frames."""
    cap = state["video_cap"]
    imgs, times = [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = int(fps * FRAME_INTERVAL)
    
    print(f"\n📥 Batch {state['batch_count'] + 1} Loading...", end="")
    
    for _ in range(FRAMES_PER_BATCH):
        for _ in range(skip): cap.grab()
        ret, frame = cap.retrieve()
        if not ret: break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(frame))
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)
        print(".", end="", flush=True)

    if not imgs: return {"video_cap": None}
    return {"batch_images": imgs, "batch_timestamps": times}

def node_analyze(state: AgentState):
    """Brain: High-Stability Analysis."""
    if not state["batch_images"]: return {}
    
    imgs = state["batch_images"]
    ctx = state.get("running_context", "Start of video.")
    time_lbl = f"{state['batch_timestamps'][0]:.0f}s"
    
    print(f"\n🧠 Analyzing {time_lbl} | Memory: \"{ctx[:50]}...\"")

    payload_imgs = [img_to_b64(i) for i in imgs]
    
    # --- STRICT PROMPT ---
    # We explicitly tell it NOT to loop and keep it short.
    prompt = f"""
    CONTEXT: "{ctx}"
    TASK: Analyze these {len(imgs)} frames (Index 0-{len(imgs)-1}).
    
    REQUIREMENTS:
    1. summary: One concise sentence about the action.
    2. updated_context: Update the story so far.
    3. captures: List 0 or 1 clear code/diagram frames.
    
    Do not repeat text. Keep it under 50 words.
    """

    try:
        res = requests.post(f"{OLLAMA_HOST}/api/chat", json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt, "images": payload_imgs}],
            "format": SmartAnalysis.model_json_schema(),
            "stream": False,
            "options": {
                "num_ctx": CONTEXT_WINDOW,
                "num_predict": 1024,      # Enough space to finish JSON
                "top_p": 0.4,             # Strict sampling (fixes hallucination)
                "repeat_penalty": 1.25,   # Hard ban on loops ("frame frame frame")
                "temperature": 0.1
            }
        }, timeout=180)
        
        response_json = res.json()
        
        if "error" in response_json:
            print(f"\n❌ OLLAMA ERROR: {response_json['error']}")
            return {"analysis_result": None}
            
        content = response_json["message"]["content"]
        return {"analysis_result": SmartAnalysis.model_validate_json(content)}
        
    except Exception as e:
        print(f"\n❌ Analysis Failed: {e}")
        # Return empty result to skip batch instead of crashing
        return {"analysis_result": None}

def node_process(state: AgentState):
    """Scribe: HTML Generation."""
    res = state.get("analysis_result")
    if not res: return {}
    
    imgs = state["batch_images"]
    times = state["batch_timestamps"]
    
    # 1. Process Screenshots
    img_html = ""
    for cap in res.captures:
        if 0 <= cap.frame_index < len(imgs):
            try:
                img = imgs[cap.frame_index]
                bbox = normalize_bbox(cap.bbox, img.width, img.height)
                fname = f"seq_{int(times[cap.frame_index])}s_{cap.label[:10]}.jpg" # Limit label len
                
                img.crop(bbox).save(SCREENSHOT_DIR / fname)
                print(f"  📸 Captured: {fname}")
                
                img_html += f"""
                <div class="img-card">
                    <img src="screenshots/{fname}" onclick="window.open(this.src)">
                    <p>{cap.label}</p>
                </div>
                """
            except: pass

    # 2. Generate HTML
    segment_html = f"""
    <div class="timeline-item">
        <div class="time-badge">{int(times[0])}s</div>
        <div class="content-box">
            <p>{res.summary}</p>
            <div class="gallery">{img_html}</div>
        </div>
    </div>
    """

    return {
        "final_html_segments": [segment_html],
        "running_context": res.updated_context,
        "batch_count": state["batch_count"] + 1
    }

# ==========================================
# 4. GRAPH & HTML
# ==========================================

workflow = StateGraph(AgentState)
workflow.add_node("input", node_input)
workflow.add_node("analyze", node_analyze)
workflow.add_node("process", node_process)

workflow.set_entry_point("input")
workflow.add_conditional_edges("input", lambda s: "end" if not s["video_cap"] else "analyze", {"analyze": "analyze", "end": END})
workflow.add_edge("analyze", "process")
workflow.add_edge("process", "input")

app = workflow.compile()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { background: #121212; color: #ddd; font-family: sans-serif; padding: 20px; }
        .container { max-width: 800px; margin: auto; }
        .timeline-item { display: flex; margin-bottom: 20px; border-left: 2px solid #333; padding-left: 20px; }
        .time-badge { min-width: 60px; color: #4CAF50; font-weight: bold; }
        .content-box { background: #1E1E1E; padding: 15px; border-radius: 5px; flex: 1; }
        .gallery { display: flex; gap: 10px; margin-top: 10px; }
        .img-card { background: #000; padding: 5px; text-align: center; font-size: 0.8em; }
        .img-card img { max-height: 150px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Analysis Report: [VIDEO]</h1>
        [CONTENT]
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    if not ensure_ollama_ready():
        print(f"❌ Cannot connect to {OLLAMA_HOST}")
        sys.exit(1)
        
    print(f"🚀 Started on: {VIDEO_PATH}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Invalid Video Path")
        sys.exit(1)

    final = app.invoke({
        "video_cap": cap, 
        "batch_images": [], 
        "running_context": "Start.", 
        "batch_count": 0, 
        "final_html_segments": []
    }, {"recursion_limit": 5000})

    html = HTML_TEMPLATE.replace("[VIDEO]", os.path.basename(VIDEO_PATH))
    html = html.replace("[CONTENT]", "\n".join(final["final_html_segments"]))
    
    with open(OUTPUT_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"\n✅ Done: {OUTPUT_DIR / 'index.html'}")