import os
import io
import base64
import json
import httpx
import edge_tts
import yt_dlp
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="YouSEO Studio AI Backend")

# Frontend (Vercel/GitHub Pages) से कनेक्शन के लिए CORS चालू
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


async def call_gemini(prompt: str, image_bytes: bytes = None, mime_type: str = None) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set on server.")

    parts = [{"text": prompt}]
    if image_bytes and mime_type:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": b64_img
            }
        })

    payload = {"contents": [{"parts": parts}]}
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload)
        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=f"Gemini Error: {res.text}")
        data = res.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise HTTPException(status_code=500, detail="Unexpected AI response format.")


# 1. Health Check (UptimeRobot के लिए ताकि सर्वर 24/7 चालू रहे)
@app.get("/")
def home():
    return {"status": "online", "service": "YouSEO Studio AI Backend"}


# 2. All-in-One YouSEO AI Tools + Auto-Pilot 100% SEO Pack
class ToolRequest(BaseModel):
    tool: str  # trending_ideas, viral_ideas, keywords, script, hook, title, description, community_post, ai_coach, auto_seo_pack
    topic: str
    language: str = "Hindi"
    extra_info: str = ""


TOOL_PROMPTS = {
    "trending_ideas": "Generate 10 trending, low-competition YouTube video ideas for the niche/topic: '{topic}'. Language: {language}. Include estimated search demand (High/Medium) and why it will trend.",
    "viral_ideas": "Generate 10 high-CTR viral YouTube video concepts for '{topic}' in {language}. Give each a catchy angle and hook idea.",
    "keywords": "Act as a YouTube SEO expert. Generate 20 high-volume, low-competition YouTube keywords and long-tail tags for '{topic}' in {language}. Group them into Primary, Secondary, and Long-Tail.",
    "script": "Write a complete, engaging YouTube video script on '{topic}' in {language}. Structure: 0-15s Viral Hook, Intro, Main Points with retention triggers, and Call-to-Action (CTA). Extra context: {extra_info}",
    "hook": "Write 7 scroll-stopping 3-second viral YouTube hooks for '{topic}' in {language} so viewers don't drop off.",
    "title": "Generate 10 high-CTR, SEO-optimized YouTube titles (under 65 characters) for '{topic}' in {language}.",
    "description": "Write a fully SEO-optimized YouTube video description for '{topic}' in {language}. Include hook lines, keyword-rich summary, timestamps placeholder, and 15 relevant hashtags.",
    "community_post": "Create 3 engaging YouTube Community tab posts (1 poll, 1 question, 1 teaser) about '{topic}' in {language}.",
    "ai_coach": "You are an expert YouTube Growth Coach. Answer this creator's question with practical, actionable steps in {language}: '{topic}'. Context: {extra_info}",
    "auto_seo_pack": """Act as a 100% YouTube SEO Automation Engine. For the video topic '{topic}' in {language}, return ONLY valid JSON (no markdown fences) with these exact keys:
    {{
      "seo_score": 98,
      "title": "Best high-CTR SEO title under 65 chars",
      "description": "Full SEO description with keywords and call to action",
      "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10"],
      "hashtags": ["#tag1", "#tag2", "#tag3"],
      "best_time_to_post": "Best day and IST time slot for maximum initial views with short reason"
    }}"""
}


@app.post("/api/ai-tool")
async def run_ai_tool(req: ToolRequest):
    template = TOOL_PROMPTS.get(req.tool)
    if not template:
        raise HTTPException(status_code=400, detail="Invalid tool name.")
    prompt = template.format(topic=req.topic, language=req.language, extra_info=req.extra_info)
    result = await call_gemini(prompt)
    return {"tool": req.tool, "result": result}


# 3. Thumbnail Click Score (CTR Analyzer using Gemini Vision)
@app.post("/api/thumbnail-score")
async def analyze_thumbnail(file: UploadFile = File(...), title: str = Form("")):
    img_bytes = await file.read()
    prompt = f"""Analyze this YouTube thumbnail for a video titled '{title}'.
    Give:
    1. Click Score (CTR Potential) out of 100
    2. Visual Clarity & Contrast rating
    3. Text Readability on Mobile
    4. 3 specific improvements to increase clicks.
    Keep it concise and actionable."""
    result = await call_gemini(prompt, image_bytes=img_bytes, mime_type=file.content_type)
    return {"analysis": result}


# 4. Free & Long-Lasting AI Voiceover (Microsoft Edge Neural TTS - No API Key Needed)
class VoiceRequest(BaseModel):
    text: str
    voice: str = "hi-IN-MadhurNeural"  # Options: hi-IN-MadhurNeural, hi-IN-SwaraNeural, en-IN-PrabhatNeural, en-IN-NeerjaNeural
    rate: str = "+0%"


@app.post("/api/voiceover")
async def generate_voiceover(req: VoiceRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        communicate = edge_tts.Communicate(req.text, req.voice, rate=req.rate)
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=voiceover.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voiceover error: {str(e)}")


# 5. YouTube Video Extractor & SEO Analyzer (Tags, Views, Likes, Score)
class ExtractRequest(BaseModel):
    url: str


@app.post("/api/extract-video")
def extract_video(req: ExtractRequest):
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=False)
            title = info.get("title", "")
            tags = info.get("tags") or []
            desc = info.get("description", "")

            # Basic SEO Score Calculation
            score = 40
            if 30 <= len(title) <= 70: score += 20
            if len(tags) >= 8: score += 20
            if len(desc) >= 200: score += 20

            return {
                "title": title,
                "channel": info.get("uploader", ""),
                "views": info.get("view_count", 0),
                "likes": info.get("like_count", 0),
                "duration_sec": info.get("duration", 0),
                "thumbnail": info.get("thumbnail", ""),
                "tags": tags,
                "description": desc[:500],
                "seo_score": min(score, 100)
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not extract video: {str(e)}")
