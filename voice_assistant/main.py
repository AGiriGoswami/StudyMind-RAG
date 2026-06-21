import os
import json
import asyncio
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq
import edge_tts
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Mount static files to serve the frontend
# app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = AsyncGroq(api_key=GROQ_API_KEY)

# Simple in-memory history per session (for MVP, we use one global history or reset it on connect)
chat_history = [
    {"role": "system", "content": "You are a highly helpful, concise, and friendly AI voice assistant. Speak naturally as if in a conversation. Do not use markdown formatting like asterisks or bold text, as your output will be read aloud by a text-to-speech engine. Keep answers brief unless asked for details."}
]

def split_into_sentences(text):
    # Very basic sentence splitter
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s for s in sentences if s.strip()]

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket")
    
    try:
        while True:
            # Wait for audio blob from the client
            data = await websocket.receive_bytes()
            print(f"Received audio chunk of {len(data)} bytes")
            
            # Send status update
            await websocket.send_json({"type": "status", "text": "Transcribing..."})
            
            # Save the received audio to a temporary file for Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                # Transcribe using Groq Whisper
                with open(tmp_path, "rb") as file:
                    transcription = await client.audio.transcriptions.create(
                        file=(tmp_path, file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="text",
                        language="en" # can be removed for auto-detect or Hindi support
                    )
                
                user_text = transcription.strip()
                print(f"User: {user_text}")
                
                if not user_text:
                    await websocket.send_json({"type": "status", "text": "Listening..."})
                    continue
                
                await websocket.send_json({"type": "transcription", "text": user_text})
                await websocket.send_json({"type": "status", "text": "Thinking..."})
                
                # Append to history
                chat_history.append({"role": "user", "content": user_text})
                
                # Stream LLM Response
                completion = await client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=chat_history,
                    stream=True,
                )
                
                full_response = ""
                sentence_buffer = ""
                
                # We need to chunk the text into sentences to stream to TTS
                async for chunk in completion:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        sentence_buffer += token
                        
                        # Send text token to frontend for real-time display
                        await websocket.send_json({"type": "text_stream", "text": token})
                        
                        # If we have a full sentence, generate TTS for it
                        if any(punct in token for punct in [".", "?", "!", "\n"]) and len(sentence_buffer.strip()) > 5:
                            await stream_tts(sentence_buffer.strip(), websocket)
                            sentence_buffer = ""
                
                # Flush the remaining sentence buffer
                if sentence_buffer.strip():
                    await stream_tts(sentence_buffer.strip(), websocket)
                
                # Save assistant response to history
                chat_history.append({"role": "assistant", "content": full_response})
                
                # Tell frontend we are done
                await websocket.send_json({"type": "done", "text": ""})
                await websocket.send_json({"type": "status", "text": "Listening..."})
                
            finally:
                os.unlink(tmp_path)
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({"type": "error", "text": str(e)})
        except:
            pass

async def stream_tts(text, websocket: WebSocket):
    print(f"Generating TTS for: {text}")
    # Using Edge-TTS which is fast and supports Hindi/English
    # "en-US-AriaNeural" for English. Can dynamically choose based on lang.
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural", rate="+10%")
    
    # We will buffer MP3 chunks and send them as binary
    audio_buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.extend(chunk["data"])
            
    # Send the audio binary over WebSocket
    if audio_buffer:
        await websocket.send_bytes(bytes(audio_buffer))
