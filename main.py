from fastapi import FastAPI, Form, Path
from typing import List, Dict, Optional
import openai
import os
import sqlite3
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "chat_history.db"
MODEL_NAME = "gpt-4o-search-preview"

# Initialize SQLite table
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        
        """)

init_db()

# Save message to DB
def save_message(user_id: str, role: str, content: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )

# Load all prior messages for user
def get_user_history(user_id: str) -> List[Dict[str, str]]:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id ASC",
            (user_id,)
        )
        return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

@app.post("/chat/")
async def chat(
    user_id: str = Form(...),
    
    message: str = Form(...),
    country: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    region: Optional[str] = Form(None)
):
    # Default location
    location = {
        "country": country or "IN",
        "city": city or "Delhi",
        "region": region or "Delhi"
    }

    # Save user message
    save_message(user_id, "user", message)

    # Load full chat history
    messages = get_user_history(user_id)

    # Build web search options
    web_search_options = {
        "search_context_size": "medium",
        "user_location": {
            "type": "approximate",
            "approximate": location
        }
    }

    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            web_search_options=web_search_options,
        )

        reply = response.choices[0].message.content
        annotations = getattr(response.choices[0].message, "annotations", [])

        # Save assistant reply
        save_message(user_id, "assistant", reply)

        return {
            "reply": reply,
            "annotations": annotations
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/history/{user_id}")
async def get_history(user_id: str = Path(...)):
    """
    Returns the chat history for a specific user.
    """
    history = get_user_history(user_id)
    return {"user_id": user_id, "history": history}

# Mount the static directory to serve JS/CSS if needed
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()