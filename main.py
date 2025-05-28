from fastapi import FastAPI, Form, Path
from typing import List, Dict, Optional
import openai
import os
import sqlite3
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from gradio.themes.default import Default

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

# Gradio chat function
async def gradio_chat(message: str, chat_history: List[List[str]], user_id: str):
    # Default location (can be made configurable in Gradio UI if needed later)
    location = {
        "country": "IN",
        "city": "Delhi",
        "region": "Delhi"
    }

    # Ensure user_id is provided
    if not user_id:
        # This is a basic way to handle it; Gradio can also make fields required.
        # Returning history with an error message or raising an error.
        # For now, let's assume user_id will be entered.
        # If user_id can be empty, the save_message and get_user_history might fail or misbehave.
        # For robustness, one might add:
        # chat_history.append([message, "Error: User ID is required."])
        # return chat_history
        # However, gr.Textbox can be configured with min_length or similar, or handled in UI logic.
        pass


    # Save user message to DB
    save_message(user_id, "user", message)

    # Construct messages for OpenAI API from Gradio's chat_history
    # Gradio chat_history is List[List[str]] -> [["user", "assistant"], ["user", "assistant"]]
    # OpenAI expects List[Dict[str, str]] -> [{"role": "user", "content": "..."}, ...]
    openai_messages = []
    for user_turn_content, assistant_turn_content in chat_history:
        openai_messages.append({"role": "user", "content": user_turn_content})
        if assistant_turn_content: # assistant_turn_content might be None if it's the latest user message turn
            openai_messages.append({"role": "assistant", "content": assistant_turn_content})
    
    # Add the current user message to the payload for OpenAI
    # Note: `gr.ChatInterface` adds the new message to `chat_history` *before* calling this function.
    # So, `chat_history` already contains the current `message` as the last user message.
    # The `message` parameter is the content of this latest user message.
    # The current construction of openai_messages based on chat_history is correct as it rebuilds from history.

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
            messages=openai_messages, # This should be the full history up to the current user message
            web_search_options=web_search_options,
        )

        assistant_reply = response.choices[0].message.content
        
        # Save assistant reply to DB
        save_message(user_id, "assistant", assistant_reply)

        # Update Gradio's chat_history by appending the new assistant reply
        # The `chat_history` already has the user's latest message.
        # We just need to add the assistant's response to the last turn.
        chat_history[-1][1] = assistant_reply # Update the last turn's assistant message
        return chat_history

    except Exception as e:
        print(f"Error in gradio_chat for user {user_id}: {e}") # Log error
        # Return an error message in the chat.
        # Update the last turn in history with an error message for the assistant.
        chat_history[-1][1] = f"Error: {str(e)}"
        return chat_history

# Create and mount Gradio app
with gr.Blocks(theme=Default()) as demo:
    gr.Markdown("# GPT-4o Chat Interface with Web Search")
    user_id_input = gr.Textbox(label="User ID", placeholder="Enter your User ID (e.g., user123)")
    
    # Note: gr.ChatInterface handles chatbot display and message input internally.
    # The fn (gradio_chat) signature must be: (message, history, additional_input1, ...)
    # So, gradio_chat(message: str, chat_history: List[List[str]], user_id: str) is correct.
    
    gr.ChatInterface(
        fn=gradio_chat,
        additional_inputs=[user_id_input], 
        title="GPT-4o Chat",
        description="Enter your User ID and start chatting. The bot uses web search capabilities.",
        # examples=[ # Optional: provide some examples for users
        #     {"user_id": "example_user_1", "message": "What's the weather in London?"}, # This format needs fn to handle dicts
        # ],
        # For ChatInterface, examples are typically List[str] or List[List[str]] for message & history.
        # If additional_inputs are used, examples get more complex or might not directly map.
        # Let's keep examples commented out for now to ensure core functionality.
        autofocus=True
    )

# Mount the Gradio app to the FastAPI app
# The existing `app = FastAPI()` is the app to mount on.
# The variable `app` will be reassigned by gr.mount_gradio_app.
app = gr.mount_gradio_app(app, demo, path="/gradio")
