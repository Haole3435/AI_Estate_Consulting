# chat_app.py

import gradio as gr
import uuid
from fastapi import FastAPI

from config import GROQ_API_KEY, STT_MODEL_NAME, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, TTS_MODEL_NAME
from agent.graph import agent_executor

# --- Các hàm xử lý ---
def response_generator(text_input: str, thread_id: str):
    if not text_input.strip(): return ""
    print(f"💬 [User - {thread_id}]: {text_input}")
    config = {"configurable": {"thread_id": thread_id}}
    agent_response = agent_executor.invoke({"messages": [("user", text_input)]}, config=config)
    final_response = agent_response["messages"][-1].content
    print(f"🤖 [AI - {thread_id}]: {final_response}")
    return final_response

def text_chat_handler(message: str, history: list, thread_id: str | None = None):
    if thread_id is None: thread_id = str(uuid.uuid4())
    response = response_generator(message, thread_id)
    return response, thread_id

# --- Giao diện Chat Văn bản ---
with gr.Blocks(theme=gr.themes.Soft()) as chat_app_ui:
    gr.Markdown("# AI Agent - Chatbot văn bản 💬")
    
    thread_id_state = gr.State(value=str(uuid.uuid4()))
    chatbot = gr.Chatbot(height=550, label="Hội thoại", type='messages')
    
    with gr.Row():
        msg_textbox = gr.Textbox(
            placeholder="Nhập câu hỏi của bạn và nhấn Enter...",
            container=False, scale=7, label="Tin nhắn"
        )
        submit_button = gr.Button("Gửi", variant="primary", scale=1)

    def text_submit_handler(message_text, chat_history_messages, thread_id):
        if not message_text.strip():
            return "", chat_history_messages, thread_id
        
        chat_history_messages.append({"role": "user", "content": message_text})
        response_text, updated_thread_id = text_chat_handler(message_text, chat_history_messages, thread_id)
        chat_history_messages.append({"role": "assistant", "content": response_text})
        return "", chat_history_messages, updated_thread_id

    submit_args = {
        "fn": text_submit_handler,
        "inputs": [msg_textbox, chatbot, thread_id_state],
        "outputs": [msg_textbox, chatbot, thread_id_state]
    }
    msg_textbox.submit(**submit_args)
    submit_button.click(**submit_args)

# --- Tạo ứng dụng FastAPI ---
app = FastAPI(title="Chat Agent")
app = gr.mount_gradio_app(app, chat_app_ui, path="/")

if __name__ == "__main__":
    import uvicorn
    print("🔥 Khởi chạy ứng dụng CHAT VĂN BẢN tại http://127.0.0.1:7861")
    # Chạy ở một cổng khác để tránh xung đột
    uvicorn.run(app, host="127.0.0.1", port=7861)