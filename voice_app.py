import gradio as gr
import numpy as np
import uuid
import time
from fastapi import FastAPI

from fastrtc import Stream, ReplyOnPause, audio_to_bytes, AlgoOptions
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


from config import GROQ_API_KEY, STT_MODEL_NAME, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, TTS_MODEL_NAME
from agent.graph import agent_executor

# --- Khởi tạo Clients ---
groq_client = Groq(api_key=GROQ_API_KEY)
tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def response_generator(text_input: str, thread_id: str):
    """
    Hàm này gọi agent, lấy kết quả và log ra màn hình.
    """
    print(f"💬 [User - {thread_id}]: {text_input}")
    
    # Cấu hình để LangGraph sử dụng đúng luồng hội thoại
    config = {"configurable": {"thread_id": thread_id}}
    
    # Gọi agent để xử lý
    agent_response = agent_executor.invoke(
        {"messages": [("user", text_input)]}, config=config
    )
    
    # Lấy câu trả lời cuối cùng từ agent
    final_response = agent_response["messages"][-1].content
    print(f"🤖 [AI - {thread_id}]: {final_response}")
    
    return final_response


def voice_chat_handler(audio: tuple[int, np.ndarray], thread_id: str | None = None):
    """
    Hàm xử lý chính của FastRTC: nhận audio, chuyển thành text,
    gọi agent, nhận text trả về, chuyển thành audio và stream lại.
    """
    if audio is None:
        return
        
    start_time = time.time()
    
    # Tạo thread_id nếu là lần đầu tiên
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        
    # 1. Speech-to-Text với Groq Whisper
    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", audio_to_bytes(audio)),
        model=STT_MODEL_NAME,
        language="vi" # Chỉ định ngôn ngữ là tiếng Việt
    )
    user_text = transcription.text
    if not user_text.strip():
        print("🎤 Âm thanh trống, bỏ qua.")
        return

    stt_time = time.time() - start_time
    print(f"⏱️ STT time: {stt_time:.2f}s")
    
    # 2. Lấy câu trả lời từ Agent
    response_text = response_generator(user_text, thread_id)
    agent_time = time.time() - start_time - stt_time
    print(f"⏱️ Agent response time: {agent_time:.2f}s")
    # print("Các phương thức có sẵn cho text_to_speech:", dir(tts_client.text_to_speech))
    # 3. Text-to-Speech với ElevenLabs (streaming)
    audio_stream = tts_client.text_to_speech.stream(
        text=response_text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=TTS_MODEL_NAME,
        output_format="pcm_24000",
        # Thêm cài đặt giọng nói để biểu cảm
        voice_settings=VoiceSettings(
        stability=0.6,
        similarity_boost=0.8,
        style=0.3,
        use_speaker_boost=True
    )
        # stream=True
    )

    # 4. Stream audio trả về cho người dùng
    for chunk in audio_stream:
        if chunk:
            # Đảm bảo buffer luôn có số byte chẵn
            if len(chunk) % 2 != 0:
                chunk = chunk[:-1]

            # Kiểm tra lại xem chunk có còn dữ liệu không sau khi cắt
            if not chunk:
                continue

            audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
            yield (24000, audio_array)


# --- Cấu hình FastRTC Stream ---
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(
        voice_chat_handler,
        algo_options=AlgoOptions(
            speech_threshold=0.5,  # Ngưỡng nhận diện giọng nói
            # max_speech_duration=10,  # Thời gian tối đa cho mỗi đoạn nói
        )
        # Thêm một input ẩn để lưu thread_id giữa các lần gọi
    ),additional_inputs=[gr.State(None)],
    # Cài đặt ngưỡng nhận diện giọng nói
    ui_args={"title": "AI Agent Bất Động Sản InGo 🏡"}
)

# --- Tạo ứng dụng FastAPI và gắn giao diện Gradio ---
app = FastAPI()
app = gr.mount_gradio_app(app, stream.ui, path="/")

if __name__ == "__main__":
    import uvicorn
    print("🔥 Khởi chạy ứng dụng tại http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=7860)