from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from config import LLM_MODEL_NAME, SYSTEM_PROMPT
from agent.tools import get_agent_tools
def create_agent():
    """
    Tạo ra một AI agent với LangGraph có khả năng sử dụng công cụ và ghi nhớ hội thoại.
    """
    # 1. Khởi tạo model LLM từ Groq
    model = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=0.7,
        max_tokens=2048
    )

    # 2. Lấy các công cụ đã định nghĩa (ở đây là RAG tool)
    tools = get_agent_tools()

    # 3. Thiết lập bộ nhớ cho hội thoại
    memory = InMemorySaver()

    # 4. Sử dụng hàm create_react_agent để tạo một agent hoàn chỉnh
    # Agent này có khả năng tự suy luận (ReAct) để quyết định khi nào cần dùng tool
    agent_executor = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT, # Sử dụng system prompt đã định nghĩa
        checkpointer=memory,
    )
    
    print("🤖 Agent đã được tạo và sẵn sàng.")
    return agent_executor

# Tạo agent một lần khi ứng dụng khởi động
agent_executor = create_agent()