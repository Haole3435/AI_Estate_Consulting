# tools.py

from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from core.rag_retriever import rag_retriever

def get_agent_tools():
    """
    Tạo ra một danh sách các Tool object cho agent sử dụng.
    Bao gồm cả công cụ RAG nội bộ và công cụ tìm kiếm web.
    """
    # Công cụ 1: Tìm kiếm trong tài liệu nội bộ (RAG)
    rag_tool = Tool(
        name="TraCuuThongTinNoiBo",
        func=rag_retriever.invoke if rag_retriever else lambda x: "Không có dữ liệu nội bộ.",
        description="Rất hữu ích để tra cứu thông tin chi tiết về các dự án bất động sản, chính sách bán hàng, giá cả có trong tài liệu nội bộ của công ty In Go."
    )

    # Công cụ 2: Tìm kiếm thông tin pháp lý trên Internet (Deep Search)
    search_tool = TavilySearchResults(
        max_results=3, 
        description="Hữu ích để tìm kiếm thông tin pháp lý, các quy định, luật, nghị định của nhà nước về bất động sản trên internet."
    )
    search_tool.name = "TimKiemThongTinPhapLy"


    tools = [rag_tool, search_tool]
    
    # Chỉ trả về các tool thực sự có thể hoạt động
    if rag_retriever is None:
        tools.pop(0) # Xóa rag_tool nếu không có dữ liệu

    return tools