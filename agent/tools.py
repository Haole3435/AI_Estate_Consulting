from langchain.tools import Tool
from core.rag_retriever import rag_retriever

def get_rag_tool():
    """Tạo ra một Tool object cho agent sử dụng để tra cứu thông tin."""
    if rag_retriever is None:
        return []

    return [
        Tool(
            name="TraCuuThongTinBatDongSan",
            func=rag_retriever.invoke,
            description="Rất hữu ích để tra cứu và trả lời các câu hỏi về thông tin chi tiết của các dự án bất động sản, chính sách bán hàng, giá cả, pháp lý, và các thông tin liên quan khác.",
        )
    ]