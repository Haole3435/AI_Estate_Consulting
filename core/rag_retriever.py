from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import DATA_DIRECTORY

def create_rag_retriever():
    """
    Tải dữ liệu từ các file .txt, chia nhỏ, tạo vector embeddings và lưu vào FAISS.
    Trả về một retriever object để tìm kiếm thông tin.
    """
    print("🚀 Bắt đầu thiết lập RAG retriever...")
    
    # 1. Tải tài liệu từ thư mục /data
    loader = DirectoryLoader(DATA_DIRECTORY, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    if not docs:
        print("⚠️ Cảnh báo: Không tìm thấy tài liệu nào trong thư mục 'data'. Agent sẽ không thể tra cứu thông tin.")
        return None

    print(f"✅ Đã tải {len(docs)} tài liệu.")

    # 2. Chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"✅ Đã chia tài liệu thành {len(splits)} đoạn nhỏ.")

    # 3. Tạo embeddings (sử dụng model đa ngôn ngữ tốt)
    # 'vinai/phobert-base-v2' là một lựa chọn tốt cho tiếng Việt
    print("🧠 Đang tạo embeddings... (Quá trình này có thể mất vài phút lần đầu tiên)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Tạo Vector Store với FAISS
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("✅ Đã tạo xong Vector Store bằng FAISS.")

    # 5. Trả về retriever
    return vectorstore.as_retriever(search_kwargs={'k': 3}) # Lấy 3 kết quả liên quan nhất

# Tạo retriever một lần khi khởi động ứng dụng
rag_retriever = create_rag_retriever()