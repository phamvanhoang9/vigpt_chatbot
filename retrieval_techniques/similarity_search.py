from embeddings import PhoBertEmbeddings
from langchain.vectorstores import Chroma


TEXT = ["Python là ngôn ngữ lập trình linh hoạt và được sử dụng rộng rãi, nổi tiếng với cú pháp rõ ràng và dễ đọc, dựa vào thụt lề cho cấu trúc mã",
         "Đây là ngôn ngữ có mục đích chung phù hợp để phát triển web, phân tích dữ liệu, AI, học máy và tự động hóa. Python cung cấp một thư viện tiêu chuẩn mở rộng với các mô-đun bao gồm nhiều nhiệm vụ, giúp nhà phát triển có hiệu quả.",
         "Nó đa nền tảng, chạy trên Windows, macOS, Linux, v.v., cho phép khả năng tương thích ứng dụng rộng rãi."
         "Python có một cộng đồng lớn và năng động phát triển thư viện, cung cấp tài liệu và hỗ trợ cho người mới.",
         "Nó đặc biệt trở nên phổ biến trong khoa học dữ liệu và học máy nhờ tính dễ sử dụng cũng như sự sẵn có của các thư viện và khung công tác mạnh mẽ."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

embedding_function = PhoBertEmbeddings(
    model_name="vinai/phobert-base"
)

vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

response = vector_db.similarity_search(
    query="Hãy nói cho tôi biết ngôn ngữ sử dụng cho ngành khoa học máy tính", k=2)

print(response)