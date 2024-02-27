import sys
import os

# Add the project directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from utils.embeddings import PhoBertEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import OpenAI 
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo 
from decouple import config


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

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="This is the source documents there are 4 main documents, `document 1`, `document 2`, `document 3`, `document 4`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the details of Python",
        type="integer",
    ),
]

document_content_description = "Info on Python Programming Language"
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=config("OPENAI_API_KEY"))

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

docs = retriever.get_relevant_documents(
    "Hãy nói cho tôi biết ngôn ngữ sử dụng cho ngành khoa học máy tính"
)

print(docs)
