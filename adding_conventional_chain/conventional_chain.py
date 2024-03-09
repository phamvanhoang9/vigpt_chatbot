from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain.prompts import PromptTemplate 
from langchain.chains import ConversationalRetrievalChain 
from langchain_community.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory
from decouple import config


emdding_function = HuggingFaceBgeEmbeddings(
    model_name="vinai/phobert-base"
)

vector_db = Chroma(
    persist_directory="../vector_db",
    collection_name="hust_info",
    embedding_function=emdding_function,
)

# Create prompt
QA_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the user question.
    chat_history: {chat_history}
    Context: {text}
    Question: {question}
    Answer:""",
    input_variables=["text","question","chat_history"]
)

# create chat model
llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0)

# create memory
memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history"
)

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3},
        search_type='mmr'
    ),
    chain_type="refine",
)

# Question
question = "Đại học Bách Khoa Hà Nội thành lập vào năm nào?"
question2 = "Đại học Bách Khoa Hà Nội có bao nhiêu ngành học?"

# Call QA chain
response = qa_chain({"question": question})
response2 = qa_chain({"question": question2})

print(response.get("answer"))
print("\n")
print(response2.get("answer")) 