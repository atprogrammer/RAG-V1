import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.chains import RetrievalQA

# ตั้งค่า Streamlit
st.title("(prototype)💬 ระบบสอบถามข้อมูลอัจฉริยะ สสจ.ขอนแก่น")
st.subheader("ถามคำถามเกี่ยวกับข้อมูลที่มีอยู่ในระบบ")

# เชื่อมต่อ Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "text_embeddings"

# ใช้โมเดล Embedding จาก Ollama
embedding_model = OllamaEmbeddings(model="bge-m3:latest")

# ใช้ Ollama เป็น LLM
llm = Ollama(model="gemma2:9b")

# ฟังก์ชันค้นหาข้อมูล
def retrieve_top_k(query, k=3):
    try:
        query_embedding = embedding_model.embed_query(query)
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        if not search_result:
            return []
        
        return [Document(page_content=hit.payload["content"]) for hit in search_result]
    except Exception as e:
        st.write(f"เกิดข้อผิดพลาดในการดึงข้อมูลจาก Qdrant: {e}")
        return []

# ฟังก์ชันสร้างคำตอบจาก LLM

def generate_answer_with_llm(query, chat_history):
    query = f"ตอบเป็นภาษาไทย: {query}"
    retrieved_docs = retrieve_top_k(query, k=10)
    if not retrieved_docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล"
    
    vector_store = FAISS.from_documents(retrieved_docs, embedding_model)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    try:
        full_query = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]) + f"\nผู้ใช้: {query}"
        response = qa_chain.run(full_query)
        if not response:
            return "ไม่สามารถสร้างคำตอบจากข้อมูลที่มี"
        return response
    except Exception as e:
        st.write(f"เกิดข้อผิดพลาดในการสร้างคำตอบจาก LLM: {e}")
        return "เกิดข้อผิดพลาดในการสร้างคำตอบ"

# สร้าง UI
if "messages" not in st.session_state: 
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("พิมพ์คำถามของคุณที่นี่...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    answer = generate_answer_with_llm(user_input, st.session_state["messages"])
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
