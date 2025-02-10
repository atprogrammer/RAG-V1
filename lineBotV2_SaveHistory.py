import asyncio
import os
import psycopg2
from fastapi import FastAPI, Request, HTTPException
from qdrant_client import QdrantClient
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from linebot.v3.messaging import MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from asgiref.sync import async_to_sync

# โหลด environment variables
load_dotenv()

# ตั้งค่า FastAPI
app = FastAPI()

# ตั้งค่า Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "text_embeddings"

# ตั้งค่า Ollama
embedding_model = OllamaEmbeddings(model="bge-m3:latest")
llm = Ollama(model="gemma2:9b", system="กรุณาตอบเป็นภาษาไทยเสมอ")


# ตรวจสอบ LINE Bot Credentials
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("Missing LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET")

# ตั้งค่า LINE Bot
config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_bot_api = MessagingApi(ApiClient(config))
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ตั้งค่า PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL").replace("postgresql+asyncpg://", "postgresql://")
conn = psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)
cursor = conn.cursor()

# ฟังก์ชันบันทึกแชทลง PostgreSQL
def save_chat(user_id, message, is_bot=False):
    if isinstance(message, dict):  # ✅ แปลง dict เป็น string
        message = str(message)
    
    cursor.execute(
        "INSERT INTO chat_history (user_id, message, is_bot) VALUES (%s, %s, %s)",
        (user_id, message, is_bot)
    )
    conn.commit()

# ฟังก์ชันดึงประวัติแชทล่าสุด
def get_chat_history(user_id, limit=5):
    cursor.execute(
        "SELECT message FROM chat_history WHERE user_id = %s ORDER BY id DESC LIMIT %s",
        (user_id, limit),
    )
    return [row["message"] for row in cursor.fetchall()][::-1]

# ฟังก์ชันค้นหาข้อมูลจาก Qdrant
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
        print(f"Error retrieving from Qdrant: {e}")
        return []

# ฟังก์ชันสร้างคำตอบจาก LLM
async def generate_answer_with_llm(query: str, user_id: str):
    query = f"ตอบเป็นภาษาไทยเท่านั้น: {query}"
    retrieved_docs = retrieve_top_k(query, k=5)
    
    chat_history = get_chat_history(user_id)
    chat_context = [("user", msg) for msg in chat_history]  # ✅ ปรับให้เป็น List[Tuple[str, str]]
    
    if not retrieved_docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล"
    
    vector_store = FAISS.from_documents(retrieved_docs, embedding_model)
    retriever = vector_store.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    
    try:
        response = qa_chain.invoke({"question": query, "chat_history": chat_context})  # ✅ ใช้ invoke() แทน run()
        if isinstance(response, dict):  # ✅ ป้องกัน dict response
            response = response.get("answer", "ไม่สามารถสร้างคำตอบจากข้อมูลที่มี")
        return response
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "เกิดข้อผิดพลาดในการสร้างคำตอบ"

@app.post("/webhook")
async def line_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature")

    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature")

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return {"message": "OK"}

# ฟังก์ชันจัดการเหตุการณ์ข้อความ
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_input = event.message.text
    user_id = event.source.user_id  # ✅ ดึง user_id ของผู้ใช้
    reply_token = event.reply_token

    # ✅ ส่ง user_id ไปที่ process_message
    asyncio.create_task(process_message(user_input, reply_token, user_id))

# ฟังก์ชันประมวลผลข้อความแยกต่างหาก
async def process_message(user_input: str, reply_token: str, user_id: str):
    # ✅ บันทึกข้อความของผู้ใช้ก่อน
    save_chat(user_id, user_input, is_bot=False)

    answer = await generate_answer_with_llm(user_input, user_id)

    # ✅ บันทึกข้อความของบอท
    save_chat(user_id, answer, is_bot=True)

    # ✅ เอา await ออก เพราะ reply_message() เป็น synchronous function
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=answer)]
        )
    )