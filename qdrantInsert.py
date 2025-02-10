from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import ollama
import pandas as pd

# ✅ เชื่อมต่อ Qdrant
client = QdrantClient(host="localhost", port=6333)

# ✅ กำหนดชื่อ Collection
collection_name = "text_embeddings"

# ✅ ตรวจสอบว่ามี Collection หรือไม่ ถ้าไม่มีให้สร้างใหม่
if collection_name not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

# ✅ ใช้โมเดลผ่าน Ollama
model_name = "bge-m3:latest"

# ✅ อ่านข้อมูลจากไฟล์ CSV โดยเลือกเฉพาะคอลัมน์ "text"
file_path = "chunked_output.csv"  # กำหนดพาธไฟล์ CSV
df = pd.read_csv(file_path, usecols=["text"], dtype={"text": str})  # ใช้เฉพาะคอลัมน์ "text"

# ✅ ตรวจสอบว่ามีข้อมูลหรือไม่
if df.empty:
    raise ValueError("📌 ไฟล์ CSV ไม่มีข้อมูลในคอลัมน์ 'text'")

texts = df["text"].dropna().tolist()  # ลบค่า NaN และแปลงเป็น List

# ✅ แปลงข้อความเป็นเวกเตอร์โดยใช้ Ollama (แก้ไขตรงนี้!)
embeddings = []
for text in texts:
    response = ollama.embeddings(model=model_name, prompt=text)
    embeddings.append(response["embedding"])  # ดึงเฉพาะค่าเวกเตอร์

# ✅ เพิ่มข้อมูลลงใน Qdrant
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"content": texts[i]})
    for i in range(len(texts))
]
client.upsert(collection_name=collection_name, points=points)

print("✅ อัพเดทข้อมูลสำเร็จ!")
