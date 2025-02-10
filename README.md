### อธิบายการใช้งานโค้ดทีละขั้นตอน ไฟล์ ChunksPDF.py

1. **นำเข้าไลบรารีที่จำเป็น**  
   ```python
   import os
   import csv
   import fitz  # PyMuPDF
   ```
   - `os` ใช้สำหรับจัดการไฟล์และโฟลเดอร์  
   - `csv` ใช้สำหรับเขียนไฟล์ CSV  
   - `fitz` (PyMuPDF) ใช้สำหรับอ่านไฟล์ PDF  

2. **สร้างฟังก์ชันสำหรับอ่านไฟล์ PDF**  
   ```python
   def read_pdf_with_fitz(file_path):
       text = ""
       with fitz.open(file_path) as doc:
           for page in doc:
               text += page.get_text("text") + "\n"
       return text.strip()
   ```
   - ฟังก์ชันนี้รับพาธของไฟล์ PDF แล้วอ่านข้อความจากทุกหน้าของ PDF  
   - ใช้ `fitz.open(file_path)` เพื่อเปิดไฟล์  
   - วนลูปดึงข้อความจากแต่ละหน้าโดยใช้ `page.get_text("text")`  
   - นำข้อความทั้งหมดมาต่อกันและคืนค่าออกมา  

3. **ฟังก์ชันแบ่งข้อความเป็นส่วนย่อย (Chunking)**  
   ```python
   def chunk_by_length(text, max_length=512, overlap=100):
       chunks = []
       start = 0
       while start < len(text):
           end = min(start + max_length, len(text))
           chunks.append(text[start:end])
           start += max_length - overlap
       return chunks
   ```
   - ฟังก์ชันนี้ใช้สำหรับแบ่งข้อความที่อ่านจาก PDF ออกเป็นส่วนๆ  
   - แต่ละส่วนมีความยาวไม่เกิน `max_length` (ค่าเริ่มต้นคือ 512 ตัวอักษร)  
   - มีการทับซ้อน (`overlap`) ระหว่างส่วนละ 100 ตัวอักษรเพื่อให้ข้อความต่อเนื่องกัน  
   - ใช้ `while` ลูปเพื่อตัดข้อความจาก `start` ถึง `end` และเพิ่มเข้าไปใน `chunks`  

4. **กำหนดโฟลเดอร์และไฟล์ผลลัพธ์**  
   ```python
   pdf_folder = "pdf_files"
   csv_output = "chunked_output.csv"
   data_rows = []
   ```
   - กำหนดชื่อโฟลเดอร์ที่เก็บไฟล์ PDF (`pdf_files`)  
   - กำหนดชื่อไฟล์ CSV ที่ใช้บันทึกผล (`chunked_output.csv`)  
   - สร้างลิสต์ `data_rows` เพื่อเก็บข้อมูลที่จะแปลงเป็น CSV  

5. **วนลูปอ่านไฟล์ PDF ทั้งหมดในโฟลเดอร์**  
   ```python
   for filename in os.listdir(pdf_folder):
       if filename.endswith(".pdf"):
           file_path = os.path.join(pdf_folder, filename)
           print(f"📖 กำลังอ่านไฟล์: {filename}")
           text = read_pdf_with_fitz(file_path)
   ```
   - ใช้ `os.listdir(pdf_folder)` เพื่อดึงรายชื่อไฟล์ทั้งหมดในโฟลเดอร์  
   - ตรวจสอบว่าไฟล์ลงท้ายด้วย `.pdf` เพื่อกรองเฉพาะไฟล์ PDF  
   - แปลงชื่อไฟล์ให้เป็นพาธเต็ม (`file_path`)  
   - แสดงข้อความแจ้งว่ากำลังอ่านไฟล์  

6. **แบ่งข้อความและเก็บข้อมูลลงในลิสต์**  
   ```python
   chunks = chunk_by_length(text, max_length=512, overlap=100)
   for i, chunk in enumerate(chunks):
       chunk_clean = chunk.replace('"', '""')
       data_rows.append([filename, i + 1, chunk_clean])
   ```
   - เรียก `chunk_by_length()` เพื่อแบ่งข้อความจากไฟล์ PDF ออกเป็นหลายส่วน  
   - วนลูปบันทึกแต่ละส่วนลงใน `data_rows` พร้อมระบุหมายเลขลำดับ (`chunk_id`)  
   - ใช้ `chunk.replace('"', '""')` เพื่อป้องกันปัญหาการใช้เครื่องหมาย `"` ใน CSV  

7. **บันทึกข้อมูลลงไฟล์ CSV**  
   ```python
   with open(csv_output, mode="w", encoding="utf-8", newline="") as file:
       writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
       writer.writerow(["filename", "chunk_id", "text"])
       writer.writerows(data_rows)
   ```
   - เปิดไฟล์ CSV ในโหมดเขียน (`"w"`) และกำหนด `encoding="utf-8"`  
   - ใช้ `csv.writer()` เพื่อสร้างออบเจ็กต์สำหรับเขียนไฟล์ CSV  
   - เขียนหัวข้อคอลัมน์ `["filename", "chunk_id", "text"]`  
   - ใช้ `writer.writerows(data_rows)` บันทึกข้อมูลทั้งหมดที่เก็บไว้ใน `data_rows`  

8. **แสดงข้อความแจ้งเมื่อบันทึกเสร็จ**  
   ```python
   print(f"✅ บันทึกไฟล์ CSV เรียบร้อย: {csv_output}")
   ```
   - แสดงข้อความแจ้งเตือนว่าการบันทึกเสร็จสมบูรณ์  

### **สรุปการทำงานของโค้ด**  ไฟล์ qdrantInsert.py
- อ่านไฟล์ PDF ทั้งหมดจากโฟลเดอร์ `pdf_files`  
- แปลงเนื้อหา PDF เป็นข้อความ  
- แบ่งข้อความออกเป็นส่วนย่อยๆ (chunk) ความยาว 512 ตัวอักษร พร้อม overlap 100 ตัวอักษร  
- บันทึกข้อมูลลงไฟล์ CSV โดยมีฟิลด์ `filename`, `chunk_id`, `text`  
- ใช้งานได้ง่าย เพียงนำไฟล์ PDF ไปไว้ในโฟลเดอร์ที่กำหนด และรันโค้ด

------------------------------------------------------------------------
### อธิบายการใช้งานโค้ดทีละขั้นตอน ไฟล์ qdrantInsert.py
โค้ดนี้มีเป้าหมายหลักคือ **นำข้อความจาก CSV ไปแปลงเป็นเวกเตอร์ฝังตัว (Embeddings) และบันทึกลง Qdrant** เพื่อใช้สำหรับการสืบค้นแบบเวกเตอร์ (Vector Search) โดยใช้โมเดล `bge-m3` ผ่าน **Ollama**  
   
### **📌 คำอธิบายการทำงานของโค้ด**
---

### **1️⃣ เชื่อมต่อ Qdrant**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
```
- นำเข้าไลบรารี `qdrant_client` สำหรับเชื่อมต่อและจัดการข้อมูลใน Qdrant  

```python
client = QdrantClient(host="localhost", port=6333)
```
- เชื่อมต่อ Qdrant ที่รันอยู่บนเครื่องที่ `localhost` และพอร์ต `6333`  

---

### **2️⃣ ตรวจสอบและสร้าง Collection**
```python
collection_name = "text_embeddings"
```
- กำหนดชื่อ Collection ที่จะใช้เก็บข้อมูลเวกเตอร์  

```python
if collection_name not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
```
- ตรวจสอบว่ามี Collection ชื่อ `"text_embeddings"` หรือไม่  
- ถ้าไม่มี จะสร้าง Collection ใหม่ โดยกำหนด  
  - `size=1024` → กำหนดขนาดของเวกเตอร์ (ต้องตรงกับขนาดเวกเตอร์ที่ได้จากโมเดล)  
  - `distance=Distance.COSINE` → ใช้ **Cosine Similarity** ในการเปรียบเทียบเวกเตอร์  

---

### **3️⃣ โหลดโมเดล Embeddings ผ่าน Ollama**
```python
import ollama
model_name = "bge-m3:latest"
```
- ใช้โมเดล `bge-m3:latest` ผ่าน Ollama เพื่อแปลงข้อความเป็นเวกเตอร์  

---

### **4️⃣ อ่านข้อมูลจาก CSV**
```python
import pandas as pd

file_path = "chunked_output.csv"
df = pd.read_csv(file_path, usecols=["text"], dtype={"text": str})
```
- อ่านข้อมูลจากไฟล์ `chunked_output.csv`  
- เลือกใช้เฉพาะคอลัมน์ `"text"`  

```python
if df.empty:
    raise ValueError("📌 ไฟล์ CSV ไม่มีข้อมูลในคอลัมน์ 'text'")
```
- ถ้าคอลัมน์ `"text"` ไม่มีข้อมูล จะหยุดโปรแกรมและแจ้งเตือน  

```python
texts = df["text"].dropna().tolist()
```
- ลบค่า `NaN` ออกจากข้อมูล  
- แปลงคอลัมน์ `"text"` เป็น **List**  

---

### **5️⃣ แปลงข้อความเป็นเวกเตอร์**
```python
embeddings = []
for text in texts:
    response = ollama.embeddings(model=model_name, prompt=text)
    embeddings.append(response["embedding"])
```
- วนลูป **นำข้อความแต่ละรายการไปแปลงเป็นเวกเตอร์** โดยใช้โมเดล `bge-m3`  
- เก็บเวกเตอร์ที่ได้ลงใน `embeddings`  

---

### **6️⃣ เพิ่มข้อมูลลงใน Qdrant**
```python
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"content": texts[i]})
    for i in range(len(texts))
]
client.upsert(collection_name=collection_name, points=points)
```
- สร้างรายการข้อมูลที่มี  
  - `id=i` → กำหนด ID ของข้อมูล  
  - `vector=embeddings[i]` → ใส่เวกเตอร์ที่ได้จาก Ollama  
  - `payload={"content": texts[i]}` → ใส่ข้อความเดิมเป็น Payload  
- ใช้ `.upsert()` เพื่อเพิ่มหรืออัปเดตข้อมูลลง Qdrant  

---

### **7️⃣ แสดงข้อความแจ้งเตือน**
```python
print("✅ อัพเดทข้อมูลสำเร็จ!")
```
- แสดงข้อความเมื่ออัปโหลดเสร็จสมบูรณ์  

---

### **📌 บทสรุปการทำงาน**
✅ **โหลดข้อความจาก CSV**  
✅ **แปลงข้อความเป็นเวกเตอร์ด้วย Ollama**  
✅ **บันทึกเวกเตอร์ลง Qdrant เพื่อใช้งาน Vector Search**  


--------------------------------------------------------------------------------------
### อธิบายการใช้งานโค้ดทีละขั้นตอน ไฟล์ streamlitAppUseQdV2(UseHistory).py
โค้ดนี้เป็น **Chatbot สำหรับค้นหาข้อมูลและตอบคำถามอัจฉริยะ** โดยใช้ **Streamlit, Qdrant และ Ollama** โดยสามารถทำงานได้ดังนี้  

1. **รับคำถามจากผู้ใช้ผ่าน UI ของ Streamlit**  
2. **ค้นหาข้อมูลที่เกี่ยวข้องจาก Qdrant**  
3. **ใช้ FAISS และ Ollama สร้างคำตอบจากข้อมูลที่ค้นพบ**  
4. **แสดงผลลัพธ์ในรูปแบบแชทโต้ตอบ**  

---

## **📌 คำอธิบายโค้ด**
---

### **1️⃣ นำเข้าไลบรารีที่จำเป็น**
```python
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.chains import RetrievalQA
```
- **`streamlit`** → ใช้สร้าง UI  
- **`qdrant_client`** → ใช้จัดการฐานข้อมูลเวกเตอร์ Qdrant  
- **`FAISS`** → ใช้สร้างดัชนีเวกเตอร์สำหรับค้นหาข้อมูล  
- **`OllamaEmbeddings`** → ใช้สร้างเวกเตอร์จากข้อความ  
- **`Ollama`** → ใช้สร้างคำตอบจากโมเดลภาษาขนาดใหญ่ (LLM)  
- **`RetrievalQA`** → ใช้ LLM ตอบคำถามโดยอ้างอิงข้อมูล  

---

### **2️⃣ ตั้งค่าหน้าเว็บ Streamlit**
```python
st.title("(prototype)💬 ระบบสอบถามข้อมูลอัจฉริยะ สสจ.ขอนแก่น")
st.subheader("ถามคำถามเกี่ยวกับข้อมูลที่มีอยู่ในระบบ")
```
- สร้าง **Title** และ **Subtitle** สำหรับ UI  

---

### **3️⃣ เชื่อมต่อ Qdrant**
```python
client = QdrantClient(host="localhost", port=6333)
collection_name = "text_embeddings"
```
- เชื่อมต่อกับ Qdrant **(ต้องแน่ใจว่า Qdrant ทำงานอยู่)**  

---

### **4️⃣ ตั้งค่าโมเดล Embedding และ LLM**
```python
embedding_model = OllamaEmbeddings(model="bge-m3:latest")
llm = Ollama(model="gemma2:9b")
```
- ใช้ **`bge-m3:latest`** ในการแปลงข้อความเป็นเวกเตอร์  
- ใช้ **`gemma2:9b`** เป็นโมเดลสร้างคำตอบ  

---

### **5️⃣ ฟังก์ชันดึงข้อมูลจาก Qdrant**
```python
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
```
✅ **ทำงาน**:  
- แปลงข้อความ **`query`** เป็นเวกเตอร์  
- ค้นหาข้อมูลที่ใกล้เคียงที่สุด **`k` รายการ** จาก Qdrant  
- คืนค่าผลลัพธ์เป็น **`Document`** เพื่อให้ LLM ใช้งาน  

---

### **6️⃣ ฟังก์ชันสร้างคำตอบจาก LLM**
```python
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
```
✅ **ทำงาน**:  
- ค้นหาข้อมูลจาก Qdrant  
- ใช้ **FAISS** สร้างดัชนีจากข้อมูลที่ค้นพบ  
- ใช้ **LLM** สร้างคำตอบโดยอ้างอิงข้อมูลที่ดึงมา  
- รวม **ข้อความประวัติการแชท** เพื่อให้ LLM เข้าใจบริบท  

---

### **7️⃣ สร้าง UI แชทใน Streamlit**
```python
if "messages" not in st.session_state: 
    st.session_state["messages"] = []
```
- ใช้ **`st.session_state`** เก็บประวัติการแชท  

```python
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
```
- วนลูปแสดงข้อความที่เคยส่งไปแล้ว  

```python
user_input = st.chat_input("พิมพ์คำถามของคุณที่นี่...")
```
- **รับข้อความจากผู้ใช้**  

```python
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
```
- บันทึก **ข้อความของผู้ใช้** และแสดงใน UI  

```python
answer = generate_answer_with_llm(user_input, st.session_state["messages"])
st.session_state["messages"].append({"role": "assistant", "content": answer})
with st.chat_message("assistant"):
    st.markdown(answer)
```
- **เรียกใช้ฟังก์ชัน `generate_answer_with_llm`**  
- **บันทึกและแสดงผลลัพธ์จาก LLM**  

---

## **📌 บทสรุปการทำงาน**
✅ **สร้าง UI แชทบอทด้วย Streamlit**  
✅ **เชื่อมต่อ Qdrant เพื่อค้นหาข้อมูล**  
✅ **ใช้ Ollama แปลงข้อความเป็นเวกเตอร์**  
✅ **ใช้ LLM (Gemma2:9B) ตอบคำถาม**  

---




