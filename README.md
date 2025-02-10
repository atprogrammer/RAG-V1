# วิธีการเรียกใช้งาน

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

### **สรุปการทำงานของโค้ด**  ไฟล์
- อ่านไฟล์ PDF ทั้งหมดจากโฟลเดอร์ `pdf_files`  
- แปลงเนื้อหา PDF เป็นข้อความ  
- แบ่งข้อความออกเป็นส่วนย่อยๆ (chunk) ความยาว 512 ตัวอักษร พร้อม overlap 100 ตัวอักษร  
- บันทึกข้อมูลลงไฟล์ CSV โดยมีฟิลด์ `filename`, `chunk_id`, `text`  
- ใช้งานได้ง่าย เพียงนำไฟล์ PDF ไปไว้ในโฟลเดอร์ที่กำหนด และรันโค้ด