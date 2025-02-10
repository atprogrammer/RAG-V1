import os
import csv
import fitz  # PyMuPDF

def read_pdf_with_fitz(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def chunk_by_length(text, max_length=512, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

pdf_folder = "pdf_files"
csv_output = "chunked_output.csv"
data_rows = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        print(f"📖 กำลังอ่านไฟล์: {filename}")
        text = read_pdf_with_fitz(file_path)

        chunks = chunk_by_length(text, max_length=512, overlap=100)
        for i, chunk in enumerate(chunks):
            chunk_clean = chunk.replace('"', '""')
            data_rows.append([filename, i + 1, chunk_clean])

with open(csv_output, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    writer.writerow(["filename", "chunk_id", "text"])
    writer.writerows(data_rows)

print(f"✅ บันทึกไฟล์ CSV เรียบร้อย: {csv_output}")
