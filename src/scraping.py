# !pip install requests beautifulsoup4 pdfplumber sentence-transformers faiss-cpu

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

places = {
    "Luxor": "Luxor",
    "Aswan": "Aswan",
    "Alexandria": "Alexandria",
    "Hurghada": "Hurghada",
    "Sharm El Sheikh": "Sharm_el-Sheikh",
    "Dahab": "Dahab",
    "Siwa Oasis": "Siwa_Oasis",
    "Marsa Alam": "Marsa_Alam",
    "Valley of the Kings": "Valley_of_the_Kings",
    "Karnak Temple": "Karnak"
}

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text(title):
    url = f"https://en.wikipedia.org/wiki/{title}"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error: {title}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")

    text = " ".join([p.get_text() for p in paragraphs])
    return clean_text(text)

dataset = []

for place, wiki_title in places.items():
    print(f"Scraping: {place} ...")

    text = get_text(wiki_title)

    if text:
        dataset.append({
            "place": place,
            "city": place,
            "type": "tourism",
            "text": text,
            "source": f"https://en.wikipedia.org/wiki/{wiki_title}"
        })

        print(f"{place} DONE")

    time.sleep(1)


with open("egypt_places.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("Dataset saved ")

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

chunked_data = []

for item in dataset:
    chunks = chunk_text(item["text"])

    for i, chunk in enumerate(chunks):
        chunked_data.append({
            "place": item["place"],
            "city": item["city"],
            "type": item["type"],
            "chunk_id": i,
            "text": chunk,
            "source": item["source"]
        })


with open("chunked_egypt_places.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=4)

print("Chunking done")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [item["text"] for item in chunked_data]

embeddings = model.encode(texts, show_progress_bar=True)

for i, item in enumerate(chunked_data):
    item["embedding"] = embeddings[i].tolist()


with open("embedded_egypt_places.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=4)

print("Embeddings saved")

embeddings_array = np.array(embeddings).astype("float32")

dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings_array)

print("FAISS index created")


faiss.write_index(index, "faiss_index.index")

with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump(chunked_data, f)

print("FAISS + Metadata saved")
