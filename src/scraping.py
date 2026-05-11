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
    "Luxor": {
        "wiki_title": "Luxor",
        "links": ["https://en.wikipedia.org/wiki/Luxor","https://www.earthtrekkers.com/best-things-to-do-in-luxor-egypt/","https://www.journeyera.com/things-to-do-in-luxor/","https://theblondeabroad.com/ultimate-guide-to-luxor/","https://www.egypttoursportal.com/luxor-tours/","https://www.nomadicmatt.com/travel-guides/egypt-travel-tips/luxor/"]
    },
    "Aswan": {
        "wiki_title": "Aswan",
        "links": ["https://en.wikipedia.org/wiki/Aswan","https://www.jacksonroves.com/blog/aswan-egypt","https://theplanetd.com/things-to-do-in-aswan/","https://www.egypttoursportal.com/aswan-tours/","https://unusualtraveler.com/aswan/","https://www.earthtrekkers.com/best-things-to-do-in-aswan-egypt/"]
    },
    "Alexandria": {
        "wiki_title": "Alexandria",
        "links": ["https://en.wikipedia.org/wiki/Alexandria","https://www.lonelyplanet.com/egypt/alexandria","https://www.egypttoursportal.com/alexandria-tours/","https://www.cairo360.com/city-guide/alexandria/","https://www.travelawaits.com/2556738/best-things-to-do-in-alexandria-egypt/","https://theculturetrip.com/africa/egypt/articles/the-10-best-things-to-see-and-do-in-alexandria-egypt/"]
    },
    "Hurghada": {
        "wiki_title": "Hurghada",
        "links": ["https://en.wikipedia.org/wiki/Hurghada","https://www.thecrowdedplanet.com/things-to-do-in-hurghada/","https://www.journeyera.com/hurghada-egypt/","https://www.egypttoursportal.com/hurghada-excursions/","https://www.divezone.net/diving/red-sea","https://www.saltinourhair.com/egypt/hurghada/"]
    },
    "Sharm El Sheikh": {
        "wiki_title": "Sharm_el-Sheikh",
        "links": ["https://en.wikipedia.org/wiki/Sharm_el-Sheikh","https://wearetravelgirls.com/sharm-el-sheikh/","https://www.nomadicmatt.com/travel-guides/egypt-travel-tips/sharm-el-sheikh/","https://www.egypttoursportal.com/sharm-el-sheikh-excursions/","https://theplanetd.com/sharm-el-sheikh-egypt-things-to-do/","https://www.planetware.com/tourist-attractions-egypt/sharm-el-sheikh-egy-sin-sharm.htm"]
    },
    "Dahab": {
        "wiki_title": "Dahab",
        "links": ["https://en.wikipedia.org/wiki/Dahab","https://www.hostelworld.com/blog/things-to-do-in-dahab/","https://www.egypttoursportal.com/dahab-excursions/","https://www.nomadasaurus.com/things-to-do-in-dahab/","https://www.bucketlistly.blog/posts/dahab-egypt-things-to-do","https://againstthecompass.com/en/travel-dahab-egypt/"]
    },
    "Siwa Oasis": {
        "wiki_title": "Siwa_Oasis",
        "links": ["https://en.wikipedia.org/wiki/Siwa_Oasis","https://againstthecompass.com/en/siwa-oasis-travel-guide/","https://theculturetrip.com/africa/egypt/articles/a-guide-to-egypts-siwa-oasis/","https://handluggageonly.co.uk/2019/02/10/the-ultimate-guide-to-visiting-siwa-oasis-in-egypt/","https://www.egypttoursportal.com/siwa-oasis-tours/","https://cairoscene.com/Travel/The-Ultimate-Guide-to-Siwa-Oasis"]
    },
    "Marsa Alam": {
        "wiki_title": "Marsa_Alam",
        "links": ["https://en.wikipedia.org/wiki/Marsa_Alam","https://www.marsaalam.com/Marsa_Alam_Excursions.html","https://padi.com/dive-site/egypt/marsa-alam/","https://www.egypttoursportal.com/marsa-alam-excursions/","https://divemagazine.com/destinations/egypt/marsa-alam-diving-guide","https://www.theguardian.com/travel/2018/nov/10/egypt-marsa-alam-red-sea-diving-hotels"]
    },
    "Valley of the Kings": {
        "wiki_title": "Valley_of_the_Kings",
        "links": ["https://en.wikipedia.org/wiki/Valley_of_the_Kings","https://www.britannica.com/place/Valley-of-the-Kings","https://www.egypttoursportal.com/valley-of-the-kings/","https://www.nationalgeographic.com/history/article/valley-of-the-kings","https://whc.unesco.org/en/list/87/","https://www.worldhistory.org/Valley_of_the_Kings/"]
    },
    "Karnak Temple": {
        "wiki_title": "Karnak",
        "links": ["https://en.wikipedia.org/wiki/Karnak","https://www.britannica.com/topic/Karnak-temple-complex-Egypt","https://ancientegyptonline.co.uk/karnak/","https://discoveringegypt.com/karnak-temple/","https://www.egypttoursportal.com/karnak-temple/","https://madainproject.com/karnak_temple_complex"]
    }
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

for place, place_data in places.items():
    wiki_title = place_data["wiki_title"]
    links = place_data.get("links", [])
    print(f"Scraping: {place} ...")

    text = get_text(wiki_title)

    if text:
        dataset.append({
            "place": place,
            "city": place,
            "type": "tourism",
            "text": text,
            "links": links,
            # Keep source for backward compatibility with existing consumers.
            "source": links[0] 
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
            "links": item.get("links", []),
            "source": item.get("source", "")
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
