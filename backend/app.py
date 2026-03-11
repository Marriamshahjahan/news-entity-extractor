from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from collections import defaultdict
import re
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ner = pipeline(
    "ner",
    model="Babelscape/wikineural-multilingual-ner",
    aggregation_strategy="simple"
)

@app.get("/")
def home():
    return {"message": "News Entity Extraction API running"}

def extract_entities(text):

    results = ner(text)
    entities = defaultdict(set)

    label_map = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location"
    }

    for r in results:
        if r["entity_group"] in label_map:
            label = label_map[r["entity_group"]]
            word = text[r["start"]:r["end"]].strip()
            entities[label].add(word)

    dates = re.findall(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2}\s\d{4}\b',
        text
    )

    for d in dates:
        entities["Date"].add(d)

    money = re.findall(
        r'[$₹]\s?\d+(?:\.\d+)?\s?(?:million|billion|lakh|crore)?',
        text
    )

    for m in money:
        entities["Money"].add(m)

    perc = re.findall(r'\b\d+%', text)

    for p in perc:
        entities["Percentage"].add(p)

    output = ""

    for label, words in entities.items():
        if words:
            output += f"{label}:\n"
            for w in sorted(words):
                output += f"- {w}\n"
            output += "\n"

    return output if output else "No entities found."

@app.post("/extract")
def extract(data: dict):

    text = data.get("text", "").strip()

    if not text:
        return {"result": "Please provide some text."}

    result = extract_entities(text)

    return {"result": result}
