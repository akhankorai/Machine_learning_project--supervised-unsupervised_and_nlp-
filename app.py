from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pickle
import io
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware



from preprocessing import (
    clean_resume_text,
    remove_stopwords,
    resume_stemming
)
with open("resume_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("resume_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("resume_label_mapping.pkl", "rb") as f:
    label_map = pickle.load(f)

app = FastAPI(title="Resume Classifier API")


def preprocess(text: str) -> str:
    text = clean_resume_text(text)
    text = remove_stopwords(text)
    text = resume_stemming(text)
    return text

def predict_category(text: str) -> str:
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    pred_num = model.predict(vec)[0]
    return label_map[pred_num]


class Resume(BaseModel):
    text: str

app = FastAPI(title="Resume Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],         
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Resume Classifier API is running"}

@app.post("/predict")
def predict(resume: Resume):
    if not resume.text or len(resume.text.split()) < 5:
        return {"prediction": "Text too short, please provide more resume content."}

    prediction = predict_category(resume.text)
    return {"prediction": prediction}

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    # Check file type
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or plain text file."
        )

    content = await file.read()

 
    extracted_text = ""

    if file.content_type == "text/plain":
        extracted_text = content.decode("utf-8", errors="ignore")

    elif file.content_type == "application/pdf":
        try:
            pdf_reader = PdfReader(io.BytesIO(content))
            text_chunks = []
            for page in pdf_reader.pages:
                text_chunks.append(page.extract_text() or "")
            extracted_text = "\n".join(text_chunks)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Failed to read PDF file."
            )

    if not extracted_text or len(extracted_text.split()) < 5:
        return {"prediction": "File content too short or unreadable. Please upload a proper resume."}

    prediction = predict_category(extracted_text)
    return {
        "prediction": prediction,
        "filename": file.filename,
        "chars_extracted": len(extracted_text)
    }
