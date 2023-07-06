from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertModel, BertTokenizer
import numpy as np
import torch
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from kobert_transformers import get_tokenizer
import re
from collections import Counter
from ITglossary2 import ITGlossary, ITGlossaryUpdater
import pymysql
import pandas as pd
from keybert import KeyBERT
from kiwipiepy import Kiwi
import urllib.request
from soyspacing.countbase import RuleDict, CountSpace
import json
from decimal import Decimal
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://www.ohmystack.co"],  # 클라이언트의 도메인 주소를 여기에 추가
    allow_methods=["POST"],  # 요청 허용 메서드
    allow_headers=["*"],  # 요청 허용 헤더
)

# # Load the job data
# conn = pymysql.connect(host="database-1.cb6dvhektjhd.ap-northeast-2.rds.amazonaws.com", user='gihun', password='EM7E7e', db='production')

@app.on_event("startup")
def load_data():
    global df
    global glossary
    global kw_model
    global model
    global tokenizer

    # query = "SELECT id, embedding FROM production.cosine"
    # with conn.cursor() as cursor:
    #     df = pd.read_sql_query(query, conn)

    df = pd.read_csv('embedding2.csv', dtype=str)
    
    glossary = ITGlossary()
    updater = ITGlossaryUpdater(glossary)
    updater.update_glossary()
    glossary.print_glossary()

    model = BertModel.from_pretrained('monologg/kobert')
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

    kw_model = KeyBERT(model)

@app.on_event("shutdown")
# def close_connection():
#     conn.close()

# Define request/response models
class JobRecommendationRequest(BaseModel):
    self_intr: str

class JobRecommendationResponse(BaseModel):
    job_recommendations: list

# Define utility functions
def clean_text(input_text):
    if not isinstance(input_text, str):
        return input_text
    input_text = input_text.lower()
    # Remove special characters
    cleaned_text = re.sub(r'[^\w\s()]', '', input_text)
    # Remove newline characters
    cleaned_text = cleaned_text.replace('\n', '')
    # Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    # Remove URLs
    cleaned_text = re.sub(r'http\S+|www\S+', '', cleaned_text)
    return cleaned_text

def remove_english(text):
    pattern = re.compile(r'[a-zA-Z]+')
    text_without_english = re.sub(pattern, '', text)
    return text_without_english

def translate_IT(input_text, it_glossary):
    input_text = str(input_text)
    words = re.findall(r'\b[a-zA-Z]+\b', input_text)
    for word in words:
        try:
            input_text = input_text.replace(word, it_glossary[word])
        except:
            pass
        input_text = input_text.replace('[', '')
        input_text = input_text.replace(']', '')
    return input_text

def get_embedding(kw_model, keywords, self_intr):
    embeddings = []  # 임베딩 리스트
    weights = []  # 가중치 리스트
    for keyword in keywords:
        if len(keyword) > 1:  # 키워드의 길이가 1보다 큰지 확인
            weights.append(keyword[1])  # 가중치를 리스트에서 가져옴
            input_ids = tokenizer.encode(keyword[0], add_special_tokens=True)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]
                embedding = last_hidden_states[0].mean(dim=0).numpy()
                embeddings.append(embedding)
    if not weights or all(weight == 0 for weight in weights):
        print("가중치 Null값이 발생하였습니다.")
        return None
    return np.average(embeddings, axis=0, weights=weights)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return str(obj)
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


@app.post("/job_recommendation", response_model=JobRecommendationResponse)
def job_recommendation(request: JobRecommendationRequest):
    self_intr = request.self_intr
    self_intr = translate_IT(clean_text(self_intr), glossary.glossary)
    self_intr = remove_english(self_intr)
    self_intr = clean_text(self_intr)

    keywords = kw_model.extract_keywords(self_intr, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=10)
    input_embedding = get_embedding(kw_model, keywords, self_intr)

    embeddings = np.array(df['embedding'].apply(process_embedding).tolist())

    cos_sim = cosine_similarity([input_embedding], embeddings).ravel()
    cos_sim[np.isnan(cos_sim)] = 0

    # top_indices = np.argpartition(cos_sim, -50)[-50:]  # 상위 50개 추천을 선택
    top_indices = np.argsort(cos_sim)[::-1]

    job_recommendations = df.iloc[top_indices]['id'].astype(int).tolist()  # 정수형으로 변환하여 리스트로 저장

    response_data = {"job_recommendations": job_recommendations}
    response_content = json.dumps(response_data)

    return JSONResponse(content=response_content, media_type="application/json")


def process_embedding(s):
    if isinstance(s, float):
        return np.zeros((768,), dtype=float)

    s = re.sub(r'[\[\]\n]', '', s)
    arr = np.fromstring(s, sep=' ')
    arr[np.isnan(arr)] = 0  # NaN 값을 0으로 대체
    arr[np.isinf(arr)] = 0  # Infinity 값을 0으로 대체
    arr = arr.reshape((768,))
    
    return arr
