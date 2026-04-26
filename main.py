# 바이트 스트림 처리를 위해 io를 불러온다.
import io

# 파일 경로 관리를 위해 Path를 불러온다.
from pathlib import Path

# OpenCV를 사용하기 위해 cv2를 불러온다.
import cv2

# 수치 계산을 위해 numpy를 불러온다.
import numpy as np

# 이미지를 다루기 위해 PIL Image를 불러온다.
from PIL import Image

# PyTorch 본체를 불러온다.
import torch

# softmax 계산을 위해 torch.nn.functional을 불러온다.
import torch.nn.functional as F

# torchvision transforms를 불러온다.
from torchvision import transforms

# FastAPI를 불러온다.
from fastapi import FastAPI

# 업로드 파일 처리를 위해 UploadFile을 불러온다.
from fastapi import UploadFile

# 업로드 파일 필수 입력 처리를 위해 File을 불러온다.
from fastapi import File

# CORS 처리를 위해 CORSMiddleware를 불러온다.
from fastapi.middleware.cors import CORSMiddleware

# 모델 클래스를 model.py에서 불러온다.
from model import PersonalColorNet


# FastAPI 앱 객체를 생성한다.
app = FastAPI()

# CORS를 허용할 주소를 등록한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 현재 파일 기준 디렉토리를 구한다.
BASE_DIR = Path(__file__).resolve().parent

# 모델 가중치 파일 경로를 지정한다.
MODEL_PATH = BASE_DIR / "model.pth"

# 디바이스를 설정한다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 순서를 정의한다.
CLASSES = ["spring", "summer", "autumn", "winter"]

# 한글 클래스명을 정의한다.
CLASS_KR = {
    "spring": "봄 웜톤",
    "summer": "여름 쿨톤",
    "autumn": "가을 웜톤",
    "winter": "겨울 쿨톤",
}

# ImageNet 정규화 평균값을 정의한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]

# ImageNet 정규화 표준편차를 정의한다.
IMAGENET_STD = [0.229, 0.224, 0.225]

# 추론용 이미지 transform을 정의한다.
INFERENCE_TRANSFORM = transforms.Compose([
    # 이미지를 256x256으로 리사이즈한다.
    transforms.Resize((256, 256)),
    # 중앙 224x224 영역을 자른다.
    transforms.CenterCrop(224),
    # PIL 이미지를 텐서로 변환한다.
    transforms.ToTensor(),
    # ImageNet 기준으로 정규화한다.
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 전역 모델 변수를 초기화한다.
model = None


# PIL 이미지 전체에서 평균 LAB 값을 추출하는 함수를 정의한다.
def extract_simple_lab(pil_img):
    # PIL 이미지를 RGB numpy 배열로 변환한다.
    img_np = np.array(pil_img.convert("RGB"))

    # RGB 이미지를 LAB로 변환한다.
    lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 전체 이미지의 평균 LAB 값을 계산한다.
    lab_mean = lab_img.reshape(-1, 3).mean(axis=0)

    # 평균 LAB 값을 float32로 반환한다.
    return lab_mean.astype(np.float32)


# PIL 이미지를 추론용으로 전처리하는 함수를 정의한다.
def preprocess_pil_image(pil_img):
    # 전체 이미지에서 평균 LAB 값을 추출한다.
    blended = extract_simple_lab(pil_img)

    # 원본 이미지와 계산된 LAB를 반환한다.
    return pil_img, None, None, blended


# 서버 시작 시 모델을 한 번 로드하는 함수를 정의한다.
@app.on_event("startup")
def load_model():
    # 전역 모델 변수를 사용한다고 선언한다.
    global model

    # 모델 파일이 없으면 예외를 발생시킨다.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    # 모델 객체를 생성한다.
    model = PersonalColorNet().to(DEVICE)

    # state_dict를 로드한다.
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # 모델에 가중치를 적용한다.
    model.load_state_dict(state_dict)

    # 평가 모드로 전환한다.
    model.eval()


# 서버 상태 확인용 엔드포인트를 정의한다.
@app.get("/health")
def health():
    # 상태를 반환한다.
    return {"status": "ok"}


# 이미지 업로드 후 추론하는 엔드포인트를 정의한다.
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 업로드된 파일의 바이트를 읽는다.
    contents = await file.read()

    # 바이트 데이터를 PIL 이미지로 변환한다.
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # 이미지를 전처리한다.
    face_pil, skin_lab, eye_lab, blended = preprocess_pil_image(pil_img)

    # LAB 값을 얻지 못하면 에러 메시지를 반환한다.
    if blended is None:
        return {"error": "LAB 값을 추출하지 못했습니다."}

    # 얼굴 이미지를 텐서로 변환한다.
    image_tensor = INFERENCE_TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)

    # blended LAB를 텐서로 변환한다.
    lab_tensor = torch.tensor(blended, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # gradient 계산을 끈다.
    with torch.no_grad():
        # 모델 추론을 수행한다.
        logits = model(image_tensor, lab_tensor)

        # softmax 확률을 계산한다.
        probs = F.softmax(logits, dim=1)

        # 최고 확률과 예측 인덱스를 구한다.
        max_prob, pred_idx = torch.max(probs, dim=1)

    # 예측된 클래스명을 구한다.
    pred_class = CLASSES[pred_idx.item()]

    # 결과를 반환한다.
    return {
        "season": pred_class,
        "season_kr": CLASS_KR[pred_class],
        "confidence": round(float(max_prob.item()) * 100, 2),
        "probs": {
            CLASSES[i]: round(float(probs[0][i].item()), 4)
            for i in range(len(CLASSES))
        },
        "lab": {
            "L": round(float(blended[0]), 2),
            "a": round(float(blended[1]), 2),
            "b": round(float(blended[2]), 2),
        },
    }