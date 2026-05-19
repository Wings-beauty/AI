import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from model import PersonalColorNet


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_VERSION = "fullface-v1"

CLASSES = ["spring", "summer", "autumn", "winter"]

CLASS_KR = {
    "spring": "봄 웜톤",
    "summer": "여름 쿨톤",
    "autumn": "가을 웜톤",
    "winter": "겨울 쿨톤",
}

SEASON_ATTRIBUTES = {
    "spring": {
        "temperature": "warm",
        "brightness": "light",
        "clarity": "clear",
    },
    "summer": {
        "temperature": "cool",
        "brightness": "light",
        "clarity": "muted",
    },
    "autumn": {
        "temperature": "warm",
        "brightness": "deep",
        "clarity": "muted",
    },
    "winter": {
        "temperature": "cool",
        "brightness": "deep",
        "clarity": "clear",
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

model = None


def extract_simple_lab(pil_img: Image.Image) -> np.ndarray:
    img_np = np.array(pil_img.convert("RGB"))

    lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    lab_mean = lab_img.reshape(-1, 3).mean(axis=0)

    return lab_mean.astype(np.float32)


def preprocess_pil_image(pil_img: Image.Image):
    blended = extract_simple_lab(pil_img)

    return pil_img, None, None, blended


def derive_attributes_from_season(season: str) -> dict:
    return SEASON_ATTRIBUTES.get(
        season,
        {
            "temperature": "unknown",
            "brightness": "unknown",
            "clarity": "unknown",
        },
    )


def get_top2_from_probs(probs_dict: dict) -> tuple:
    sorted_items = sorted(
        probs_dict.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top1_season, top1_prob = sorted_items[0]
    top2_season, top2_prob = sorted_items[1]

    return top1_season, top1_prob, top2_season, top2_prob


def get_question_reason(
    top1_season: str,
    top1_confidence_percent: float,
    top1_top2_gap_percent: float,
) -> tuple[bool, str]:
    if top1_confidence_percent < 50:
        return True, "low_confidence"

    if top1_top2_gap_percent < 12:
        return True, "close_top2"

    if top1_season == "autumn" and top1_confidence_percent < 60:
        return True, "autumn_needs_cool_check"

    return False, "high_confidence"


def build_prediction_response(
    probs_dict: dict,
    lab_values: np.ndarray,
    face_detected: bool = True,
    skin_extract_success: bool = True,
) -> dict:
    top1_season, top1_prob, top2_season, top2_prob = get_top2_from_probs(probs_dict)

    confidence_percent = float(top1_prob) * 100
    top2_confidence_percent = float(top2_prob) * 100
    top1_top2_gap_percent = confidence_percent - top2_confidence_percent

    needs_questions, question_reason = get_question_reason(
        top1_season=top1_season,
        top1_confidence_percent=confidence_percent,
        top1_top2_gap_percent=top1_top2_gap_percent,
    )

    attributes = derive_attributes_from_season(top1_season)

    return {
        "success": True,
        "model_version": MODEL_VERSION,

        # 기존 응답 호환용 필드
        "season": top1_season,
        "season_kr": CLASS_KR[top1_season],
        "confidence": round(confidence_percent, 2),
        "probs": {
            season: round(float(prob), 4)
            for season, prob in probs_dict.items()
        },
        "lab": {
            "L": round(float(lab_values[0]), 2),
            "a": round(float(lab_values[1]), 2),
            "b": round(float(lab_values[2]), 2),
        },

        # 신규 필드
        "top2_season": top2_season,
        "top2_season_kr": CLASS_KR[top2_season],
        "top2_confidence": round(top2_confidence_percent, 2),
        "top1_top2_gap": round(top1_top2_gap_percent, 2),

        "needs_questions": needs_questions,
        "question_reason": question_reason,

        "attributes": attributes,

        "quality": {
            "face_detected": face_detected,
            "skin_extract_success": skin_extract_success,
        },
    }


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    model = PersonalColorNet().to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state_dict)

    model.eval()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    except Exception:
        return {
            "success": False,
            "error": "INVALID_IMAGE",
            "message": "이미지 파일을 열 수 없습니다.",
        }

    try:
        face_pil, skin_lab, eye_lab, blended = preprocess_pil_image(pil_img)

        if blended is None:
            return {
                "success": False,
                "error": "LAB_EXTRACTION_FAILED",
                "message": "LAB 값을 추출하지 못했습니다.",
            }

        image_tensor = INFERENCE_TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)

        lab_tensor = torch.tensor(
            blended,
            dtype=torch.float32,
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image_tensor, lab_tensor)

            probs = F.softmax(logits, dim=1)

        probs_dict = {
            CLASSES[i]: float(probs[0][i].item())
            for i in range(len(CLASSES))
        }

        return build_prediction_response(
            probs_dict=probs_dict,
            lab_values=blended,
            face_detected=True,
            skin_extract_success=True,
        )

    except Exception as e:
        return {
            "success": False,
            "error": "PREDICTION_FAILED",
            "message": "모델 추론 중 오류가 발생했습니다.",
            "detail": str(e),
        }