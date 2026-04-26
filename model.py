# PyTorch를 사용하기 위해 torch를 불러온다.
import torch

# PyTorch 신경망 모듈을 사용하기 위해 nn을 불러온다.
import torch.nn as nn

# ResNet18 백본을 사용하기 위해 torchvision models를 불러온다.
from torchvision import models


# 이미지와 LAB feature를 함께 사용하는 모델 클래스를 정의한다.
class PersonalColorNet(nn.Module):
    # 생성자를 정의한다.
    def __init__(self):
        # 부모 클래스를 초기화한다.
        super().__init__()

        # ResNet18 백본을 생성한다.
        backbone = models.resnet18(weights=None)

        # 사전학습 가중치 구조에 맞게 fc를 4클래스로 맞춘다.
        backbone.fc = nn.Linear(backbone.fc.in_features, 4)

        # 마지막 fc를 제거하고 feature extractor만 남긴다.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # LAB 3차원 벡터를 처리할 MLP 브랜치를 정의한다.
        self.lab_branch = nn.Sequential(
            # 3차원 LAB를 64차원으로 확장한다.
            nn.Linear(3, 64),
            # 비선형성을 추가한다.
            nn.ReLU(),
            # 64차원을 32차원으로 줄인다.
            nn.Linear(64, 32),
            # 비선형성을 추가한다.
            nn.ReLU(),
        )

        # 이미지 특징과 LAB 특징을 합친 뒤 분류할 classifier를 정의한다.
        self.classifier = nn.Sequential(
            # 과적합 방지를 위해 dropout을 적용한다.
            nn.Dropout(0.4),
            # 512 + 32 차원을 256차원으로 줄인다.
            nn.Linear(512 + 32, 256),
            # 비선형성을 추가한다.
            nn.ReLU(),
            # 추가 dropout을 적용한다.
            nn.Dropout(0.2),
            # 최종 4개 클래스 logits를 출력한다.
            nn.Linear(256, 4),
        )

    # 순전파 함수를 정의한다.
    def forward(self, image, lab):
        # 이미지에서 backbone feature를 추출한다.
        img_feat = self.backbone(image)

        # backbone feature를 1차원 벡터로 펼친다.
        img_feat = img_feat.flatten(1)

        # LAB feature를 MLP에 통과시킨다.
        lab_feat = self.lab_branch(lab)

        # 이미지 feature와 LAB feature를 이어붙인다.
        fused = torch.cat([img_feat, lab_feat], dim=1)

        # 최종 logits를 반환한다.
        return self.classifier(fused)