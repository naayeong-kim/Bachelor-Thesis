# Usability of CNNs and Attention Mechanisms for The Diagnosis of Melanoma

악성 흑색종 (이하 Melanoma)는 서구 국가, 특히 미국에서 전체 악성 종양의 약 2%를 차지하며 매년 9,000명 이상이 사망하는 질병입니다. 일반적으로 피부 병변은 육안으로 정확하게 진단하기 어렵지만 초기 단계에서 정확하게 탐지해낼 수 있다면 추가 진단을 위한 불필요한 시간과 비용을 줄이는 데 큰 도움이 됩니다. 본 연구에서는 주어진 피부 병변 이미지 데이터에 대해 피부암 분류 문제 해결을 위한 딥러닝 기반 CNN을 활용한 솔루션을 제안합니다. <br/>
전처리를 통해 주어진 데이터 셋이 가진 클래스 불균형 문제를 해결하고, 백본 아키텍처 모델을 선택하여 해당 모델의 진단 작업에서의 성능을 알아봅니다. 이후 더 나아가서 기존 모델에 새로운 딥러닝 기법인 'Attention mechanisms'을 적용하여, 대체된 레이어를 갖는 모델 아키텍처가 더 나은 성능을 갖는지 확인합니다.<br/><br/>


이 프로젝트는 Google Colab의 Python 환경에서 진행되었으며, Pytorch 프레임워크의 라이브러리 활용을 위해 GPU or TPU 환경에서의 CUDA 사용이 필수적입니다. 더 자세한 분석 방법 및 결과는 다음과 같은 추가 코드 파일 및 리포트에서 확인하실 수 있습니다.<br/> 
- [(해당 연구에 대한 code 보러가기)](project.ipynb) <br/>
- [(해당 연구에 대한 full report 보러가기)](report.pdf) <br/> 


## Database

본 연구에서는 ISIC에서 제공하는 2020 공개 피부 병변 이미지 데이터를 사용하였습니다. 해당 데이터를 병변 부위를 기준으로 정사각형으로 자른 뒤 128x128 크기로 조정된 이미지 셋을 활용하였습니다. 데이터에는 환자의 피부 병변 이미지뿐만 아니라 점의 위치, 환자의 나이 및 성별 등의 메타데이터도 포함되어 있습니다. 그 중 이 프로젝트에서 가장 초점을 맞춘 부분은 malignant/benign 의 이진 분류 형식으로 나타나는 실제 melanoma 여부 입니다.<br/> 

## Pre-Processing

### Imbalanced Class Problem

### Data Augmentation


## Model

### AlexNet

### AllConvNet

## Attention Mechanisms on AllConvNet

