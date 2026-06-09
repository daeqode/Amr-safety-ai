# BEM Robotics Vision ACS

물류 로봇 장애물 인식 + LLM 기반 관제 AI 시스템

AMR/AGV 카메라 영상에서 장애물을 실시간 감지하고, LLM이 안전 규정을 참조해
위험도를 판단한 뒤 관제 시스템(ACS)에 자연어 경고 리포트를 자동 생성하는
end-to-end 파이프라인입니다.

---

## 프로젝트 개요

물류 현장에서 AMR/AGV가 주행 중 카메라로 장애물을 인식하고,
LLM이 사내 안전 규정(RAG)을 참조해 위험도를 판단한 뒤
관제 시스템(ACS)에 자연어 경고 리포트를 자동 생성합니다.

벰로보틱스 제품 구조 반영

| 이 프로젝트                        | 벰로보틱스 제품                      |
|------------------------------------|--------------------------------------|
| YOLOv8 객체 인식 + 거리 추정       | PCS (위치·제어 솔루션)               |
| LLM + RAG 기반 상황 판단           | ACS (관제 솔루션) AI화 방향          |
| FastAPI + WebSocket 대시보드        | ACS 관제 화면                        |
| 온프레미스 Ollama 로컬 LLM         | 공장 내 인터넷 단절 환경 대응        |

---

## 시스템 흐름

    카메라/이미지 입력
           |
           v
    YOLOv8 객체 감지 (person, truck, car ...)
    MPS 가속 (Apple Silicon 최적화)
           |
           v
    ContextParser
    바운딩박스 -> 위치/거리/위험도 텍스트 변환
    risk_score 계산 (SAFE / CAUTION / WARNING / CRITICAL)
           |
           v
    RAG Pipeline + ChromaDB
    물류 안전 규정 벡터 DB similarity search
           |
           v
    LLM Client
    Ollama llama3.2:3b (로컬 온프레미스)
    OpenAI gpt-4o-mini (멀티 LLM 지원)
           |
           v
    FastAPI + WebSocket
    ACS 스타일 관제 대시보드 실시간 푸시

---

## 실행 방법

1. 사전 요구사항
   - Python 3.11+
   - Ollama 설치 (https://ollama.ai)
   - Apple Silicon (MPS) 또는 CUDA GPU 권장

2. 설치

       git clone https://github.com/YOUR_USERNAME/bemrobotics-vision.git
       cd bemrobotics-vision
       python -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt

3. Ollama 모델 준비

       ollama pull llama3.2:3b

4. 서버 실행

       python src/main.py

5. 대시보드 접속

       http://localhost:8000

---

## 멀티 LLM 백엔드 지원

llm_client.py의 LLMClient 클래스는 provider 파라미터로 LLM 백엔드를 교체할 수 있습니다.

    # 로컬 온프레미스 (인터넷 불필요)
    client = LLMClient(provider="ollama", model="llama3.2:3b")

    # 클라우드 API
    client = LLMClient(provider="openai", model="gpt-4o-mini")

공장 내 보안 네트워크 환경에서는 Ollama 로컬 모드로,
개발/테스트 환경에서는 OpenAI API로 즉시 전환 가능합니다.

---

## API 엔드포인트

| Method | Endpoint  | 설명                        |
|--------|-----------|-----------------------------|
| GET    | /         | ACS 대시보드                |
| POST   | /analyze  | 이미지 분석 요청            |
| GET    | /alerts   | 경고 이력 조회 (최근 50개)  |
| GET    | /status   | 시스템 상태 확인            |
| WS     | /ws       | 실시간 경고 푸시            |

---

## 기술 스택

| 분류       | 기술                                  |
|------------|---------------------------------------|
| 객체 인식  | YOLOv8n (Ultralytics)                 |
| 영상 처리  | OpenCV                                |
| LLM        | Ollama llama3.2:3b / OpenAI gpt-4o-mini |
| RAG        | LangChain + ChromaDB                  |
| 백엔드     | FastAPI + WebSocket                   |
| 가속       | Apple MPS (M1/M2/M3)                  |

---

## 향후 확장 계획

ROS2 연동
- 현재 파이프라인을 ROS2 노드로 래핑하면 실제 AMR/AGV에 직접 탑재 가능
- /camera/image 토픽 -> Vision ACS 노드 -> /cmd_vel 제어 명령

LiDAR 융합
- 현재: 단안 카메라 바운딩박스 기반 거리 추정
- 확장: 3D LiDAR 포인트클라우드와 카메라 융합으로 정밀 거리 측정
- 구현: detector.py의 estimate_distance()를 LiDAR depth 데이터로 교체

---

## 프로젝트 구조

    bemrobotics-vision/
    ├── src/
    │   ├── detector.py        # YOLOv8 객체 인식
    │   ├── context_parser.py  # 인식 결과 -> 컨텍스트 변환
    │   ├── llm_client.py      # LLM 추상화 (Ollama/OpenAI)
    │   ├── rag_pipeline.py    # RAG 파이프라인 (ChromaDB)
    │   └── main.py            # FastAPI 서버
    ├── static/
    │   └── dashboard.html     # ACS 관제 대시보드
    ├── data/
    │   ├── docs/              # 안전 규정 문서
    │   └── chroma_db/         # 벡터 DB
    ├── requirements.txt
    └── README.md

---

## 개발 배경

벰로보틱스의 PCS(위치·제어)와 ACS(관제) 제품 구조를 분석하고,
카메라 Vision + LLM 판단 + 관제 대시보드를 하나의 파이프라인으로 구현했습니다.
공장 내 보안 환경을 고려해 외부 API 없이 완전 로컬(Ollama)로 동작하도록 설계했으며,
향후 ROS2 노드 및 3D LiDAR 융합으로 실제 AMR 탑재까지 확장 가능한 구조입니다.
