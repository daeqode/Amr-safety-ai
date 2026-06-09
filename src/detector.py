import cv2
from ultralytics import YOLO
from pathlib import Path

# 물류 현장 관련 클래스만 필터링
WAREHOUSE_CLASSES = {
    0: "person",
    24: "backpack",
    25: "umbrella",
    28: "suitcase",
    39: "bottle",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    72: "refrigerator",
    73: "book",
    74: "clock",
    76: "scissors",
    77: "teddy bear",
    79: "toothbrush",
}

# 실제 물류 핵심 클래스
TARGET_CLASSES = {
    0: "person",      # 작업자/보행자
    2: "car",         # 차량
    3: "motorcycle",  # 오토바이
    5: "bus",         # 버스
    7: "truck",       # 트럭/지게차 대용
    24: "backpack",   # 짐
    28: "suitcase",   # 화물 박스 대용
}


class WarehouseDetector:
    def __init__(self, model_size: str = "yolov8n.pt", device: str = "mps"):
        """
        model_size: yolov8n(빠름) / yolov8s / yolov8m(정확)
        device: mps(M2 가속) / cpu
        """
        self.model = YOLO(model_size)
        self.device = device
        print(f"모델 로드 완료: {model_size} / 디바이스: {device}")

    def detect_frame(self, frame) -> list[dict]:
        """
        단일 프레임에서 객체 감지
        반환: [{"class": "person", "confidence": 0.92, "bbox": [x1,y1,x2,y2]}, ...]
        """
        results = self.model(frame, device=self.device, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in TARGET_CLASSES:
                    continue

                confidence = float(box.conf[0])
                if confidence < 0.4:  # 신뢰도 40% 미만 제외
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": TARGET_CLASSES[class_id],
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

    def estimate_distance(self, bbox: list, frame_height: int) -> str:
        """
        바운딩박스 크기로 상대 거리 추정
        실제 LiDAR 없이 카메라만으로 거리 근사
        """
        _, y1, _, y2 = bbox
        box_height = y2 - y1
        ratio = box_height / frame_height

        if ratio > 0.5:
            return "Very close (< 1m)"
        elif ratio > 0.3:
            return "Close (1~2m)"
        elif ratio > 0.15:
            return "Mid range (2~4m)"
        else:
            return "Far (4m+)"

    def draw_detections(self, frame, detections: list[dict]) -> any:
        """감지 결과를 프레임에 시각화"""
        colors = {
            "person": (0, 0, 255),    # 빨강 - 사람 (위험)
            "truck": (0, 165, 255),   # 주황 - 트럭
            "car": (0, 255, 255),     # 노랑 - 차량
            "default": (0, 255, 0),   # 초록 - 기타
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class"]
            conf = det["confidence"]
            distance = self.estimate_distance(det["bbox"], frame.shape[0])

            color = colors.get(label, colors["default"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.0%} | {distance}"
            cv2.putText(frame, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 감지 수 표시
        cv2.putText(frame, f"Detected: {len(detections)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame


def test_with_webcam():
    """웹캠으로 실시간 테스트"""
    detector = WarehouseDetector()
    cap = cv2.VideoCapture(0)

    print("웹캠 테스트 시작 - q 누르면 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_frame(frame)
        frame = detector.draw_detections(frame, detections)

        if detections:
            print(f"감지: {[d['class'] for d in detections]}")

        cv2.imshow("Warehouse Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_with_image():
    """이미지 파일로 테스트"""
    import urllib.request
    
    # 테스트용 샘플 이미지 다운로드 (COCO 샘플)
    url = "https://ultralytics.com/images/bus.jpg"
    save_path = "data/videos/test_sample.jpg"
    
    print("샘플 이미지 다운로드 중...")
    urllib.request.urlretrieve(url, save_path)
    
    detector = WarehouseDetector()
    frame = cv2.imread(save_path)
    
    detections = detector.detect_frame(frame)
    frame = detector.draw_detections(frame, detections)
    
    print(f"\n감지 결과 ({len(detections)}개):")
    for d in detections:
        distance = detector.estimate_distance(d["bbox"], frame.shape[0])
        print(f"  - {d['class']} | 신뢰도: {d['confidence']:.0%} | 거리: {distance}")
    
    cv2.imwrite("data/videos/test_result.jpg", frame)
    print("\n결과 이미지 저장: data/videos/test_result.jpg")


if __name__ == "__main__":
    test_with_image()
    