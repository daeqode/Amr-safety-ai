from datetime import datetime


class ContextParser:
    """
    YOLO 감지 결과 → LLM 프롬프트용 텍스트 컨텍스트 변환
    실제 AMR/AGV가 ACS에 상황을 보고하는 구조를 모방
    """

    RISK_WEIGHTS = {
        "person": 3,   # 사람 — 최고 위험
        "truck": 2,    # 트럭
        "car": 2,      # 차량
        "bus": 2,      # 버스
        "motorcycle": 1,
        "default": 1,
    }

    def parse(self, detections: list[dict], frame_shape: tuple) -> dict:
        """
        detections: detector.py의 detect_frame() 반환값
        frame_shape: (height, width) — 화면 위치 계산용
        반환: LLM에 넘길 컨텍스트 딕셔너리
        """
        if not detections:
            return {
                "timestamp": datetime.now().isoformat(),
                "total_objects": 0,
                "risk_score": 0,
                "risk_level": "SAFE",
                "objects": [],
                "summary": "No obstacles detected. Path is clear.",
            }

        height, width = frame_shape[:2]
        parsed_objects = []

        for det in detections:
            position = self._get_position(det["bbox"], width)
            distance = self._get_distance_label(det["bbox"], height)
            parsed_objects.append({
                "type": det["class"],
                "confidence": det["confidence"],
                "position": position,
                "distance": distance,
            })

        risk_score = self._calc_risk(parsed_objects)
        risk_level = self._risk_level(risk_score)
        summary = self._build_summary(parsed_objects)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_objects": len(parsed_objects),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "objects": parsed_objects,
            "summary": summary,
        }

    def _get_position(self, bbox: list, frame_width: int) -> str:
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        ratio = center_x / frame_width
        if ratio < 0.35:
            return "left"
        elif ratio > 0.65:
            return "right"
        else:
            return "front"

    def _get_distance_label(self, bbox: list, frame_height: int) -> str:
        _, y1, _, y2 = bbox
        ratio = (y2 - y1) / frame_height
        if ratio > 0.5:
            return "very_close"
        elif ratio > 0.3:
            return "close"
        elif ratio > 0.15:
            return "mid"
        else:
            return "far"

    def _calc_risk(self, objects: list) -> int:
        score = 0
        for obj in objects:
            weight = self.RISK_WEIGHTS.get(obj["type"], self.RISK_WEIGHTS["default"])
            distance_mult = {"very_close": 4, "close": 2, "mid": 1, "far": 0}.get(
                obj["distance"], 1
            )
            score += weight * distance_mult
        return score

    def _risk_level(self, score: int) -> str:
        if score >= 8:
            return "CRITICAL"
        elif score >= 4:
            return "WARNING"
        elif score >= 1:
            return "CAUTION"
        else:
            return "SAFE"

    def _build_summary(self, objects: list) -> str:
        lines = []
        for obj in objects:
            lines.append(
                f"- {obj['type']} detected at {obj['position']}, "
                f"distance: {obj['distance'].replace('_', ' ')}, "
                f"confidence: {obj['confidence']:.0%}"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    # detector.py 결과로 테스트
    sample_detections = [
        {"class": "person", "confidence": 0.87, "bbox": [100, 200, 250, 750]},
        {"class": "person", "confidence": 0.85, "bbox": [300, 180, 450, 760]},
        {"class": "bus",    "confidence": 0.87, "bbox": [0,   130, 810, 780]},
    ]
    sample_shape = (810, 810)

    parser = ContextParser()
    result = parser.parse(sample_detections, sample_shape)

    print("=== Context Parser 결과 ===")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Risk Score : {result['risk_score']}")
    print(f"Objects    : {result['total_objects']}개")
    print(f"\nSummary:\n{result['summary']}")