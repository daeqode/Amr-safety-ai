import ollama
import json
from context_parser import ContextParser


SYSTEM_PROMPT = """You are an AI safety controller for an AMR/AGV logistics robot system (similar to BEM Robotics ACS).

Your job is to analyze obstacle detection data from the robot's camera and provide:
1. A risk assessment
2. Recommended action for the robot
3. A brief alert message for the control dashboard

Always respond in this exact JSON format:
{
    "risk_level": "SAFE|CAUTION|WARNING|CRITICAL",
    "recommended_action": "CONTINUE|SLOW_DOWN|STOP|REROUTE",
    "alert_message": "brief description for dashboard",
    "reasoning": "short explanation",
    "rag_reference": "relevant safety rule if available, else null"
}

Be concise. Prioritize worker safety above all."""


class LLMClient:
    """
    Ollama / OpenAI 추상화 클라이언트
    LLM 백엔드를 교체해도 인터페이스 동일
    """

    def __init__(self, provider: str = "ollama", model: str = "llama3.2:3b"):
        self.provider = provider
        self.model = model
        print(f"LLM Client 초기화: {provider} / {model}")

    def analyze(self, context: dict, rag_context: str = "") -> dict:
        prompt = self._build_prompt(context, rag_context)

        if self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_prompt(self, context: dict, rag_context: str) -> str:
        rag_section = ""
        if rag_context:
            rag_section = f"""
Relevant safety regulations:
{rag_context}
"""
        return f"""
Current robot environment scan:
- Timestamp: {context['timestamp']}
- Total obstacles: {context['total_objects']}
- Pre-calculated risk level: {context['risk_level']}
- Risk score: {context['risk_score']}

Detected objects:
{context['summary']}
{rag_section}
Analyze this situation and provide your assessment in the required JSON format.
"""

    def _call_ollama(self, prompt: str) -> dict:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.1},
            )
            raw = response["message"]["content"]
            return self._parse_json(raw)
        except Exception as e:
            return self._fallback(str(e))

    def _call_openai(self, prompt: str) -> dict:
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            return self._parse_json(raw)
        except Exception as e:
            return self._fallback(str(e))

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return {
                "risk_level": "WARNING",
                "recommended_action": "SLOW_DOWN",
                "alert_message": raw[:120],
                "reasoning": "JSON parse failed — raw response used",
                "rag_reference": None,
            }

    def _fallback(self, error: str) -> dict:
        return {
            "risk_level": "WARNING",
            "recommended_action": "SLOW_DOWN",
            "alert_message": "LLM unavailable — defaulting to caution mode",
            "reasoning": f"Error: {error}",
            "rag_reference": None,
        }


if __name__ == "__main__":
    sample_detections = [
        {"class": "person", "confidence": 0.87, "bbox": [100, 200, 250, 750]},
        {"class": "person", "confidence": 0.85, "bbox": [300, 180, 450, 760]},
        {"class": "bus",    "confidence": 0.87, "bbox": [0,   130, 810, 780]},
    ]
    sample_shape = (810, 810)

    parser = ContextParser()
    context = parser.parse(sample_detections, sample_shape)

    print("=== LLM 분석 시작 (Ollama llama3.2:3b) ===")
    print(f"입력 Risk Level: {context['risk_level']}\n")

    client = LLMClient(provider="ollama", model="llama3.2:3b")
    result = client.analyze(context)

    print("=== LLM 판단 결과 ===")
    for key, val in result.items():
        print(f"{key:20s}: {val}")