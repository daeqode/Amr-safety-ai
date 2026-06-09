import sys
import cv2
import json
import asyncio
import base64
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).parent))
from detector import WarehouseDetector
from context_parser import ContextParser
from llm_client import LLMClient
from rag_pipeline import RAGPipeline

app = FastAPI(title="BEM Robotics Vision ACS", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 컴포넌트 초기화
detector = WarehouseDetector()
parser = ContextParser()
llm = LLMClient(provider="ollama", model="llama3.2:3b")
rag = RAGPipeline()

# 경고 이력 저장 (메모리)
alert_history: list[dict] = []
connected_clients: list[WebSocket] = []


@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path("static/dashboard.html").read_text()
    return HTMLResponse(content=html)


@app.get("/alerts")
async def get_alerts():
    return {"alerts": alert_history[-50:]}  # 최근 50개


@app.get("/status")
async def get_status():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "model": "llama3.2:3b",
        "rag_chunks": rag.vectorstore._collection.count() if rag.vectorstore else 0,
        "total_alerts": len(alert_history),
    }


@app.post("/analyze")
async def analyze_image(data: dict):
    """
    base64 인코딩된 이미지 프레임 분석
    { "image": "<base64>", "robot_id": "AMR-01" }
    """
    try:
        img_bytes = base64.b64decode(data["image"])
        img_array = cv2.imdecode(
            __import__("numpy").frombuffer(img_bytes, __import__("numpy").uint8),
            cv2.IMREAD_COLOR,
        )
        robot_id = data.get("robot_id", "AMR-01")

        detections = detector.detect_frame(img_array)
        context = parser.parse(detections, img_array.shape)

        rag_context = ""
        if context["risk_level"] != "SAFE":
            rag_context = rag.retrieve(context["summary"])

        llm_result = llm.analyze(context, rag_context)

        alert = {
            "timestamp": context["timestamp"],
            "robot_id": robot_id,
            "risk_level": llm_result.get("risk_level", context["risk_level"]),
            "action": llm_result.get("recommended_action", "SLOW_DOWN"),
            "message": llm_result.get("alert_message", ""),
            "reasoning": llm_result.get("reasoning", ""),
            "rag_reference": llm_result.get("rag_reference"),
            "objects": context["objects"],
            "risk_score": context["risk_score"],
        }

        alert_history.append(alert)

        # WebSocket으로 연결된 대시보드에 실시간 푸시
        await broadcast(alert)

        return {"success": True, "alert": alert}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        # 연결 시 최근 이력 전송
        await websocket.send_json({
            "type": "history",
            "alerts": alert_history[-10:],
        })
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def broadcast(alert: dict):
    """모든 연결된 대시보드에 알림 푸시"""
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json({"type": "alert", "data": alert})
        except Exception:
            disconnected.append(client)
    for c in disconnected:
        connected_clients.remove(c)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)