import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 물류 안전 규정 — PDF 없을 때 사용할 내장 텍스트
BUILTIN_SAFETY_RULES = """
WAREHOUSE ROBOT SAFETY REGULATIONS

1. PERSONNEL SAFETY ZONES
- Robots must stop immediately when a person is detected within 1 meter.
- Speed must be reduced to 0.3 m/s when a person is detected within 3 meters.
- Audible warning must activate when a person is detected within 5 meters.
- Robot must not resume movement until the safety zone is clear.

2. VEHICLE INTERACTION
- Maintain minimum 2 meter clearance from forklifts and trucks at all times.
- Yield to all manned vehicles at intersections.
- Never cross paths with moving vehicles without ACS confirmation.

3. OBSTACLE RESPONSE PROTOCOL
- CRITICAL (risk score >= 8): Immediate full stop. Alert human operator via ACS.
- WARNING (risk score 4-7): Reduce speed 50%, activate warning lights, request reroute.
- CAUTION (risk score 1-3): Reduce speed 20%, increase sensor scan frequency.
- SAFE (risk score 0): Normal operation continues.

4. REROUTING RULES
- If primary path blocked for more than 10 seconds, request alternate route from ACS.
- Never attempt to navigate around a person — always wait or reroute via ACS.
- Log all obstacle events with timestamp and object type for ACS review.

5. EMERGENCY PROCEDURES
- If LLM or ACS connection lost, default to STOP and wait for manual override.
- Hardware e-stop takes priority over all software commands.
- All critical events must be logged even if ACS is offline.
"""


class RAGPipeline:
    """
    물류 안전 규정 문서를 벡터 DB에 저장하고
    LLM이 규정을 참조해 판단할 수 있도록 컨텍스트 제공
    """

    def __init__(self, docs_dir: str = "data/docs", persist_dir: str = "data/chroma_db"):
        self.docs_dir = Path(docs_dir)
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model="llama3.2:3b")
        self.vectorstore = None
        self._initialize()

    def _initialize(self):
        """벡터 DB 초기화 — 이미 있으면 로드, 없으면 생성"""
        db_path = Path(self.persist_dir)

        if db_path.exists() and any(db_path.iterdir()):
            print("기존 벡터 DB 로드 중...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            print(f"벡터 DB 로드 완료: {self.vectorstore._collection.count()}개 청크")
        else:
            print("벡터 DB 새로 생성 중...")
            self._build_vectorstore()

    def _build_vectorstore(self):
        """문서 로드 → 청크 분할 → 임베딩 → ChromaDB 저장"""
        docs = []

        # 1. PDF 파일이 있으면 로드
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if pdf_files:
            for pdf_path in pdf_files:
                print(f"PDF 로드: {pdf_path.name}")
                loader = PyPDFLoader(str(pdf_path))
                docs.extend(loader.load())

        # 2. 내장 안전 규정 텍스트 항상 추가
        print("내장 안전 규정 로드 중...")
        builtin_path = self.docs_dir / "builtin_safety_rules.txt"
        builtin_path.parent.mkdir(parents=True, exist_ok=True)
        builtin_path.write_text(BUILTIN_SAFETY_RULES, encoding="utf-8")

        loader = TextLoader(str(builtin_path), encoding="utf-8")
        docs.extend(loader.load())

        # 3. 텍스트 청크 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(docs)
        print(f"총 {len(chunks)}개 청크 생성")

        # 4. ChromaDB에 임베딩 저장
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        print("벡터 DB 저장 완료")

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        쿼리와 관련된 안전 규정 검색
        반환: LLM 프롬프트에 삽입할 텍스트
        """
        if not self.vectorstore:
            return ""

        docs = self.vectorstore.similarity_search(query, k=k)
        if not docs:
            return ""

        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"[Rule {i}] {doc.page_content.strip()}")

        return "\n\n".join(results)


if __name__ == "__main__":
    print("=== RAG Pipeline 테스트 ===\n")
    rag = RAGPipeline()

    queries = [
        "person detected very close to robot",
        "what to do when obstacle blocks path",
        "emergency stop procedure",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = rag.retrieve(query)
        print(f"Retrieved:\n{result[:300]}...")