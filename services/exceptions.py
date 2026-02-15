"""
Samvad XR — Custom Exception Types
===================================
Shared between Dev A and Dev B.
Dev B raises these from services; Dev B catches them in main.py.
Dev A may also raise them from generate_vendor_response().
"""


class SarvamServiceError(Exception):
    """Raised when Sarvam AI API (STT or TTS) is unreachable or returns non-200."""

    def __init__(self, service: str, status_code: int | None = None, detail: str = ""):
        self.service = service  # "STT" or "TTS"
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Sarvam {service} error ({status_code}): {detail}")


class RAGServiceError(Exception):
    """Raised when ChromaDB query fails."""

    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(f"RAG error: {detail}")
