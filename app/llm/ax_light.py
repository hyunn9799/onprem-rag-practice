"""LLM 답변 생성 — OpenAI 호환 클라이언트.

핵심: vLLM 은 OpenAI 호환 /v1/chat/completions 를 제공한다. 그래서 로컬 vLLM 과
OpenAI 를 '같은 클라이언트'로 호출하고 base_url / api_key / model 만 바꾸면 된다.

- settings.llm_provider="local"  → vLLM(로컬, 기본).  api_key 는 검사 안 함.
- settings.llm_provider="openai" → OpenAI. openai_api_key 필요.

T4 한 장에 임베딩+리랭커(TEI)까지 올리면 7B LLM fp16 은 메모리가 빠듯하다.
그럴 때 llm_provider=openai 로 두면 GPU 를 검색 모델에만 쓰고 LLM 은 API 로 뺄 수 있다.
"""
from __future__ import annotations

from typing import List

from openai import OpenAI

from app.config import settings

SYSTEM_PROMPT = (
    "너는 온프레미스 기업문서 RAG 어시스턴트다. "
    "반드시 제공된 근거 안에서만 답하고, 근거가 없거나 부족하면 "
    "'제공된 문서에서 확인되지 않습니다'라고 답한다. 답변 끝에 사용한 근거 번호를 표기한다."
)


class AXLightLLM:
    def __init__(self) -> None:
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError(
                    "llm_provider=openai 인데 openai_api_key 가 비어 있습니다."
                )
            base_url = settings.openai_base_url
            api_key = settings.openai_api_key
            self.model = settings.openai_model
        else:  # local vLLM (OpenAI 호환)
            base_url = settings.llm_base_url
            api_key = "EMPTY"  # vLLM 은 키를 검사하지 않음
            self.model = settings.llm_model

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def _build_context(contexts: List[str]) -> str:
        return "\n\n".join(f"[근거 {i + 1}]\n{c}" for i, c in enumerate(contexts))

    def generate(self, question: str, contexts: List[str], max_new_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"아래 근거만 사용해 질문에 답해줘.\n\n"
                    f"{self._build_context(contexts)}\n\n[질문]\n{question}",
                },
            ],
            temperature=0,  # 사내 문서 QA 는 재현성 위해 greedy
            max_tokens=max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
