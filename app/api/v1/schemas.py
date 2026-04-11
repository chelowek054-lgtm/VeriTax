from __future__ import annotations

from pydantic import BaseModel, Field


class ExportIndexStatsDTO(BaseModel):
    path: str
    source_namespace: str
    threads_total: int
    threads_indexed: int
    messages_loaded: int


class IndexRunResponse(BaseModel):
    exports_processed: int
    threads_indexed: int
    errors: list[str] = Field(default_factory=list)
    exports: list[ExportIndexStatsDTO] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    source_namespace: str | None = Field(
        default=None,
        description="Optional chat folder (namespace). Empty string = all chats.",
    )


class AskSourceDTO(BaseModel):
    thread_id: int
    root_message_id: int
    message_ids: list[int] = Field(default_factory=list)
    source_namespace: str = ""
    source_chat: str = ""
    date_from: str = ""
    date_to: str = ""
    authors: list[str] = Field(default_factory=list)
    text: str = ""
    score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSourceDTO] = Field(default_factory=list)


class IndexJobCreateResponse(BaseModel):
    job_id: str


class IndexJobStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str
    exports_total: int
    exports_done: int
    current_source_namespace: str | None = None
    threads_indexed: int = 0
    current_export_relative_path: str | None = None
    current_export_threads_total: int = 0
    current_export_threads_done: int = 0
    errors: list[str] = Field(default_factory=list)
    fatal_error: str | None = None
    result: IndexRunResponse | None = None
