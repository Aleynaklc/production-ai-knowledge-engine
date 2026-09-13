"""Document upload API contracts, persistence, and bounded body validation."""

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.documents.library import DocumentLibrary
from tests.legacy_api import create_app


class WordCodec:
    """Keep upload API tests independent of tokenizer downloads and inference."""

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


@pytest.fixture
def upload_api(tmp_path: Path) -> Iterator[tuple[Settings, DocumentLibrary, TestClient]]:
    settings = Settings(
        environment="test",
        rag_preload_on_startup=False,
        documents_path=tmp_path / "documents",
        upload_max_bytes=4_096,
        chunk_size_tokens=32,
        chunk_overlap_tokens=4,
    )
    library = DocumentLibrary(settings, project_root=tmp_path, codec=WordCodec())
    with TestClient(create_app(settings, document_library=library)) as client:
        yield settings, library, client


def _upload(client: TestClient, filename: str, content: bytes) -> httpx.Response:
    return cast(
        httpx.Response,
        client.post(
            "/api/v1/documents",
            params={"filename": filename},
            content=content,
            headers={"Content-Type": "application/octet-stream"},
        ),
    )


def _assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id"}
    assert error["code"] == code
    assert error["message"]
    assert error["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize("corrupt_chunks", [False, True])
def test_corrupt_persisted_records_return_storage_error(
    upload_api: tuple[Settings, DocumentLibrary, TestClient], corrupt_chunks: bool
) -> None:
    _, library, client = upload_api
    assert _upload(client, "guide.md", b"# Guide\n\nSupport information.").status_code == 201
    connection = sqlite3.connect(library.path)
    try:
        if corrupt_chunks:
            connection.execute("UPDATE chunks SET payload = '{}' ")
        else:
            connection.execute("UPDATE documents SET record_json = '{}' ")
        connection.commit()
    finally:
        connection.close()
    response = cast(
        httpx.Response,
        client.post("/api/v1/answers", json={"question": "What is in the guide?"}),
    )
    _assert_error(response, 503, "storage_unavailable")
    if not corrupt_chunks:
        _assert_error(
            cast(httpx.Response, client.get("/api/v1/documents")), 503, "storage_unavailable"
        )


def test_upload_returns_ready_document_and_retrievable_chunks(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
) -> None:
    settings, library, client = upload_api
    content = ("# Company handbook\n\n" + "Emergency contacts are in the directory. " * 15).encode()

    response = _upload(client, "handbook.md", content)

    assert response.status_code == 201
    assert response.headers["x-request-id"]
    payload = response.json()
    assert payload["duplicate"] is False
    document = payload["document"]
    assert document["id"]
    assert document["filename"] == "handbook.md"
    assert document["title"] == "Company handbook"
    assert document["size_bytes"] == len(content)
    assert document["sha256"] == hashlib.sha256(content).hexdigest()
    assert datetime.fromisoformat(document["created_at"]).tzinfo is not None
    assert document["status"] == "ready"
    assert document["chunk_strategy"] == settings.chunk_strategy
    assert document["chunk_size_tokens"] == settings.chunk_size_tokens
    assert document["overlap_tokens"] == settings.chunk_overlap_tokens

    revision, chunks = library.snapshot()
    assert revision
    assert len(chunks) == document["chunk_count"] > 1
    assert all(chunk.document_id == document["id"] for chunk in chunks)
    assert all(WordCodec().count(chunk.text) <= settings.chunk_size_tokens for chunk in chunks)
    assert any("Emergency contacts" in chunk.text for chunk in chunks)
    listing = client.get("/api/v1/documents")
    assert listing.status_code == 200
    assert listing.json()["documents"] == [document]
    assert listing.json()["count"] == 1


def test_duplicate_upload_is_idempotent_even_with_another_filename(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
) -> None:
    _, library, client = upload_api
    content = b"# Team policy\n\nContact the support team for help."
    original = _upload(client, "policy.md", content)
    original_snapshot = library.snapshot()

    duplicate = _upload(client, "renamed.md", content)

    assert original.status_code == 201
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document"] == original.json()["document"]
    assert library.snapshot() == original_snapshot
    assert client.get("/api/v1/documents").json()["count"] == 1


def test_uploaded_library_persists_across_application_instances(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
    tmp_path: Path,
) -> None:
    settings, library, client = upload_api
    response = _upload(client, "knowledge.txt", "Türkçe belge: destek ekibine ulaşın.".encode())
    assert response.status_code == 201
    original_snapshot = library.snapshot()
    restored = DocumentLibrary(settings, project_root=tmp_path, codec=WordCodec())

    with TestClient(create_app(settings, document_library=restored)) as second_client:
        listing = second_client.get("/api/v1/documents")

    assert listing.status_code == 200
    assert listing.json()["documents"] == [response.json()["document"]]
    assert restored.snapshot() == original_snapshot


def test_empty_library_lists_limits_without_loading_models(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, client = upload_api

    def unexpected_model_load(*args: object, **kwargs: object) -> None:
        pytest.fail("Listing uploaded documents must not load a generation model")

    monkeypatch.setattr("backend.app.rag.factory.load_runtime", unexpected_model_load)

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [],
        "count": 0,
        "max_file_bytes": settings.upload_max_bytes,
        "supported_extensions": [".md", ".txt"],
        "scope": "shared",
    }


@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("report.pdf", b"not a supported text document", 415, "unsupported_extension"),
        ("../escape.md", b"path traversal", 400, "invalid_filename"),
        ("/tmp/escape.md", b"absolute path", 400, "invalid_filename"),
        ("..\\escape.md", b"Windows path traversal", 400, "invalid_filename"),
        ("  ", b"missing name", 400, "invalid_filename"),
        ("broken.txt", b"invalid UTF-8: \xff\xfe", 422, "invalid_encoding"),
        ("empty.txt", b"", 422, "empty_document"),
        ("blank.md", b" \r\n\t\n", 422, "empty_document"),
        ("binary.txt", b"text\x00binary", 422, "binary_document"),
    ],
)
def test_invalid_uploads_return_correlated_errors_without_persisting(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
    filename: str,
    content: bytes,
    status: int,
    code: str,
) -> None:
    _, library, client = upload_api

    response = _upload(client, filename, content)

    _assert_error(response, status, code)
    assert library.list_documents() == []
    assert library.snapshot()[1] == []


def test_missing_filename_uses_request_validation_error(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
) -> None:
    _, library, client = upload_api

    response = client.post("/api/v1/documents", content=b"a document")

    _assert_error(response, 422, "request_validation_failed")
    assert library.list_documents() == []


@pytest.mark.parametrize("content_length", ["-1", "not-a-number", "1.5"])
def test_invalid_content_length_is_rejected_without_persisting(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
    content_length: str,
) -> None:
    _, library, client = upload_api

    response = client.post(
        "/api/v1/documents?filename=guide.txt",
        content=b"a document",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": content_length,
        },
    )

    _assert_error(response, 400, "invalid_content_length")
    assert library.list_documents() == []


@pytest.mark.parametrize("header_mode", ["accurate", "missing", "understated"])
def test_actual_body_limit_cannot_be_bypassed_with_content_length(
    upload_api: tuple[Settings, DocumentLibrary, TestClient],
    header_mode: str,
) -> None:
    settings, library, client = upload_api
    request = client.build_request(
        "POST",
        "/api/v1/documents?filename=too-large.txt",
        content=b"x" * (settings.upload_max_bytes + 1),
        headers={"Content-Type": "application/octet-stream"},
    )
    if header_mode == "missing":
        del request.headers["Content-Length"]
    elif header_mode == "understated":
        request.headers["Content-Length"] = "1"

    response = client.send(request)

    _assert_error(response, 413, "file_too_large")
    assert library.list_documents() == []


def test_document_capacity_preserves_existing_upload_and_allows_duplicate(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        rag_preload_on_startup=False,
        documents_path=tmp_path / "documents",
        upload_max_documents=1,
    )
    library = DocumentLibrary(settings, project_root=tmp_path, codec=WordCodec())
    with TestClient(create_app(settings, document_library=library)) as client:
        first = _upload(client, "first.txt", b"The first document.")
        duplicate = _upload(client, "first-copy.txt", b"The first document.")
        overflow = _upload(client, "second.txt", b"Another document.")

        assert first.status_code == 201
        assert duplicate.status_code == 200
        assert duplicate.json()["duplicate"] is True
        _assert_error(overflow, 409, "document_limit_exceeded")
        assert client.get("/api/v1/documents").json()["documents"] == [first.json()["document"]]


def test_chunk_capacity_rejects_upload_without_partial_document(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        rag_preload_on_startup=False,
        documents_path=tmp_path / "documents",
        chunk_strategy="fixed",
        chunk_size_tokens=32,
        chunk_overlap_tokens=0,
        upload_max_chunks=2,
    )
    library = DocumentLibrary(settings, project_root=tmp_path, codec=WordCodec())
    with TestClient(create_app(settings, document_library=library)) as client:
        response = _upload(client, "too-many-chunks.txt", b"word " * 70)

    _assert_error(response, 409, "chunk_limit_exceeded")
    assert library.list_documents() == []
    assert library.snapshot()[1] == []
