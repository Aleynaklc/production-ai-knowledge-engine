"""Short-lived parser worker with CPU limits and Linux address-space limits."""

import base64
import json
import resource
import sys


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    from backend.app.config import Settings
    from backend.app.documents.validation import DocumentUploadError
    from backend.app.workspaces.parsing import extract_in_process

    payload = json.loads(sys.stdin.read())
    settings = Settings(
        upload_max_expanded_bytes=payload["max_bytes"],
        upload_max_pages=payload["max_pages"],
        upload_ocr_enabled=payload.get("ocr_enabled", True),
        upload_ocr_languages=payload.get("ocr_languages", "eng"),
    )
    try:
        units = extract_in_process(payload["filename"], base64.b64decode(payload["data"]), settings)
        print(json.dumps({"units": [unit.model_dump() for unit in units]}, ensure_ascii=False))
    except DocumentUploadError as error:
        print(json.dumps({"error": {"code": error.code, "message": error.message}}))
        sys.exit(1)


if __name__ == "__main__":
    main()
