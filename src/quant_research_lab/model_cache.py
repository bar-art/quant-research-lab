"""Resume model work by replaying validated responses; market state is recomputed."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .agents import StructuredModel

T = TypeVar("T", bound=BaseModel)


class CachedModel:
    def __init__(
        self, model: StructuredModel, directory: Path, registration: dict[str, Any]
    ) -> None:
        self.model = model
        self.namespace = hashlib.sha256(
            json.dumps(registration, sort_keys=True).encode()
        ).hexdigest()
        self.directory = directory / self.namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self.ordinal = 0

    def generate(self, role: str, payload: dict[str, Any], schema: type[T]) -> T:
        request = {
            "ordinal": self.ordinal,
            "role": role,
            "payload": payload,
            "schema": schema.model_json_schema(),
            "namespace": self.namespace,
        }
        self.ordinal += 1
        key = hashlib.sha256(
            json.dumps(request, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        path = self.directory / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            response = record["response"]
            digest = hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
            if record.get("request_sha256") != key or record.get("response_sha256") != digest:
                raise ValueError("Model cache hash mismatch; refuse to silently replace evidence")
            result = schema.model_validate(response)
            self.cache_hits += 1
            return result
        self.calls += 1
        result = self.model.generate(role, payload, schema)
        response = result.model_dump(mode="json")
        record = {
            "request_sha256": key,
            "response": response,
            "response_sha256": hashlib.sha256(
                json.dumps(response, sort_keys=True).encode()
            ).hexdigest(),
        }
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # Atomic creation; never overwrite another session's result.
        except FileExistsError:
            if json.loads(path.read_text()) != record:
                raise ValueError("Conflicting cached responses; use an isolated cache") from None
        finally:
            temporary.unlink()
        return result
