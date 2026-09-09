__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import Any

from .base import (
    MemoryMutationResult,
    MemoryRecord,
    MemorySnapshot,
    MemoryStore,
    MemoryStoreError,
)


SCHEMA_VERSION = 1


class SQLiteMemoryStore(MemoryStore):
    """
    Transparent deterministic SQLite implementation of persistent memory.

    The store performs exact-ID retrieval only. It contains no semantic
    search, ranking, summarization, consolidation, or hidden retrieval
    logic.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        """
        Initialize or validate the SQLite memory database.

        Initialization is explicit. Normal operations will not silently
        create a missing database.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with sqlite3.connect(self.path) as connection:
                current_version = connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]

                if current_version not in {0, SCHEMA_VERSION}:
                    raise MemoryStoreError(
                        code="schema_version_mismatch",
                        message=(
                            "Unsupported memory database schema version: "
                            f"{current_version}"
                        ),
                    )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        namespace TEXT NOT NULL,
                        memory_id TEXT NOT NULL,
                        memory_key TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        created_by_task TEXT,
                        created_sequence INTEGER,
                        metadata_json TEXT NOT NULL,
                        PRIMARY KEY (namespace, memory_id)
                    )
                    """
                )

                connection.execute(
                    f"PRAGMA user_version = {SCHEMA_VERSION}"
                )

                connection.commit()

        except sqlite3.DatabaseError as exc:
            raise MemoryStoreError(
                code="database_initialization_failure",
                message=f"Unable to initialize memory database: {self.path}",
            ) from exc

    def write(
        self,
        *,
        namespace: str,
        memory_id: str,
        key: str,
        value: Any,
        created_by_task: str | None = None,
        created_sequence: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        self._validate_identifier("namespace", namespace)
        self._validate_identifier("memory_id", memory_id)
        self._validate_identifier("key", key)

        metadata_value = metadata or {}

        value_json = self._canonical_json(
            value,
            field_name="value",
        )

        metadata_json = self._canonical_json(
            metadata_value,
            field_name="metadata",
        )

        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO memories (
                        namespace,
                        memory_id,
                        memory_key,
                        value_json,
                        created_by_task,
                        created_sequence,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        namespace,
                        memory_id,
                        key,
                        value_json,
                        created_by_task,
                        created_sequence,
                        metadata_json,
                    ),
                )

        except sqlite3.IntegrityError as exc:
            raise MemoryStoreError(
                code="duplicate_memory_id",
                message=(
                    "Memory already exists for namespace "
                    f"{namespace!r} with ID {memory_id!r}."
                ),
            ) from exc

        finally:
            connection.close()

        record = self.read(
            namespace=namespace,
            memory_id=memory_id,
        )

        if record is None:
            raise MemoryStoreError(
                code="write_verification_failure",
                message=(
                    "Memory write completed but the record could not "
                    "be read back."
                ),
            )

        return record

    def read(
        self,
        *,
        namespace: str,
        memory_id: str,
    ) -> MemoryRecord | None:
        self._validate_identifier("namespace", namespace)
        self._validate_identifier("memory_id", memory_id)

        connection = self._connect()

        try:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT
                    namespace,
                    memory_id,
                    memory_key,
                    value_json,
                    created_by_task,
                    created_sequence,
                    metadata_json
                FROM memories
                WHERE namespace = ?
                  AND memory_id = ?
                """,
                (
                    namespace,
                    memory_id,
                ),
            ).fetchone()

        finally:
            connection.close()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_records(
        self,
        *,
        namespace: str,
    ) -> tuple[MemoryRecord, ...]:
        self._validate_identifier("namespace", namespace)

        connection = self._connect()

        try:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT
                    namespace,
                    memory_id,
                    memory_key,
                    value_json,
                    created_by_task,
                    created_sequence,
                    metadata_json
                FROM memories
                WHERE namespace = ?
                ORDER BY memory_id ASC
                """,
                (namespace,),
            ).fetchall()

        finally:
            connection.close()

        return tuple(
            self._row_to_record(row)
            for row in rows
        )

    def targeted_remove(
        self,
        *,
        namespace: str,
        memory_id: str,
    ) -> MemoryMutationResult:
        before = self.read(
            namespace=namespace,
            memory_id=memory_id,
        )

        if before is None:
            raise MemoryStoreError(
                code="memory_not_found",
                message=(
                    f"Cannot remove missing memory {memory_id!r} "
                    f"from namespace {namespace!r}."
                ),
            )

        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    """
                    DELETE FROM memories
                    WHERE namespace = ?
                      AND memory_id = ?
                    """,
                    (
                        namespace,
                        memory_id,
                    ),
                )

        finally:
            connection.close()

        return MemoryMutationResult(
            operation="remove",
            namespace=namespace,
            memory_id=memory_id,
            before=before,
            after=None,
        )

    def targeted_replace(
        self,
        *,
        namespace: str,
        memory_id: str,
        replacement_value: Any,
    ) -> MemoryMutationResult:
        before = self.read(
            namespace=namespace,
            memory_id=memory_id,
        )

        if before is None:
            raise MemoryStoreError(
                code="memory_not_found",
                message=(
                    f"Cannot replace missing memory {memory_id!r} "
                    f"in namespace {namespace!r}."
                ),
            )

        replacement_json = self._canonical_json(
            replacement_value,
            field_name="replacement_value",
        )

        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    """
                    UPDATE memories
                    SET value_json = ?
                    WHERE namespace = ?
                      AND memory_id = ?
                    """,
                    (
                        replacement_json,
                        namespace,
                        memory_id,
                    ),
                )

        finally:
            connection.close()

        after = self.read(
            namespace=namespace,
            memory_id=memory_id,
        )

        if after is None:
            raise MemoryStoreError(
                code="replace_verification_failure",
                message=(
                    "Memory replacement completed but the record could "
                    "not be read back."
                ),
            )

        return MemoryMutationResult(
            operation="replace",
            namespace=namespace,
            memory_id=memory_id,
            before=before,
            after=after,
        )

    def snapshot(
        self,
        *,
        namespace: str,
    ) -> MemorySnapshot:
        records = self.list_records(
            namespace=namespace,
        )

        checksum = self._snapshot_checksum(
            namespace=namespace,
            records=records,
        )

        return MemorySnapshot(
            namespace=namespace,
            records=records,
            checksum_sha256=checksum,
        )

    def restore(
        self,
        snapshot: MemorySnapshot,
    ) -> None:
        self._validate_identifier(
            "namespace",
            snapshot.namespace,
        )

        expected_checksum = self._snapshot_checksum(
            namespace=snapshot.namespace,
            records=snapshot.records,
        )

        if expected_checksum != snapshot.checksum_sha256:
            raise MemoryStoreError(
                code="snapshot_checksum_mismatch",
                message=(
                    "Memory snapshot checksum validation failed for "
                    f"namespace {snapshot.namespace!r}."
                ),
            )

        connection = self._connect()

        try:
            with connection:
                connection.execute(
                    """
                    DELETE FROM memories
                    WHERE namespace = ?
                    """,
                    (snapshot.namespace,),
                )

                for record in snapshot.records:
                    if record.namespace != snapshot.namespace:
                        raise MemoryStoreError(
                            code="snapshot_namespace_mismatch",
                            message=(
                                "Snapshot contains a record from a "
                                "different namespace."
                            ),
                        )

                    value_json = self._canonical_json(
                        record.value,
                        field_name="snapshot.value",
                    )

                    metadata_json = self._canonical_json(
                        record.metadata,
                        field_name="snapshot.metadata",
                    )

                    connection.execute(
                        """
                        INSERT INTO memories (
                            namespace,
                            memory_id,
                            memory_key,
                            value_json,
                            created_by_task,
                            created_sequence,
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.namespace,
                            record.memory_id,
                            record.key,
                            value_json,
                            record.created_by_task,
                            record.created_sequence,
                            metadata_json,
                        ),
                    )

        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise MemoryStoreError(
                code="store_not_initialized",
                message=(
                    "Memory database does not exist. "
                    "Call initialize() first."
                ),
            )

        try:
            connection = sqlite3.connect(self.path)

            current_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]

            if current_version != SCHEMA_VERSION:
                connection.close()

                raise MemoryStoreError(
                    code="schema_version_mismatch",
                    message=(
                        "Memory database schema version is "
                        f"{current_version}; expected {SCHEMA_VERSION}."
                    ),
                )

            return connection

        except sqlite3.DatabaseError as exc:
            raise MemoryStoreError(
                code="database_connection_failure",
                message=f"Unable to open memory database: {self.path}",
            ) from exc

    @staticmethod
    def _validate_identifier(
        field_name: str,
        value: Any,
    ) -> None:
        if not isinstance(value, str) or not value:
            raise MemoryStoreError(
                code="invalid_identifier",
                message=(
                    f"{field_name!r} must be a non-empty string."
                ),
            )

    @staticmethod
    def _canonical_json(
        value: Any,
        *,
        field_name: str,
    ) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise MemoryStoreError(
                code="non_serializable_value",
                message=(
                    f"{field_name!r} must be losslessly representable "
                    "as JSON."
                ),
            ) from exc

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> MemoryRecord:
        return MemoryRecord(
            namespace=row["namespace"],
            memory_id=row["memory_id"],
            key=row["memory_key"],
            value=json.loads(row["value_json"]),
            created_by_task=row["created_by_task"],
            created_sequence=row["created_sequence"],
            metadata=json.loads(row["metadata_json"]),
        )

    def _snapshot_checksum(
        self,
        *,
        namespace: str,
        records: tuple[MemoryRecord, ...],
    ) -> str:
        payload = {
            "namespace": namespace,
            "records": [
                record.to_dict()
                for record in records
            ],
        }

        serialized = self._canonical_json(
            payload,
            field_name="snapshot",
        )

        return sha256(
            serialized.encode("utf-8")
        ).hexdigest()