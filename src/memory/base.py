__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


class MemoryStoreError(RuntimeError):
    """Defined failure raised by the experimental memory subsystem."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class MemoryRecord:
    """
    One persistent experimental memory.

    The memory identifier is stable within its namespace and is never
    silently reassigned by the store.
    """

    namespace: str
    memory_id: str
    key: str
    value: Any

    created_by_task: str | None = None
    created_sequence: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemorySnapshot:
    """
    Complete snapshot of one memory namespace.

    Records are stored in deterministic memory_id order. The checksum
    covers the namespace and full serialized record set.
    """

    namespace: str
    records: tuple[MemoryRecord, ...]
    checksum_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "records": [
                record.to_dict()
                for record in self.records
            ],
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass(frozen=True)
class MemoryMutationResult:
    """Result of an explicit targeted memory mutation."""

    operation: str
    namespace: str
    memory_id: str

    before: MemoryRecord | None
    after: MemoryRecord | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "namespace": self.namespace,
            "memory_id": self.memory_id,
            "before": (
                self.before.to_dict()
                if self.before is not None
                else None
            ),
            "after": (
                self.after.to_dict()
                if self.after is not None
                else None
            ),
        }


class MemoryStore(ABC):
    """Minimal interface for persistent experimental memory."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backing store."""
        raise NotImplementedError

    @abstractmethod
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
        """Create one new memory without overwriting an existing ID."""
        raise NotImplementedError

    @abstractmethod
    def read(
        self,
        *,
        namespace: str,
        memory_id: str,
    ) -> MemoryRecord | None:
        """Read exactly one memory by stable ID."""
        raise NotImplementedError

    @abstractmethod
    def list_records(
        self,
        *,
        namespace: str,
    ) -> tuple[MemoryRecord, ...]:
        """Return all namespace records in deterministic order."""
        raise NotImplementedError

    @abstractmethod
    def targeted_remove(
        self,
        *,
        namespace: str,
        memory_id: str,
    ) -> MemoryMutationResult:
        """Remove exactly one identified memory."""
        raise NotImplementedError

    @abstractmethod
    def targeted_replace(
        self,
        *,
        namespace: str,
        memory_id: str,
        replacement_value: Any,
    ) -> MemoryMutationResult:
        """Replace exactly one memory value while preserving identity."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(
        self,
        *,
        namespace: str,
    ) -> MemorySnapshot:
        """Create a complete integrity-checked namespace snapshot."""
        raise NotImplementedError

    @abstractmethod
    def restore(
        self,
        snapshot: MemorySnapshot,
    ) -> None:
        """Restore one namespace exactly from a validated snapshot."""
        raise NotImplementedError