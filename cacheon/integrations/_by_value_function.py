"""Identity-checked patch for a module function imported by value elsewhere."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable


@dataclass(frozen=True)
class ByValueFunctionPatch:
    source_module: str
    consumer_module: str | None
    function: str
    prefix: str
    callable_required: bool = True

    @property
    def patched(self) -> str:
        return f"_{self.prefix}_patched"

    @property
    def original(self) -> str:
        return f"_{self.prefix}_original"

    @property
    def dispatcher(self) -> str:
        return f"_{self.prefix}_dispatcher"

    @staticmethod
    def _initializing(module: ModuleType) -> bool:
        spec = getattr(module, "__spec__", None)
        return bool(spec is not None and getattr(spec, "_initializing", False))

    def _consumer(self) -> ModuleType | None:
        return sys.modules.get(self.consumer_module) if self.consumer_module else None

    def _consumer_state(
        self, consumer: ModuleType | None, original: object, dispatcher: object
    ) -> str:
        if consumer is None:
            return "absent"
        if not hasattr(consumer, self.function):
            if self._initializing(consumer):
                return "initializing"
            raise RuntimeError(f"{self.prefix} consumer has no reachable binding")
        binding = getattr(consumer, self.function)
        if binding is original:
            return "original"
        if binding is dispatcher:
            return "dispatcher"
        raise RuntimeError(f"{self.prefix} consumer binding drifted")

    def _installed(self, source: ModuleType) -> tuple[object, object]:
        try:
            original = getattr(source, self.original)
            dispatcher = getattr(source, self.dispatcher)
        except AttributeError as exc:
            raise RuntimeError(f"{self.prefix} seam has partial state") from exc
        if original is dispatcher or getattr(source, self.function, None) is not dispatcher:
            raise RuntimeError(f"{self.prefix} source binding drifted")
        return original, dispatcher

    def install(self, factory: Callable[[object, ModuleType], object]) -> None:
        source = sys.modules.get(self.source_module)
        if source is None:
            return
        consumer = self._consumer()
        if getattr(source, self.patched, False):
            original, dispatcher = self._installed(source)
            if self._consumer_state(consumer, original, dispatcher) == "original":
                setattr(consumer, self.function, dispatcher)
            return
        if hasattr(source, self.original) or hasattr(source, self.dispatcher):
            raise RuntimeError(f"{self.prefix} seam has stale patch state")
        original = getattr(source, self.function, None)
        if original is None:
            return
        if self.callable_required and not callable(original):
            raise RuntimeError(f"{self.prefix} source binding is not callable")
        dispatcher = factory(original, source)
        if (
            self.callable_required and not callable(dispatcher)
        ) or dispatcher is original:
            raise RuntimeError(f"{self.prefix} dispatcher is invalid")
        state = self._consumer_state(consumer, original, dispatcher)
        setattr(source, self.original, original)
        setattr(source, self.dispatcher, dispatcher)
        setattr(source, self.function, dispatcher)
        setattr(source, self.patched, True)
        if state == "original":
            setattr(consumer, self.function, dispatcher)

    def uninstall(self) -> None:
        source = sys.modules.get(self.source_module)
        if source is None or not getattr(source, self.patched, False):
            return
        original, dispatcher = self._installed(source)
        consumer = self._consumer()
        state = self._consumer_state(consumer, original, dispatcher)
        if state == "dispatcher":
            setattr(consumer, self.function, original)
        setattr(source, self.function, original)
        delattr(source, self.original)
        delattr(source, self.dispatcher)
        setattr(source, self.patched, False)

    def is_installed(self) -> bool:
        source = sys.modules.get(self.source_module)
        if source is None or not getattr(source, self.patched, False):
            return False
        try:
            original, dispatcher = self._installed(source)
            state = self._consumer_state(self._consumer(), original, dispatcher)
        except RuntimeError:
            return False
        return state in {"absent", "initializing", "dispatcher"}
