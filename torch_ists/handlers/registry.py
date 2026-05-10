"""Handler registry for packaged model routing.

This module provides handler registration and routing for the runtime package.

Minimal implementation scope:
- register(): Register handler instances
- get_handler(): Find handler for model name
- Cache: model_name -> handler (performance optimization)

Reference:
    - handler pattern guidance used by the runtime package
"""

from typing import Callable, Dict, List, Optional
from .base import BaseHandler


class HandlerRegistry:
    """Central registry for model handlers.

    This class manages handler registration and routing. It finds the
    appropriate handler for a given model name.

    Search Order:
        Handlers are searched in registration order. The first handler
        that can process the model is returned.

    Usage:
        >>> from .basic_handler import BasicHandler
        >>> HandlerRegistry.register(BasicHandler())
        >>> handler = HandlerRegistry.get_handler('cnn')
        >>> handler.name
        'basic'
    """

    # Handler instances (populated by register())
    _handlers: List[BaseHandler] = []

    # Cache: model_name -> handler (avoids repeated linear search)
    _handler_cache: Dict[str, BaseHandler] = {}
    _bootstrap: Optional[Callable[[], None]] = None
    _is_bootstrapping: bool = False

    @classmethod
    def register(cls, handler: BaseHandler) -> None:
        """Register a handler instance.

        Handlers should be registered in order from most specific to
        most general. The first handler that can process a model will
        be used.

        Args:
            handler: Handler instance to register
        """
        cls._handlers.append(handler)
        cls._handler_cache.clear()

    @classmethod
    def set_bootstrap(cls, bootstrap: Callable[[], None], *, reset: bool = False) -> None:
        """Register lazy bootstrap callback for handler initialization."""
        cls._bootstrap = bootstrap
        if reset:
            cls.clear()

    @classmethod
    def ensure_initialized(cls) -> None:
        """Initialize handlers lazily when registry is empty."""
        if cls._handlers or cls._bootstrap is None or cls._is_bootstrapping:
            return
        cls._is_bootstrapping = True
        try:
            cls._bootstrap()
        finally:
            cls._is_bootstrapping = False

    @classmethod
    def get_handler(cls, model_name: str) -> BaseHandler:
        """Get the appropriate handler for a model.

        Searches through registered handlers in order and returns the
        first one that can handle the given model name.

        Args:
            model_name: Model name (e.g., 'cnn', 'rnn', 'transformer')

        Returns:
            BaseHandler: Handler instance that can process the model

        Raises:
            ValueError: If model_name is empty or None, or if no handler can process the model
        """
        if not model_name:
            raise ValueError("model_name cannot be empty or None")

        cls.ensure_initialized()

        # Check cache first
        if model_name in cls._handler_cache:
            return cls._handler_cache[model_name]

        # Search for handler
        for handler in cls._handlers:
            if handler.can_handle(model_name):
                cls._handler_cache[model_name] = handler
                return handler

        raise ValueError(
            f"No handler found for model: '{model_name}'. "
            f"Registered handlers: {[h.name for h in cls._handlers]}"
        )

    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers.

        Useful for testing or reconfiguration.
        """
        cls._handlers.clear()
        cls._handler_cache.clear()

    @classmethod
    def handler_count(cls) -> int:
        """Get the number of registered handlers.

        Returns:
            int: Number of handlers
        """
        return len(cls._handlers)


__all__ = ['HandlerRegistry']
