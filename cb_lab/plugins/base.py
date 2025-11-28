"""Plugin architecture for cb-lab framework extensibility."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
import time

import torch

if TYPE_CHECKING:
    from cb_lab.core.scheduler import Scheduler
    from cb_lab.core.request import Request
    from cb_lab.core.kv_cache import BaseKVCache
    from cb_lab.model.tiny_llm import TinyLLM
else:
    # For runtime, we need to use string annotations
    Scheduler = None
    Request = None
    BaseKVCache = None
    TinyLLM = None


class SchedulerPlugin(ABC):
    """Plugin interface for extending scheduler functionality."""

    @abstractmethod
    def before_step(self, scheduler: Any) -> None:
        """Called before each scheduler step."""
        pass

    @abstractmethod
    def after_step(self, scheduler: Any, step_stats: Dict[str, Any]) -> None:
        """Called after each scheduler step."""
        pass

    def on_request_added(self, scheduler: Any, request: Any) -> None:
        """Called when a request is added to the scheduler."""
        pass

    def on_request_finished(self, scheduler: Any, request: Any) -> None:
        """Called when a request finishes."""
        pass

    def get_name(self) -> str:
        """Get plugin name."""
        return self.__class__.__name__


class AttentionPlugin(ABC):
    """Plugin interface for extending attention mechanisms."""

    @abstractmethod
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute attention weights and output."""
        pass

    def get_name(self) -> str:
        """Get plugin name."""
        return self.__class__.__name__


class CachePlugin(ABC):
    """Plugin interface for extending KV cache functionality."""

    @abstractmethod
    def before_append(
        self, cache: Any, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> None:
        """Called before appending to KV cache."""
        pass

    @abstractmethod
    def after_append(
        self, cache: Any, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> None:
        """Called after appending to KV cache."""
        pass

    def get_name(self) -> str:
        """Get plugin name."""
        return self.__class__.__name__


class PluginManager:
    """Manager for loading and organizing plugins."""

    def __init__(self) -> None:
        self.scheduler_plugins: List[SchedulerPlugin] = []
        self.attention_plugins: List[AttentionPlugin] = []
        self.cache_plugins: List[CachePlugin] = []
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}

    def register_scheduler_plugin(
        self, plugin: SchedulerPlugin, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a scheduler plugin."""
        self.scheduler_plugins.append(plugin)
        if config:
            self.plugin_configs[plugin.get_name()] = config

    def register_attention_plugin(
        self, plugin: AttentionPlugin, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an attention plugin."""
        self.attention_plugins.append(plugin)
        if config:
            self.plugin_configs[plugin.get_name()] = config

    def register_cache_plugin(
        self, plugin: CachePlugin, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a cache plugin."""
        self.cache_plugins.append(plugin)
        if config:
            self.plugin_configs[plugin.get_name()] = config

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        return self.plugin_configs.get(plugin_name, {})

    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for a specific plugin."""
        self.plugin_configs[plugin_name] = config

    def list_plugins(self) -> Dict[str, List[str]]:
        """List all registered plugins."""
        return {
            "scheduler": [p.get_name() for p in self.scheduler_plugins],
            "attention": [p.get_name() for p in self.attention_plugins],
            "cache": [p.get_name() for p in self.cache_plugins],
        }
