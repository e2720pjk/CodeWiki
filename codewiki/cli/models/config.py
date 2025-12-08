"""
Configuration data models for CodeWiki CLI.

This module contains the Configuration class which represents persistent
user settings stored in ~/.codewiki/config.json. These settings are converted
to the backend Config class when running documentation generation.
"""

from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from codewiki.cli.utils.validation import (
    validate_url,
    validate_api_key,
    validate_model_name,
)


@dataclass
class Configuration:
    """
    CodeWiki configuration data model.

    Attributes:
        base_url: LLM API base URL
        main_model: Primary model for documentation generation
        cluster_model: Model for module clustering
        default_output: Default output directory
        max_files: Maximum number of files to analyze
        max_entry_points: Maximum fallback entry points
        max_connectivity_files: Maximum fallback connectivity files
        max_tokens_per_module: Maximum tokens per module (keeps default)
        max_tokens_per_leaf: Maximum tokens per leaf module (keeps default)
        enable_parallel_processing: Enable parallel processing
        concurrency_limit: Maximum concurrent API calls
    """

    base_url: str
    main_model: str
    cluster_model: str
    default_output: str = "docs"
    max_files: int = 100
    max_entry_points: int = 5
    max_connectivity_files: int = 10
    max_tokens_per_module: int = 36369  # Keep default as requested
    max_tokens_per_leaf: int = 16000  # Keep default as requested
    enable_parallel_processing: bool = True
    concurrency_limit: int = 5

    def validate(self):
        """
        Validate all configuration fields.

        Raises:
            ConfigurationError: If validation fails
        """
        validate_url(self.base_url)
        validate_model_name(self.main_model)
        validate_model_name(self.cluster_model)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Configuration":
        """
        Create Configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Configuration instance
        """
        return cls(
            base_url=data.get("base_url", ""),
            main_model=data.get("main_model", ""),
            cluster_model=data.get("cluster_model", ""),
            default_output=data.get("default_output", "docs"),
            max_files=data.get("max_files", 100),
            max_entry_points=data.get("max_entry_points", 5),
            max_connectivity_files=data.get("max_connectivity_files", 10),
            max_tokens_per_module=data.get("max_tokens_per_module", 36369),
            max_tokens_per_leaf=data.get("max_tokens_per_leaf", 16000),
            enable_parallel_processing=data.get("enable_parallel_processing", True),
            concurrency_limit=data.get("concurrency_limit", 5),
        )

    def is_complete(self) -> bool:
        """Check if all required fields are set."""
        return bool(self.base_url and self.main_model and self.cluster_model)

    def to_backend_config(self, repo_path: str, output_dir: str, api_key: str):
        """
        Convert CLI Configuration to Backend Config.

        This method bridges the gap between persistent user settings (CLI Configuration)
        and runtime job configuration (Backend Config).

        Args:
            repo_path: Path to the repository to document
            output_dir: Output directory for generated documentation
            api_key: LLM API key (from keyring)

        Returns:
            Backend Config instance ready for documentation generation
        """
        from codewiki.src.config import Config

        return Config.from_cli(
            repo_path=repo_path,
            output_dir=output_dir,
            llm_base_url=self.base_url,
            llm_api_key=api_key,
            main_model=self.main_model,
            cluster_model=self.cluster_model,
            max_files=self.max_files,
            max_entry_points=self.max_entry_points,
            max_connectivity_files=self.max_connectivity_files,
            max_tokens_per_module=self.max_tokens_per_module,
            max_tokens_per_leaf=self.max_tokens_per_leaf,
            enable_parallel_processing=self.enable_parallel_processing,
            concurrency_limit=self.concurrency_limit,
        )
