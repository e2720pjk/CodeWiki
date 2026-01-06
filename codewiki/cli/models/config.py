"""
Configuration data models for CodeWiki CLI.

This module contains the Configuration class which represents persistent
user settings stored in ~/.codewiki/config.json. These settings are converted
to the backend Config class when running documentation generation.
"""

from dataclasses import dataclass, asdict

from codewiki.cli.utils.validation import (
    validate_url,
    validate_model_name,
)
from codewiki.cli.models.job import AnalysisOptions


@dataclass
class Configuration:
    """
    CodeWiki configuration data model.

    This configuration represents persistent user settings stored in ~/.codewiki/config.json.
    These settings are converted to backend Config class when running documentation generation.

    Attributes:
        base_url: LLM API base URL (e.g., https://api.anthropic.com)
                  Required for all LLM operations.

        main_model: Primary model for documentation generation
                   (e.g., claude-sonnet-4, gpt-4o)
                   This model generates most documentation content.

        cluster_model: Model for module clustering
                      Recommend top-tier model for better clustering quality
                      (e.g., claude-sonnet-4, gpt-4o)
                      Used only for module organization, not documentation generation.

        fallback_model: Fallback model for documentation generation
                        (e.g., glm-4p5, gpt-4-turbo)
                        Used when main model fails or is unavailable.
                        Default: glm-4p5

        default_output: Default output directory for generated docs
                      Relative path or absolute path
                      Default: "docs"

        max_files: Maximum number of files to analyze
                   Range: 1-5000
                   Default: 100
                   Limits analysis to prevent OOM on large repositories
                   Higher values = more comprehensive analysis but slower and more memory

        max_entry_points: Maximum fallback entry points
                          Range: 1-max_files
                          Default: 5
                          Number of entry files to identify when no obvious entry point exists
                          Used for repository structure analysis
                          Higher values = more entry points detected but potentially irrelevant

        max_connectivity_files: Maximum fallback connectivity files
                               Range: 1-max_files
                               Default: 10
                               Number of high-connectivity files to identify
                               Used for dependency graph construction
                               Higher values = more nodes in dependency graph but slower analysis

        max_tokens_per_module: Maximum tokens per module
                              Range: 1000-200000
                              Default: 36369
                              Controls module clustering and documentation generation size
                              Higher values = larger modules with more content but potentially less focused

        max_tokens_per_leaf: Maximum tokens per leaf module
                            Range: 500-100000
                            Default: 16000
                            Controls individual documentation file size
                            Higher values = longer documentation files but potentially overwhelming

        enable_parallel_processing: Enable parallel processing of leaf modules
                                 Type: boolean
                                 Default: True
                                 Improves performance on multi-core systems
                                 Set to False on systems with limited CPU or memory
                                 Parallel processing uses ThreadPoolExecutor with configurable workers

        concurrency_limit: Maximum concurrent API calls
                         Range: 1-10
                         Default: 5
                         Controls parallelism for LLM API calls
                         Higher values = faster documentation generation but higher API load
                         Consider API rate limits and system resources when adjusting

        cache_size: LLM cache size (number of cached prompts)
                    Range: 100-10000
                    Default: 1000
                    Controls memory usage and cache hit rate for LLM prompts
                    Higher values = more cache hits but higher memory usage
                    Adjust based on available system memory
    """

    base_url: str
    main_model: str
    cluster_model: str
    fallback_model: str = "glm-4p5"
    default_output: str = "docs"
    max_files: int = 100
    max_entry_points: int = 5
    max_connectivity_files: int = 10
    max_tokens_per_module: int = 36369  # Keep default as requested
    max_tokens_per_leaf: int = 16000  # Keep default as requested
    enable_parallel_processing: bool = True
    concurrency_limit: int = 5
    cache_size: int = 1000

    def validate(self):
        """
        Validate all configuration fields.

        Raises:
            ConfigurationError: If validation fails
        """
        validate_url(self.base_url)
        validate_model_name(self.main_model)
        validate_model_name(self.cluster_model)
        validate_model_name(self.fallback_model)

        if not (1 <= self.max_files <= 5000):
            raise ValueError(f"max_files must be between 1 and 5000, got {self.max_files}")
        if not (1 <= self.max_entry_points <= self.max_files):
            raise ValueError(
                f"max_entry_points must be between 1 and max_files ({self.max_files}), got {self.max_entry_points}"
            )
        if not (1 <= self.max_connectivity_files <= self.max_files):
            raise ValueError(
                f"max_connectivity_files must be between 1 and max_files ({self.max_files}), got {self.max_connectivity_files}"
            )
        if not (1000 <= self.max_tokens_per_module <= 200000):
            raise ValueError(
                f"max_tokens_per_module must be between 1000 and 200000, got {self.max_tokens_per_module}"
            )
        if not (500 <= self.max_tokens_per_leaf <= 100000):
            raise ValueError(
                f"max_tokens_per_leaf must be between 500 and 100000, got {self.max_tokens_per_leaf}"
            )
        if not (1 <= self.concurrency_limit <= 10):
            raise ValueError(
                f"concurrency_limit must be between 1 and 10, got {self.concurrency_limit}"
            )
        if not (100 <= self.cache_size <= 10000):
            raise ValueError(f"cache_size must be between 100 and 10000, got {self.cache_size}")

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
            fallback_model=data.get("fallback_model", "glm-4p5"),
            default_output=data.get("default_output", "docs"),
            max_files=data.get("max_files", 100),
            max_entry_points=data.get("max_entry_points", 5),
            max_connectivity_files=data.get("max_connectivity_files", 10),
            max_tokens_per_module=data.get("max_tokens_per_module", 36369),
            max_tokens_per_leaf=data.get("max_tokens_per_leaf", 16000),
            enable_parallel_processing=data.get("enable_parallel_processing", True),
            concurrency_limit=data.get("concurrency_limit", 5),
            cache_size=data.get("cache_size", 1000),
        )

    def is_complete(self) -> bool:
        """Check if all required fields are set."""
        return bool(
            self.base_url and self.main_model and self.cluster_model and self.fallback_model
        )

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

        analysis_options = AnalysisOptions(
            max_files=self.max_files,
            max_entry_points=self.max_entry_points,
            max_connectivity_files=self.max_connectivity_files,
            enable_parallel_processing=self.enable_parallel_processing,
            concurrency_limit=self.concurrency_limit,
            cache_size=self.cache_size,
        )

        return Config.from_cli(
            repo_path=repo_path,
            output_dir=output_dir,
            llm_base_url=self.base_url,
            llm_api_key=api_key,
            main_model=self.main_model,
            cluster_model=self.cluster_model,
            fallback_model=self.fallback_model,
            analysis_options=analysis_options,
            max_tokens_per_module=self.max_tokens_per_module,
            max_tokens_per_leaf=self.max_tokens_per_leaf,
        )
