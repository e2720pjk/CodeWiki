from dataclasses import dataclass, asdict, field
from typing import Optional, List

from codewiki.cli.utils.validation import (
    validate_url,
    validate_model_name,
)
from codewiki.cli.models.job import AnalysisOptions


@dataclass
class AgentInstructions:
    """
    Custom instructions for the documentation agent.

    Allows users to customize:
    - File filtering (include/exclude patterns)
    - Module focus (prioritize certain modules)
    - Documentation type (API docs, architecture docs, etc.)
    - Custom instructions for the LLM

    Attributes:
        include_patterns: File patterns to include (e.g., ["*.cs", "*.py"])
        exclude_patterns: File/directory patterns to exclude (e.g., ["*Tests*", "*test*"])
        focus_modules: Modules to document in more detail
        doc_type: Type of documentation to generate
        custom_instructions: Additional instructions for the documentation agent
    """

    include_patterns: Optional[List[str]] = None  # e.g., ["*.cs"] for C# projects
    exclude_patterns: Optional[List[str]] = None  # e.g., ["*Tests*", "*Specs*"]
    focus_modules: Optional[List[str]] = None  # e.g., ["src/core", "src/api"]
    doc_type: Optional[str] = None  # e.g., "api", "architecture", "user-guide"
    custom_instructions: Optional[str] = None  # Free-form instructions

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.include_patterns:
            result["include_patterns"] = self.include_patterns
        if self.exclude_patterns:
            result["exclude_patterns"] = self.exclude_patterns
        if self.focus_modules:
            result["focus_modules"] = self.focus_modules
        if self.doc_type:
            result["doc_type"] = self.doc_type
        if self.custom_instructions:
            result["custom_instructions"] = self.custom_instructions
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "AgentInstructions":
        """Create AgentInstructions from dictionary."""
        return cls(
            include_patterns=data.get("include_patterns"),
            exclude_patterns=data.get("exclude_patterns"),
            focus_modules=data.get("focus_modules"),
            doc_type=data.get("doc_type"),
            custom_instructions=data.get("custom_instructions"),
        )

    def is_empty(self) -> bool:
        """Check if all fields are empty/None."""
        return not any(
            [
                self.include_patterns,
                self.exclude_patterns,
                self.focus_modules,
                self.doc_type,
                self.custom_instructions,
            ]
        )

    def get_prompt_addition(self) -> str:
        """Generate prompt additions based on instructions."""
        additions = []

        if self.doc_type:
            doc_type_instructions = {
                "api": (
                    "Focus on API documentation: endpoints, parameters, "
                    "return types, and usage examples."
                ),
                "architecture": (
                    "Focus on architecture documentation: system design, "
                    "component relationships, and data flow."
                ),
                "user-guide": (
                    "Focus on user guide documentation: how to use features, "
                    "step-by-step tutorials."
                ),
                "developer": (
                    "Focus on developer documentation: code structure, "
                    "contribution guidelines, and implementation details."
                ),
            }
            if self.doc_type.lower() in doc_type_instructions:
                additions.append(doc_type_instructions[self.doc_type.lower()])
            else:
                additions.append(f"Focus on generating {self.doc_type} documentation.")

        if self.focus_modules:
            modules_str = ", ".join(self.focus_modules)
            additions.append(
                f"Pay special attention to and provide more detailed documentation "
                f"for these modules: {modules_str}"
            )

        if self.custom_instructions:
            additions.append(f"Additional instructions: {self.custom_instructions}")

        return "\n".join(additions) if additions else ""


@dataclass
class Configuration:
    """
    CodeWiki configuration data model.

    This configuration represents persistent user settings stored in ~/.codewiki/config.json.
    These settings are converted to backend Config class when running documentation generation.

    Attributes:
        base_url: LLM API base URL
        main_model: Primary model for documentation generation
        cluster_model: Model for module clustering
        fallback_model: Fallback model for documentation generation
        default_output: Default output directory
        max_tokens: Maximum tokens for LLM response (default: 32768)
        max_token_per_module: Maximum tokens per module for clustering (default: 36369)
        max_token_per_leaf_module: Maximum tokens per leaf module (default: 16000)
        max_depth: Maximum depth for hierarchical decomposition (default: 2)
        agent_instructions: Custom agent instructions for documentation generation

        # Analysis Options (Integrated)
        max_files: Maximum number of files to analyze
        max_entry_points: Maximum fallback entry points
        max_connectivity_files: Maximum fallback connectivity files
        enable_parallel_processing: Enable parallel processing of leaf modules
        concurrency_limit: Maximum concurrent API calls
        enable_llm_cache: Enable LLM caching
        agent_retries: Number of retries for agent tasks
        cache_size: LLM cache size (number of cached prompts)
        use_joern: Whether to use Joern for analysis
        respect_gitignore: Whether to respect .gitignore
    """

    base_url: str
    main_model: str
    cluster_model: str
    fallback_model: str = "glm-4p5"
    default_output: str = "docs"
    max_tokens: int = 32768
    max_token_per_module: int = 36369
    max_token_per_leaf_module: int = 16000
    max_depth: int = 2
    agent_instructions: AgentInstructions = field(default_factory=AgentInstructions)

    # Integrated Analysis Options fields
    max_files: int = 100
    max_entry_points: int = 5
    max_connectivity_files: int = 10
    enable_parallel_processing: bool = True
    concurrency_limit: int = 5
    enable_llm_cache: bool = True
    agent_retries: int = 3
    cache_size: int = 1000
    use_joern: bool = False
    respect_gitignore: bool = False

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
                f"max_entry_points must be between 1 and max_files "
                f"({self.max_files}), got {self.max_entry_points}"
            )
        if not (1 <= self.max_connectivity_files <= self.max_files):
            raise ValueError(
                f"max_connectivity_files must be between 1 and max_files "
                f"({self.max_files}), got {self.max_connectivity_files}"
            )
        if not (1000 <= self.max_token_per_module <= 200000):
            raise ValueError(
                f"max_token_per_module must be between 1000 and 200000, "
                f"got {self.max_token_per_module}"
            )
        if not (500 <= self.max_token_per_leaf_module <= 100000):
            raise ValueError(
                f"max_token_per_leaf_module must be between 500 and 100000, "
                f"got {self.max_token_per_leaf_module}"
            )
        if not (1 <= self.concurrency_limit <= 10):
            raise ValueError(
                f"concurrency_limit must be between 1 and 10, got {self.concurrency_limit}"
            )
        if not (100 <= self.cache_size <= 10000):
            raise ValueError(f"cache_size must be between 100 and 10000, got {self.cache_size}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = asdict(self)
        if self.agent_instructions and not self.agent_instructions.is_empty():
            result["agent_instructions"] = self.agent_instructions.to_dict()
        else:
            result.pop("agent_instructions", None)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Configuration":
        """
        Create Configuration from dictionary with type validation.

        Args:
            data: Configuration dictionary

        Returns:
            Configuration instance

        Raises:
            ValueError: If any field has an invalid type
        """

        def _get_typed_value(key: str, default_value, expected_type: type, field_name: str):
            """Get value from dictionary with strict type checking."""
            value = data.get(key, default_value)
            if value is None or value is default_value:
                return value

            actual_type = type(value)
            if expected_type is bool:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Invalid type for {field_name}: expected bool, got {actual_type.__name__}"
                    )
            else:
                if actual_type is bool:
                    raise ValueError(
                        f"Invalid type for {field_name}: expected "
                        f"{expected_type.__name__}, got bool"
                    )
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Invalid type for {field_name}: expected "
                        f"{expected_type.__name__}, got {actual_type.__name__}"
                    )
            return value

        agent_instructions = AgentInstructions()
        if "agent_instructions" in data and data["agent_instructions"]:
            agent_instructions = AgentInstructions.from_dict(data["agent_instructions"])

        # Support old naming convention if present
        max_token_per_module = _get_typed_value(
            "max_token_per_module",
            data.get("max_tokens_per_module", 36369),
            int,
            "max_token_per_module",
        )
        max_token_per_leaf_module = _get_typed_value(
            "max_token_per_leaf_module",
            data.get("max_tokens_per_leaf", 16000),
            int,
            "max_token_per_leaf_module",
        )

        config = cls(
            base_url=_get_typed_value("base_url", "", str, "base_url"),
            main_model=_get_typed_value("main_model", "", str, "main_model"),
            cluster_model=_get_typed_value("cluster_model", "", str, "cluster_model"),
            fallback_model=_get_typed_value("fallback_model", "glm-4p5", str, "fallback_model"),
            default_output=_get_typed_value("default_output", "docs", str, "default_output"),
            max_tokens=_get_typed_value("max_tokens", 32768, int, "max_tokens"),
            max_token_per_module=max_token_per_module,
            max_token_per_leaf_module=max_token_per_leaf_module,
            agent_instructions=agent_instructions,
            max_files=_get_typed_value("max_files", 100, int, "max_files"),
            max_entry_points=_get_typed_value("max_entry_points", 5, int, "max_entry_points"),
            max_connectivity_files=_get_typed_value(
                "max_connectivity_files", 10, int, "max_connectivity_files"
            ),
            enable_parallel_processing=_get_typed_value(
                "enable_parallel_processing", True, bool, "enable_parallel_processing"
            ),
            concurrency_limit=_get_typed_value("concurrency_limit", 5, int, "concurrency_limit"),
            enable_llm_cache=_get_typed_value("enable_llm_cache", True, bool, "enable_llm_cache"),
            agent_retries=_get_typed_value("agent_retries", 3, int, "agent_retries"),
            cache_size=_get_typed_value("cache_size", 1000, int, "cache_size"),
            use_joern=_get_typed_value("use_joern", False, bool, "use_joern"),
            respect_gitignore=_get_typed_value(
                "respect_gitignore", False, bool, "respect_gitignore"
            ),
            max_depth=data.get("max_depth", 2),
        )

        if config.is_complete():
            config.validate()

        return config

    def is_complete(self) -> bool:
        """Check if all required fields are set."""
        return bool(
            self.base_url and self.main_model and self.cluster_model and self.fallback_model
        )

    def to_backend_config(
        self,
        repo_path: str,
        output_dir: str,
        api_key: str,
        runtime_instructions: AgentInstructions = None,
    ):
        """
        Convert CLI Configuration to Backend Config.
        """
        from codewiki.src.config import Config

        # Merge runtime instructions with persistent settings
        final_instructions = self.agent_instructions
        if runtime_instructions and not runtime_instructions.is_empty():
            final_instructions = AgentInstructions(
                include_patterns=runtime_instructions.include_patterns
                or self.agent_instructions.include_patterns,
                exclude_patterns=runtime_instructions.exclude_patterns
                or self.agent_instructions.exclude_patterns,
                focus_modules=runtime_instructions.focus_modules
                or self.agent_instructions.focus_modules,
                doc_type=runtime_instructions.doc_type or self.agent_instructions.doc_type,
                custom_instructions=runtime_instructions.custom_instructions
                or self.agent_instructions.custom_instructions,
            )

        analysis_options = AnalysisOptions(
            max_files=self.max_files,
            max_entry_points=self.max_entry_points,
            max_connectivity_files=self.max_connectivity_files,
            enable_parallel_processing=self.enable_parallel_processing,
            concurrency_limit=self.concurrency_limit,
            enable_llm_cache=self.enable_llm_cache,
            agent_retries=self.agent_retries,
            cache_size=self.cache_size,
            use_joern=self.use_joern,
            respect_gitignore=self.respect_gitignore,
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
            max_tokens=self.max_tokens,
            max_token_per_module=self.max_token_per_module,
            max_token_per_leaf_module=self.max_token_per_leaf_module,
            max_depth=self.max_depth,
            agent_instructions=final_instructions.to_dict() if final_instructions else None,
        )
