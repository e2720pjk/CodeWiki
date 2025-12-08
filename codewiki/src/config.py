from dataclasses import dataclass
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Constants
OUTPUT_BASE_DIR = "output"
DEPENDENCY_GRAPHS_DIR = "dependency_graphs"
DOCS_DIR = "docs"
FIRST_MODULE_TREE_FILENAME = "first_module_tree.json"
MODULE_TREE_FILENAME = "module_tree.json"
OVERVIEW_FILENAME = "overview.md"
MAX_DEPTH = 2
MAX_TOKEN_PER_MODULE = 36_369
MAX_TOKEN_PER_LEAF_MODULE = 16_000

# CLI context detection
_CLI_CONTEXT = False


def set_cli_context(enabled: bool = True):
    """Set whether we're running in CLI context (vs web app)."""
    global _CLI_CONTEXT
    _CLI_CONTEXT = enabled


def is_cli_context() -> bool:
    """Check if running in CLI context."""
    return _CLI_CONTEXT


# LLM services
# In CLI mode, these will be loaded from ~/.codewiki/config.json + keyring
# In web app mode, use environment variables
MAIN_MODEL = os.getenv("MAIN_MODEL", "claude-sonnet-4")
FALLBACK_MODEL_1 = os.getenv("FALLBACK_MODEL_1", "glm-4p5")
CLUSTER_MODEL = os.getenv("CLUSTER_MODEL", MAIN_MODEL)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://0.0.0.0:4000/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-1234")


@dataclass
class Config:
    """Configuration class for CodeWiki."""

    repo_path: str
    output_dir: str
    dependency_graph_dir: str
    docs_dir: str
    max_depth: int
    # LLM configuration
    llm_base_url: str
    llm_api_key: str
    main_model: str
    cluster_model: str
    fallback_model: str = FALLBACK_MODEL_1
    # Analysis options
    respect_gitignore: bool = False
    # File analysis limits
    max_files: int = 100
    max_entry_points: int = 5
    max_connectivity_files: int = 10
    # Token configuration (keeping defaults as requested)
    max_tokens_per_module: int = MAX_TOKEN_PER_MODULE
    max_tokens_per_leaf: int = MAX_TOKEN_PER_LEAF_MODULE
    # Parallel processing configuration
    enable_parallel_processing: bool = True
    concurrency_limit: int = 5
    # LLM caching configuration
    enable_llm_cache: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create configuration from parsed arguments."""
        repo_name = os.path.basename(os.path.normpath(args.repo_path))
        sanitized_repo_name = "".join(c if c.isalnum() else "_" for c in repo_name)

        return cls(
            repo_path=args.repo_path,
            output_dir=OUTPUT_BASE_DIR,
            dependency_graph_dir=os.path.join(OUTPUT_BASE_DIR, DEPENDENCY_GRAPHS_DIR),
            docs_dir=os.path.join(OUTPUT_BASE_DIR, DOCS_DIR, f"{sanitized_repo_name}-docs"),
            max_depth=MAX_DEPTH,
            llm_base_url=LLM_BASE_URL,
            llm_api_key=LLM_API_KEY,
            main_model=MAIN_MODEL,
            cluster_model=CLUSTER_MODEL,
            fallback_model=FALLBACK_MODEL_1,
            enable_parallel_processing=True,
            concurrency_limit=5,
            enable_llm_cache=True,
        )

    @classmethod
    def from_cli(
        cls,
        repo_path: str,
        output_dir: str,
        llm_base_url: str,
        llm_api_key: str,
        main_model: str,
        cluster_model: str,
        fallback_model: str = FALLBACK_MODEL_1,
        respect_gitignore: bool = False,
        max_files: int = 100,
        max_entry_points: int = 5,
        max_connectivity_files: int = 10,
        max_tokens_per_module: int = MAX_TOKEN_PER_MODULE,
        max_tokens_per_leaf: int = MAX_TOKEN_PER_LEAF_MODULE,
        enable_parallel_processing: bool = True,
        concurrency_limit: int = 5,
    ) -> "Config":
        """
        Create configuration for CLI context.

        Args:
            repo_path: Repository path
            output_dir: Output directory for generated docs
            llm_base_url: LLM API base URL
            llm_api_key: LLM API key
            main_model: Primary model
            cluster_model: Clustering model
            fallback_model: Fallback model
            max_files: Maximum number of files to analyze
            max_entry_points: Maximum fallback entry points
            max_connectivity_files: Maximum fallback connectivity files
            max_tokens_per_module: Maximum tokens per module
            max_tokens_per_leaf: Maximum tokens per leaf module
            enable_parallel_processing: Enable parallel processing
            concurrency_limit: Maximum concurrent API calls

        Returns:
            Config instance
        """
        repo_name = os.path.basename(os.path.normpath(repo_path))
        base_output_dir = os.path.join(output_dir, "temp")

        return cls(
            repo_path=repo_path,
            output_dir=base_output_dir,
            dependency_graph_dir=os.path.join(base_output_dir, DEPENDENCY_GRAPHS_DIR),
            docs_dir=output_dir,
            max_depth=MAX_DEPTH,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            main_model=main_model,
            cluster_model=cluster_model,
            fallback_model=fallback_model,
            respect_gitignore=respect_gitignore,
            max_files=max_files,
            max_entry_points=max_entry_points,
            max_connectivity_files=max_connectivity_files,
            max_tokens_per_module=max_tokens_per_module,
            max_tokens_per_leaf=max_tokens_per_leaf,
            enable_parallel_processing=enable_parallel_processing,
            concurrency_limit=concurrency_limit,
        )
