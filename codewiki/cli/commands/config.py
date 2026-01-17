"""
Configuration commands for CodeWiki CLI.
"""

import json
import sys
import click
from typing import Optional, List

from codewiki.cli.config_manager import ConfigManager
from codewiki.cli.models.config import AgentInstructions
from codewiki.cli.utils.errors import (
    ConfigurationError,
    handle_error,
    EXIT_CONFIG_ERROR,
)
from codewiki.cli.utils.validation import (
    validate_url,
    validate_api_key,
    validate_model_name,
    mask_api_key,
)


def parse_patterns(patterns_str: str) -> List[str]:
    """Parse comma-separated patterns into a list."""
    if not patterns_str:
        return []
    return [p.strip() for p in patterns_str.split(",") if p.strip()]


@click.group(name="config")
def config_group():
    """Manage CodeWiki configuration (API credentials and settings)."""
    pass


@config_group.command(name="set")
@click.option("--api-key", type=str, help="LLM API key (stored securely in system keychain)")
@click.option("--base-url", type=str, help="LLM API base URL (e.g., https://api.anthropic.com)")
@click.option("--main-model", type=str, help="Primary model for documentation generation")
@click.option("--cluster-model", type=str, help="Model for module clustering (recommend top-tier)")
@click.option("--fallback-model", type=str, help="Fallback model for documentation generation")
@click.option(
    "--enable-parallel-processing/--disable-parallel-processing",
    default=None,
    help="Enable parallel processing of leaf modules",
)
@click.option(
    "--concurrency-limit",
    type=click.IntRange(1, 10),
    default=None,
    help="Maximum concurrent API calls (1-10)",
)
@click.option(
    "--cache-size",
    type=click.IntRange(min=100, max=10000),
    default=None,
    help="LLM cache size - number of cached prompts (default: 1000, range: 100-10000)",
)
@click.option("--max-tokens", type=int, help="Maximum tokens for LLM response (default: 32768)")
@click.option(
    "--max-token-per-module",
    type=int,
    help="Maximum tokens per module for clustering (default: 36369)",
)
@click.option(
    "--max-token-per-leaf-module", type=int, help="Maximum tokens per leaf module (default: 16000)"
)
@click.option(
    "--use-joern/--no-joern",
    default=None,
    help="Enable Joern CPG analysis",
)
@click.option(
    "--respect-gitignore/--no-gitignore",
    default=None,
    help="Respect .gitignore patterns during file collection",
)
@click.option(
    "--agent-retries",
    type=click.IntRange(1, 10),
    default=None,
    help="Number of retries for agent tasks",
)
@click.option(
    "--enable-llm-cache/--disable-llm-cache",
    default=None,
    help="Enable LLM prompt caching",
)
@click.option(
    "--max-files", type=int, default=None, help="Maximum number of files to analyze (default: 100)"
)
@click.option(
    "--max-entry-points", type=int, default=None, help="Maximum fallback entry points (default: 5)"
)
@click.option(
    "--max-connectivity-files",
    type=int,
    default=None,
    help="Maximum fallback connectivity files (default: 10)",
)
@click.option(
    "--max-depth", type=int, help="Maximum depth for hierarchical decomposition (default: 2)"
)
def config_set(
    api_key: Optional[str],
    base_url: Optional[str],
    main_model: Optional[str],
    cluster_model: Optional[str],
    fallback_model: Optional[str],
    enable_parallel_processing: Optional[bool],
    concurrency_limit: Optional[int],
    cache_size: Optional[int],
    max_tokens: Optional[int],
    max_token_per_module: Optional[int],
    max_token_per_leaf_module: Optional[int],
    use_joern: Optional[bool],
    respect_gitignore: Optional[bool],
    agent_retries: Optional[int],
    enable_llm_cache: Optional[bool],
    max_files: Optional[int],
    max_entry_points: Optional[int],
    max_connectivity_files: Optional[int],
    max_depth: Optional[int],
):
    """
    Set configuration values for CodeWiki.
    
    API keys are stored securely in your system keychain:
      • macOS: Keychain Access
      • Windows: Credential Manager  
      • Linux: Secret Service (GNOME Keyring, KWallet)
    
    Examples:
    
    \b
    # Set all configuration
    $ codewiki config set --api-key sk-abc123 --base-url https://api.anthropic.com \\
        --main-model claude-sonnet-4 --cluster-model claude-sonnet-4 --fallback-model glm-4p5
    
    \b
    # Update only API key
    $ codewiki config set --api-key sk-new-key
    
    \b
    # Set max tokens for LLM response
    $ codewiki config set --max-tokens 16384
    
    \b
    # Set all max token settings
    $ codewiki config set --max-tokens 32768 \\
        --max-token-per-module 40000 --max-token-per-leaf-module 20000

    \b
    # Set max depth for hierarchical decomposition
    $ codewiki config set --max-depth 3
    """
    try:
        # Check if at least one option is provided
        if not any(
            [
                api_key,
                base_url,
                main_model,
                cluster_model,
                fallback_model,
                enable_parallel_processing is not None,
                concurrency_limit is not None,
                cache_size is not None,
                max_tokens is not None,
                max_token_per_module is not None,
                max_token_per_leaf_module is not None,
                use_joern is not None,
                respect_gitignore is not None,
                agent_retries is not None,
                enable_llm_cache is not None,
                max_files is not None,
                max_entry_points is not None,
                max_connectivity_files is not None,
                max_depth is not None,
            ]
        ):
            click.echo("No options provided. Use --help for usage information.")
            sys.exit(EXIT_CONFIG_ERROR)

        # Validate inputs before saving
        validated_data = {}

        if api_key:
            validated_data["api_key"] = validate_api_key(api_key)

        if base_url:
            validated_data["base_url"] = validate_url(base_url)

        if main_model:
            validated_data["main_model"] = validate_model_name(main_model)

        if cluster_model:
            validated_data["cluster_model"] = validate_model_name(cluster_model)

        if fallback_model:
            validated_data["fallback_model"] = validate_model_name(fallback_model)

        if max_tokens is not None:
            if max_tokens < 1:
                raise ConfigurationError("max_tokens must be a positive integer")
            validated_data["max_tokens"] = max_tokens

        if max_token_per_module is not None:
            if max_token_per_module < 1:
                raise ConfigurationError("max_token_per_module must be a positive integer")
            validated_data["max_token_per_module"] = max_token_per_module

        if max_token_per_leaf_module is not None:
            if max_token_per_leaf_module < 1:
                raise ConfigurationError("max_token_per_leaf_module must be a positive integer")
            validated_data["max_token_per_leaf_module"] = max_token_per_leaf_module

        if max_files is not None:
            if max_files < 1:
                raise ConfigurationError("max_files must be a positive integer")
            validated_data["max_files"] = max_files

        if max_entry_points is not None:
            if max_entry_points < 1:
                raise ConfigurationError("max_entry_points must be a positive integer")
            validated_data["max_entry_points"] = max_entry_points

        if max_connectivity_files is not None:
            if max_connectivity_files < 1:
                raise ConfigurationError("max_connectivity_files must be a positive integer")
            validated_data["max_connectivity_files"] = max_connectivity_files

        if max_depth is not None:
            if max_depth < 1:
                raise ConfigurationError("max_depth must be a positive integer")
            validated_data["max_depth"] = max_depth

        # Integrated flags/values (no validation needed beyond click types)
        if enable_parallel_processing is not None:
            validated_data["enable_parallel_processing"] = enable_parallel_processing
        if concurrency_limit is not None:
            validated_data["concurrency_limit"] = concurrency_limit
        if cache_size is not None:
            validated_data["cache_size"] = cache_size
        if use_joern is not None:
            validated_data["use_joern"] = use_joern
        if respect_gitignore is not None:
            validated_data["respect_gitignore"] = respect_gitignore
        if agent_retries is not None:
            validated_data["agent_retries"] = agent_retries
        if enable_llm_cache is not None:
            validated_data["enable_llm_cache"] = enable_llm_cache
        # Create config manager and save
        manager = ConfigManager()
        manager.load()  # Load existing config if present

        manager.save(
            api_key=validated_data.get("api_key"),
            base_url=validated_data.get("base_url"),
            main_model=validated_data.get("main_model"),
            cluster_model=validated_data.get("cluster_model"),
            fallback_model=validated_data.get("fallback_model"),
            max_tokens=validated_data.get("max_tokens"),
            max_token_per_module=validated_data.get("max_token_per_module"),
            max_token_per_leaf_module=validated_data.get("max_token_per_leaf_module"),
            enable_parallel_processing=validated_data.get("enable_parallel_processing"),
            concurrency_limit=validated_data.get("concurrency_limit"),
            cache_size=validated_data.get("cache_size"),
            use_joern=validated_data.get("use_joern"),
            respect_gitignore=validated_data.get("respect_gitignore"),
            agent_retries=validated_data.get("agent_retries"),
            enable_llm_cache=validated_data.get("enable_llm_cache"),
            max_files=validated_data.get("max_files"),
            max_entry_points=validated_data.get("max_entry_points"),
            max_connectivity_files=validated_data.get("max_connectivity_files"),
            max_depth=validated_data.get("max_depth"),
        )

        # Display success messages
        click.echo()
        if api_key:
            if manager.keyring_available:
                click.secho("✓ API key saved to system keychain", fg="green")
            else:
                click.secho(
                    "⚠️  System keychain unavailable. API key stored in encrypted file.", fg="yellow"
                )

        if base_url:
            click.secho(f"✓ Base URL: {base_url}", fg="green")
        if main_model:
            click.secho(f"✓ Main model: {main_model}", fg="green")
        if cluster_model:
            click.secho(f"✓ Cluster model: {cluster_model}", fg="green")
        if fallback_model:
            click.secho(f"✓ Fallback model: {fallback_model}", fg="green")
        if max_tokens:
            click.secho(f"✓ Max tokens: {max_tokens}", fg="green")
        if max_token_per_module:
            click.secho(f"✓ Max token per module: {max_token_per_module}", fg="green")
        if max_token_per_leaf_module:
            click.secho(f"✓ Max token per leaf module: {max_token_per_leaf_module}", fg="green")

        if max_files:
            click.secho(f"✓ Max files: {max_files}", fg="green")
        if max_entry_points:
            click.secho(f"✓ Max entry points: {max_entry_points}", fg="green")
        if max_connectivity_files:
            click.secho(f"✓ Max connectivity files: {max_connectivity_files}", fg="green")

        if max_depth:
            click.secho(f"✓ Max depth: {max_depth}", fg="green")

        if use_joern is not None:
            click.secho(f"✓ Joern CPG: {'enabled' if use_joern else 'disabled'}", fg="green")
        if respect_gitignore is not None:
            click.secho(
                f"✓ Respect .gitignore: {'enabled' if respect_gitignore else 'disabled'}",
                fg="green",
            )
        if enable_parallel_processing is not None:
            click.secho(
                f"✓ Parallel processing: {'enabled' if enable_parallel_processing else 'disabled'}",
                fg="green",
            )
        if concurrency_limit is not None:
            click.secho(f"✓ Concurrency limit: {concurrency_limit}", fg="green")
        click.echo("\n" + click.style("Configuration updated successfully.", fg="green", bold=True))

    except ConfigurationError as e:
        click.secho(f"\n✗ Configuration error: {e.message}", fg="red", err=True)
        sys.exit(e.exit_code)
    except Exception as e:
        sys.exit(handle_error(e))


@config_group.command(name="show")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
def config_show(output_json: bool):
    """
    Display current configuration.
    """
    try:
        manager = ConfigManager()
        if not manager.load():
            click.secho("\n✗ Configuration not found.", fg="red", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

        config = manager.get_config()
        api_key = manager.get_api_key()

        if output_json:
            output = manager.get_config().to_dict()
            output["api_key"] = mask_api_key(api_key) if api_key else "Not set"
            output["api_key_storage"] = (
                "keychain" if manager.keyring_available else "encrypted_file"
            )
            output["config_file"] = str(manager.config_file_path)
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo()
            click.secho("CodeWiki Configuration", fg="blue", bold=True)
            click.echo("━" * 40)

            click.secho("Credentials", fg="cyan", bold=True)
            click.echo(f"  API Key:          {mask_api_key(api_key) if api_key else 'Not set'}")
            storage_type = "system keychain" if manager.keyring_available else "encrypted file"
            click.echo(f"  Storage:          {storage_type}")

            click.echo()
            click.secho("API Settings", fg="cyan", bold=True)
            click.echo(f"  Base URL:         {config.base_url}")
            click.echo(f"  Main Model:       {config.main_model}")
            click.echo(f"  Cluster Model:    {config.cluster_model}")
            click.echo(f"  Fallback Model:   {config.fallback_model}")

            click.echo()
            click.secho("Token Settings", fg="cyan", bold=True)
            click.echo(f"  Max Tokens:              {config.max_tokens}")
            click.echo(f"  Max Token/Module:        {config.max_token_per_module}")
            click.echo(f"  Max Token/Leaf Module:   {config.max_token_per_leaf_module}")

            click.echo()
            click.secho("Analysis Settings", fg="cyan", bold=True)
            click.echo(f"  Max Files:               {config.max_files}")
            click.echo(f"  Max Entry Points:        {config.max_entry_points}")
            click.echo(f"  Max Connectivity Files:  {config.max_connectivity_files}")
            click.echo(f"  Use Joern:               {config.use_joern}")
            click.echo(f"  Respect .gitignore:      {config.respect_gitignore}")
            click.echo(f"  Parallel Processing:     {config.enable_parallel_processing}")
            click.echo(f"  Concurrency Limit:       {config.concurrency_limit}")
            click.echo(
                f"  LLM Cache:               {config.enable_llm_cache} (Size: {config.cache_size})"
            )

            click.echo()
            click.secho("Decomposition Settings", fg="cyan", bold=True)
            if config:
                click.echo(f"  Max Depth:               {config.max_depth}")

            click.echo()
            click.secho("Agent Instructions", fg="cyan", bold=True)
            if not config.agent_instructions or config.agent_instructions.is_empty():
                click.echo("  Using default settings")
            else:
                agent = config.agent_instructions
                if agent.include_patterns:
                    click.echo(f"  Include:   {', '.join(agent.include_patterns)}")
                if agent.exclude_patterns:
                    click.echo(f"  Exclude:   {', '.join(agent.exclude_patterns)}")
                if agent.focus_modules:
                    click.echo(f"  Focus:     {', '.join(agent.focus_modules)}")
                if agent.doc_type:
                    click.echo(f"  Doc Type:  {agent.doc_type}")
                if agent.custom_instructions:
                    click.echo(f"  Instructions: {agent.custom_instructions}")

            click.echo()
            click.echo(f"Configuration file: {manager.config_file_path}")
            click.echo()

    except Exception as e:
        sys.exit(handle_error(e))


@config_group.command(name="validate")
@click.option("--quick", is_flag=True, help="Skip API connectivity test")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed validation steps")
def config_validate(quick: bool, verbose: bool):
    """
    Validate configuration and test LLM API connectivity.
    """
    try:
        click.echo()
        click.secho("Validating configuration...", fg="blue", bold=True)

        manager = ConfigManager()
        if not manager.load():
            click.secho("✗ Configuration file not found", fg="red")
            sys.exit(EXIT_CONFIG_ERROR)

        click.secho("✓ Configuration file exists", fg="green")

        api_key = manager.get_api_key()
        if not api_key:
            click.secho("✗ API key missing", fg="red")
            sys.exit(EXIT_CONFIG_ERROR)
        click.secho("✓ API key present", fg="green")

        config = manager.get_config()
        try:
            validate_url(config.base_url)
            click.secho(f"✓ Base URL valid: {config.base_url}", fg="green")
        except ConfigurationError as e:
            click.secho(f"✗ Invalid base URL: {e.message}", fg="red")
            sys.exit(EXIT_CONFIG_ERROR)

        if not quick:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=config.base_url)
                client.models.list()
                click.secho("✓ API connectivity test successful", fg="green")
            except Exception:
                click.secho("✗ API connectivity test failed", fg="red")
                sys.exit(EXIT_CONFIG_ERROR)

        click.echo()
        click.secho("✓ Configuration is valid!", fg="green", bold=True)
        click.echo()

    except Exception as e:
        sys.exit(handle_error(e, verbose=verbose))


@config_group.command(name="agent")
@click.option(
    "--include",
    "-i",
    type=str,
    default=None,
    help="Comma-separated file patterns to include (e.g., '*.cs,*.py')",
)
@click.option(
    "--exclude",
    "-e",
    type=str,
    default=None,
    help="Comma-separated patterns to exclude (e.g., '*Tests*,*Specs*')",
)
@click.option(
    "--focus",
    "-f",
    type=str,
    default=None,
    help="Comma-separated modules/paths to focus on (e.g., 'src/core,src/api')",
)
@click.option(
    "--doc-type",
    "-t",
    type=click.Choice(["api", "architecture", "user-guide", "developer"], case_sensitive=False),
    default=None,
    help="Default type of documentation to generate",
)
@click.option(
    "--instructions",
    type=str,
    default=None,
    help="Custom instructions for the documentation agent",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Clear all agent instructions",
)
def config_agent(
    include: Optional[str],
    exclude: Optional[str],
    focus: Optional[str],
    doc_type: Optional[str],
    instructions: Optional[str],
    clear: bool,
):
    """
    Configure default agent instructions for documentation generation.
    """
    try:
        manager = ConfigManager()
        if not manager.load():
            click.secho("\n✗ Configuration not found.", fg="red", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

        config = manager.get_config()

        if clear:
            config.agent_instructions = AgentInstructions()
            manager.save()
            click.secho("✓ Agent instructions cleared", fg="green")
            return

        if not any([include, exclude, focus, doc_type, instructions]):
            click.echo()
            click.secho("Agent Instructions", fg="blue", bold=True)
            agent = config.agent_instructions
            if agent and not agent.is_empty():
                if agent.include_patterns:
                    click.echo(f"  Include:   {', '.join(agent.include_patterns)}")
                if agent.exclude_patterns:
                    click.echo(f"  Exclude:   {', '.join(agent.exclude_patterns)}")
                if agent.focus_modules:
                    click.echo(f"  Focus:     {', '.join(agent.focus_modules)}")
                if agent.doc_type:
                    click.echo(f"  Doc Type:  {agent.doc_type}")
                if agent.custom_instructions:
                    click.echo(f"  Instructions: {agent.custom_instructions}")
            else:
                click.secho("  No custom agent instructions configured", fg="yellow")
            return

        current = config.agent_instructions or AgentInstructions()
        if include is not None:
            current.include_patterns = parse_patterns(include) if include else None
        if exclude is not None:
            current.exclude_patterns = parse_patterns(exclude) if exclude else None
        if focus is not None:
            current.focus_modules = parse_patterns(focus) if focus else None
        if doc_type is not None:
            current.doc_type = doc_type if doc_type else None
        if instructions is not None:
            current.custom_instructions = instructions if instructions else None

        config.agent_instructions = current
        manager.save()
        click.secho("✓ Agent instructions updated successfully.", fg="green", bold=True)

    except Exception as e:
        sys.exit(handle_error(e))
