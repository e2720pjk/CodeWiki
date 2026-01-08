from typing import List, Dict, Any, Optional
from collections import defaultdict
import traceback
import ast

from codewiki.src.be.logging_config import get_logger
from codewiki.src.be.dependency_analyzer.models.core import Node
from codewiki.src.be.llm_services import call_llm
from codewiki.src.be.utils import count_tokens
from codewiki.src.config import MAX_TOKEN_PER_MODULE, Config
from codewiki.src.be.prompt_template import format_cluster_prompt

logger = get_logger(__name__)


def _resolve_component_name(simple_name: str, components: Dict[str, Node]) -> Optional[str]:
    """
    Resolve a simple component name to its fully qualified component ID.
    
    Tries multiple strategies to match:
    1. Exact match (if the simple_name is already fully qualified)
    2. Match by Node.name attribute
    3. Match by ID suffix pattern (endswith `.{simple_name}`)
    
    Args:
        simple_name: The component name to resolve (may be simple or fully qualified)
        components: Dictionary mapping component IDs to Node objects
        
    Returns:
        Fully qualified component ID if found, None otherwise
    """
    if simple_name in components:
        return simple_name
    
    for comp_id, node in components.items():
        if node.name == simple_name:
            return comp_id
        if comp_id.endswith(f".{simple_name}"):
            return comp_id
    
    return None


def format_potential_core_components(
    leaf_nodes: List[str], components: Dict[str, Node]
) -> tuple[str, str]:
    """
    Format the potential core components into a string that can be used in the prompt.
    """
    # Filter out any invalid leaf nodes that don't exist in components
    valid_leaf_nodes = []
    for leaf_node in leaf_nodes:
        if leaf_node in components:
            valid_leaf_nodes.append(leaf_node)
        else:
            logger.warning(f"Skipping invalid leaf node '{leaf_node}' - not found in components")

    # group leaf nodes by file
    leaf_nodes_by_file = defaultdict(list)
    for leaf_node in valid_leaf_nodes:
        component = components[leaf_node]
        leaf_nodes_by_file[component.relative_path].append(leaf_node)

    potential_core_components = ""
    potential_core_components_with_code = ""
    for file, leaf_nodes in dict(sorted(leaf_nodes_by_file.items())).items():
        potential_core_components += f"# {file}\n"
        potential_core_components_with_code += f"# {file}\n"
        for leaf_node in leaf_nodes:
            potential_core_components += f"\t{leaf_node}\n"
            potential_core_components_with_code += f"\t{leaf_node}\n"
            component = components[leaf_node]
            potential_core_components_with_code += f"{component.source_code}\n"

    return potential_core_components, potential_core_components_with_code


def cluster_modules(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    config: Config,
    current_module_tree: dict[str, Any] = {},
    current_module_name: Optional[str] = None,
    current_module_path: Optional[List[str]] = None,
    validation_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Cluster the potential core components into modules.
    """
    if current_module_path is None:
        current_module_path = []
    
    if validation_stats is None:
        validation_stats = {"fuzzy_matched": {}, "invalid": {}}
    
    potential_core_components, potential_core_components_with_code = (
        format_potential_core_components(leaf_nodes, components)
    )

    if count_tokens(potential_core_components_with_code) <= MAX_TOKEN_PER_MODULE:
        logger.debug(
            f"Skipping clustering for {current_module_name} because the potential core components are too few: {count_tokens(potential_core_components_with_code)} tokens"
        )
        return {}

    prompt = format_cluster_prompt(
        potential_core_components, current_module_tree, current_module_name
    )
    response = call_llm(prompt, config, model=config.cluster_model)

    # parse the response
    try:
        if "<GROUPED_COMPONENTS>" not in response or "</GROUPED_COMPONENTS>" not in response:
            logger.error(
                f"Invalid LLM response format - missing component tags: {response[:200]}..."
            )
            return {}

        response_content = response.split("<GROUPED_COMPONENTS>")[1].split("</GROUPED_COMPONENTS>")[
            0
        ]
        module_tree = ast.literal_eval(response_content)

        if not isinstance(module_tree, dict):
            logger.error(f"Invalid module tree format - expected dict, got {type(module_tree)}")
            return {}

    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}. Response: {response[:200]}...")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

    # check if the module tree is valid
    if len(module_tree) <= 1:
        logger.debug(
            f"Skipping clustering for {current_module_name} because the module tree is too small: {len(module_tree)} modules"
        )
        return {}

    if current_module_tree == {}:
        current_module_tree = module_tree
    else:
        value = current_module_tree
        for key in current_module_path:
            value = value[key]["children"]
        for module_name, module_info in module_tree.items():
            del module_info["path"]
            value[module_name] = module_info

    for module_name, module_info in module_tree.items():
        sub_leaf_nodes = module_info.get("components", [])

        # Filter sub_leaf_nodes to ensure they exist in components
        valid_sub_leaf_nodes = []
        for node in sub_leaf_nodes:
            if node in components:
                valid_sub_leaf_nodes.append(node)
            else:
                resolved_name = _resolve_component_name(node, components)
                if resolved_name:
                    valid_sub_leaf_nodes.append(resolved_name)
                    logger.debug(
                        f"Fuzzy matched '{node}' to '{resolved_name}' in module '{module_name}'"
                    )
                    if module_name not in validation_stats["fuzzy_matched"]:
                        validation_stats["fuzzy_matched"][module_name] = []
                    validation_stats["fuzzy_matched"][module_name].append((node, resolved_name))
                else:
                    logger.warning(
                        f"Skipping invalid sub leaf node '{node}' in module '{module_name}' - not found in {len(components)} available components. This may indicate the LLM created a non-existent component ID (hallucination or instruction-following issue)"
                    )
                    if module_name not in validation_stats["invalid"]:
                        validation_stats["invalid"][module_name] = []
                    validation_stats["invalid"][module_name].append(node)

        current_module_path.append(module_name)
        module_info["children"] = {}
        module_info["children"] = cluster_modules(
            valid_sub_leaf_nodes,
            components,
            config,
            current_module_tree,
            module_name,
            current_module_path,
            validation_stats,
        )
        current_module_path.pop()

    # Log summary at top-level completion
    if current_module_name is None and validation_stats["invalid"]:
        total_invalid = sum(len(v) for v in validation_stats["invalid"].values())
        logger.error(
            f"Clustering completed with {total_invalid} invalid component references across {len(validation_stats['invalid'])} modules"
        )
        for module_name, components in validation_stats["invalid"].items():
            comp_list = list(components)
            logger.error(f"  Module '{module_name}': {len(comp_list)} invalid components: {comp_list[:5]}{'...' if len(comp_list) > 5 else ''}")

    return module_tree
