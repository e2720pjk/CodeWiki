from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging
import traceback

logger = logging.getLogger(__name__)

from codewiki.src.be.dependency_analyzer.models.core import Node
from codewiki.src.be.llm_services import call_llm
from codewiki.src.be.utils import count_tokens
from codewiki.src.config import MAX_TOKEN_PER_MODULE, Config
from codewiki.src.be.prompt_template import format_cluster_prompt


def format_potential_core_components(
    leaf_nodes: List[str], components: Dict[str, Node]
) -> tuple[str, str]:
    """
    Format potential core components into a string that can be used in the prompt.
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
        leaf_nodes_by_file[components[leaf_node].relative_path].append(leaf_node)

    potential_core_components = ""
    potential_core_components_with_code = ""
    for file, leaf_nodes in dict(sorted(leaf_nodes_by_file.items())).items():
        potential_core_components += f"# {file}\n"
        potential_core_components_with_code += f"# {file}\n"
        for leaf_node in leaf_nodes:
            potential_core_components += f"\t{leaf_node}\n"
            potential_core_components_with_code += f"\t{leaf_node}\n"
            potential_core_components_with_code += f"{components[leaf_node].source_code}\n"

    return potential_core_components, potential_core_components_with_code


def cluster_modules(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    config: Config,
    current_module_tree: Dict[str, Any] = {},
    current_module_name: Optional[str] = None,
    current_module_path: List[str] = [],
) -> Dict[str, Any]:
    """
    Cluster potential core components into modules.
    [CCR] Relation: Clustering Strategy.
    Reason: Joern enables graph-based clustering (Louvain), while AST-only falls back to LLM heuristics.
    """
    potential_core_components, potential_core_components_with_code = (
        format_potential_core_components(leaf_nodes, components)
    )

    # Skip clustering if module is small enough to fit in the LLM context directly
    token_count = count_tokens(potential_core_components_with_code)
    if token_count <= MAX_TOKEN_PER_MODULE:
        logger.debug(
            f"Skipping clustering for {current_module_name} because the potential core components are few: {token_count} tokens"
        )
        return {}

    module_tree = {}
    use_llm_fallback = True

    # [CCR] Strategy: Enhanced Mode. Reason: Use Joern's graph for deterministic clustering.
    if config.use_joern:
        try:
            from codewiki.src.be.dependency_analyzer.graph_clustering import (
                cluster_graph_by_communities,
            )

            sub_components = {nid: components[nid] for nid in leaf_nodes if nid in components}
            module_tree = cluster_graph_by_communities(sub_components)

            if module_tree and len(module_tree) > 1:
                logger.info(
                    f"🚀 Successfully used graph-based clustering for {current_module_name or 'root'}"
                )
                use_llm_fallback = False
            else:
                logger.debug(
                    "Graph-based clustering produced zero or one cluster, falling back to LLM."
                )
        except Exception as e:
            logger.warning(f"Graph-based clustering failed: {e}. Falling back to LLM.")

    # [CCR] Strategy: Legacy/Fallback Mode. Reason: Use LLM for clustering based on file paths and heuristics.
    if use_llm_fallback:
        prompt = format_cluster_prompt(
            potential_core_components, current_module_tree, current_module_name or "root"
        )
        response = call_llm(prompt, config, model=config.cluster_model)

        # parse the response
        try:
            if "<GROUPED_COMPONENTS>" not in response or "</GROUPED_COMPONENTS>" not in response:
                logger.error(
                    f"Invalid LLM response format - missing component tags: {response[:200]}..."
                )
                return {}

            response_content = response.split("<GROUPED_COMPONENTS>")[1].split(
                "</GROUPED_COMPONENTS>"
            )[0]
            # Use safe literal evaluation if possible, but existing code uses eval
            module_tree = eval(response_content)

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

    # Common Logic: Maintain module tree state for recursive calls
    if current_module_tree == {}:
        # Initial call at root
        for k, v in module_tree.items():
            current_module_tree[k] = v
    else:
        # Recursive call, navigate to the current module's children position
        # Added safety checks to prevent KeyError when nodes don't have "children" key
        value = current_module_tree
        for key in current_module_path:
            if key not in value:
                logger.warning(f"Key '{key}' not found in module tree during navigation")
                break
            if "children" not in value[key]:
                value[key]["children"] = {}  # Initialize missing children
            value = value[key]["children"]
        for module_name, module_info in module_tree.items():
            if "path" in module_info:
                del module_info[
                    "path"
                ]  # Path is usually determined at generation time or stored elsewhere
            value[module_name] = module_info

    # Recursive step: Cluster sub-modules if they are still too large
    for module_name, module_info in module_tree.items():
        sub_leaf_nodes = module_info.get("components", [])

        # Filter sub_leaf_nodes to ensure they exist in components
        valid_sub_leaf_nodes = [node for node in sub_leaf_nodes if node in components]

        current_module_path.append(module_name)
        module_info["children"] = cluster_modules(
            valid_sub_leaf_nodes,
            components,
            config,
            current_module_tree,
            module_name,
            current_module_path,
        )
        current_module_path.pop()

    return module_tree
