import logging
import networkx as nx
from networkx.algorithms import community
from typing import Dict, List, Any, Set
from codewiki.src.be.dependency_analyzer.models.core import Node

logger = logging.getLogger(__name__)

def cluster_graph_by_communities(
    components: Dict[str, Node], 
    min_cluster_size: int = 2
) -> Dict[str, Dict[str, Any]]:
    """
    Cluster components into modules using graph community detection.
    
    Args:
        components: Dictionary of component ID to Node
        min_cluster_size: Minimum number of components to form a distinct module
        
    Returns:
        A module tree structure compatible with the rest of CodeWiki
    """
    if not components:
        return {}

    # 1. Build the graph
    G = nx.DiGraph()
    for node_id, node in components.items():
        G.add_node(node_id)
        for dep in node.depends_on:
            if dep in components:
                G.add_edge(node_id, dep)

    # 2. Detect communities
    # We use the undirected version of the graph for community detection
    undirected_G = G.to_undirected()
    
    # Check if graph is empty or has no edges
    if undirected_G.number_of_edges() == 0:
        logger.info("No edges found in graph, falling back to file-path based grouping")
        return _fallback_path_clustering(components)

    try:
        # Use greedy modularity communities as it's efficient for medium-sized graphs
        communities = list(community.greedy_modularity_communities(undirected_G))
        logger.info(f"Detected {len(communities)} communities in dependency graph")
    except Exception as e:
        logger.warning(f"Community detection failed: {e}. Falling back to path clustering.")
        return _fallback_path_clustering(components)

    # 3. Build module tree
    module_tree = {}
    
    for i, comm in enumerate(communities):
        comm_list = list(comm)
        if len(comm_list) < min_cluster_size and len(communities) > 1:
            # Skip very small clusters if we have enough other clusters
            # We'll merge these into an 'others' or closest parent later?
            # For now, let's keep them and name them generically
            pass
            
        # Determine a common path or name for the cluster
        cluster_name = _determine_cluster_name(comm_list, components, i)
        common_path = _get_common_path(comm_list, components)
        
        module_tree[cluster_name] = {
            "path": common_path,
            "components": comm_list,
            "children": {}  # We can recursively cluster if needed, but start flat
        }

    return module_tree

def _determine_cluster_name(node_ids: List[str], components: Dict[str, Node], index: int) -> str:
    """Heuristic to name a cluster based on its most 'central' or representative node."""
    # Simple heuristic: find the node with the most connections in the cluster
    # Or just use the common path
    common_path = _get_common_path(node_ids, components)
    if common_path:
        # e.g., "codewiki/cli" -> "cli_module"
        name = common_path.split('/')[-1] or common_path.split('.')[-1]
        if not name:
            return f"module_{index}"
        return f"{name}_module"
    return f"cluster_{index}"

def _get_common_path(node_ids: List[str], components: Dict[str, Node]) -> str:
    """Find the longest common prefix of the relative paths in the cluster."""
    paths = [components[nid].relative_path for nid in node_ids if nid in components]
    if not paths:
        return ""
    
    # Split paths into parts
    split_paths = [p.replace('\\', '/').split('/') for p in paths]
    
    # Find common prefix
    common_prefix = []
    for parts in zip(*split_paths):
        if all(p == parts[0] for p in parts):
            common_prefix.append(parts[0])
        else:
            break
            
    return "/".join(common_prefix)

def _fallback_path_clustering(components: Dict[str, Node]) -> Dict[str, Dict[str, Any]]:
    """Legacy-style clustering based on directory structure if graph is empty."""
    by_dir = {}
    for node_id, node in components.items():
        dir_path = "/".join(node.relative_path.replace('\\', '/').split('/')[:-1])
        if dir_path not in by_dir:
            by_dir[dir_path] = []
        by_dir[dir_path].append(node_id)
        
    module_tree = {}
    for i, (dir_path, nodes) in enumerate(by_dir.items()):
        name = dir_path.split('/')[-1] if dir_path else "root"
        module_tree[f"{name}_{i}"] = {
            "path": dir_path,
            "components": nodes,
            "children": {}
        }
    return module_tree
