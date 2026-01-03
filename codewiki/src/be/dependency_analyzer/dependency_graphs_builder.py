from typing import Dict, List, Any
import os
from codewiki.src.config import Config
from codewiki.src.be.dependency_analyzer.ast_parser import DependencyParser
from codewiki.src.be.dependency_analyzer.topo_sort import (
    build_graph_from_components,
    get_leaf_nodes,
)
from codewiki.src.utils import file_manager

import logging

logger = logging.getLogger(__name__)


class DependencyGraphBuilder:
    """Handles dependency analysis and graph building with optional Joern enhancement."""

    def __init__(self, config: Config):
        self.config = config

        # Initialize analyzer using Factory
        from codewiki.src.be.dependency_analyzer.analysis.analyzer_factory import AnalyzerFactory, AnalyzerType
        
        analyzer_type = AnalyzerType.HYBRID if config.use_joern else AnalyzerType.AST
        self.parser = AnalyzerFactory.create_analyzer(analyzer_type)
        
        logger.info(f"🚀 Using {self.parser.__class__.__name__}")

    def build_dependency_graph(self) -> tuple[Dict[str, Any], List[str]]:
        """
        Build and save dependency graph, returning components and leaf nodes.

        Returns:
            Tuple of (components, leaf_nodes)
        """
        # Ensure output directory exists
        file_manager.ensure_directory(self.config.dependency_graph_dir)

        # Prepare dependency graph path
        repo_name = os.path.basename(os.path.normpath(self.config.repo_path))
        sanitized_repo_name = "".join(c if c.isalnum() else "_" for c in repo_name)
        dependency_graph_path = os.path.join(
            self.config.dependency_graph_dir, f"{sanitized_repo_name}_dependency_graph.json"
        )
        filtered_folders_path = os.path.join(
            self.config.dependency_graph_dir, f"{sanitized_repo_name}_filtered_folders.json"
        )

        filtered_folders = None
        # if os.path.exists(filtered_folders_path):
        #     logger.debug(f"Loading filtered folders from {filtered_folders_path}")
        #     filtered_folders = file_manager.load_json(filtered_folders_path)
        # else:
        #     # Parse repository
        #     filtered_folders = parser.filter_folders()
        #     # Save filtered folders
        #     file_manager.save_json(filtered_folders, filtered_folders_path)

        # Parse repository
        if isinstance(self.parser, DependencyParser):
            components = self.parser.parse_repository(filtered_folders or [])
        else:
            # HybridAnalysisService uses different interface
            # When using Joern/Hybrid, we want to analyze as much as possible unless explicitly limited
            hybrid_limit = 1000 if self.config.use_joern else 100
            result = self.parser.analyze_repository_hybrid(
                repo_path=self.config.repo_path, max_files=hybrid_limit
            )
            # Convert hybrid result to expected format
            raw_nodes = result.get("nodes", {})
            components = {}
            from codewiki.src.be.dependency_analyzer.models.core import Node, EnhancedNode
            NodeClass = EnhancedNode if self.config.use_joern else Node
            
            # Helper to normalize nodes whether they come as Dict[id, data] or List[data]
            node_items = []
            if isinstance(raw_nodes, list):
                # If list, assume list of node data dicts
                for node_data in raw_nodes:
                    node_id = node_data.get("id") or node_data.get("name")
                    if node_id:
                        node_items.append((node_id, node_data))
            else:
                # If dict, assume id -> data mapping
                node_items = raw_nodes.items()

            for node_id, node_data in node_items:
                if isinstance(node_data, dict):
                    try:
                        components[node_id] = NodeClass(**node_data)
                    except Exception as e:
                        logger.warning(f"Failed to convert node {node_id} to {NodeClass.__name__} object: {e}")
                else:
                    components[node_id] = node_data

        # Save dependency graph
        if isinstance(self.parser, DependencyParser):
            self.parser.save_dependency_graph(dependency_graph_path)
        else:
            # HybridAnalysisService doesn't have save_dependency_graph - save manually
            file_manager.save_json({comp_id: comp.model_dump() for comp_id, comp in components.items()}, dependency_graph_path)

        # Build graph for traversal
        graph = build_graph_from_components(components)

        # Get leaf nodes
        leaf_nodes = get_leaf_nodes(graph, components)

        # check if leaf_nodes are in components, only keep the ones that are in components
        # and type is one of the following: class, interface, struct (or function for C-based projects)

        # Determine if we should include functions based on available component types
        available_types = set()
        for comp in components.values():
            available_types.add(comp.component_type)

        # Valid types for leaf nodes - include functions for C-based codebases
        valid_types = {"class", "interface", "struct"}
        # If no classes/interfaces/structs are found, include functions
        if not available_types.intersection(valid_types):
            valid_types.add("function")

        keep_leaf_nodes = []
        for leaf_node in leaf_nodes:
            # Skip any leaf nodes that are clearly error strings or invalid identifiers
            if (
                not isinstance(leaf_node, str)
                or leaf_node.strip() == ""
                or any(
                    err_keyword in leaf_node.lower()
                    for err_keyword in ["error", "exception", "failed", "invalid"]
                )
            ):
                logger.warning(f"Skipping invalid leaf node identifier: '{leaf_node}'")
                continue

            if leaf_node in components:
                if components[leaf_node].component_type in valid_types:
                    keep_leaf_nodes.append(leaf_node)
                else:
                    # logger.debug(f"Leaf node {leaf_node} is a {components[leaf_node].component_type}, removing it")
                    pass
            else:
                logger.warning(f"Leaf node {leaf_node} not found in components, removing it")

        return components, keep_leaf_nodes
