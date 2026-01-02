"""
Hybrid Analysis Service - Combining AST and Joern CPG Analysis

This service provides enhanced analysis by combining:
1. AST-based structural analysis (stable, fast)
2. Joern CPG-based data flow analysis (deep insights)
"""

import logging
import json
import os
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

from codewiki.src.be.dependency_analyzer.analysis.analysis_service import AnalysisService
from codewiki.src.be.dependency_analyzer.models.core import Node, DataFlowRelationship
from codewiki.src.be.dependency_analyzer.simplified_joern import SimplifiedJoernAnalyzer

logger = logging.getLogger(__name__)


class HybridAnalysisService:
    """
    Hybrid analysis service that combines AST and Joern CPG analysis.

    Strategy:
    1. Use AST for stable structural analysis (functions, classes, basic relationships)
    2. Enhance with Joern for data flow and cross-module dependencies
    3. Graceful fallback when Joern is unavailable
    """

    def __init__(self, enable_joern: bool = True):
        """
        Initialize hybrid analysis service.

        Args:
            enable_joern: Whether to enable Joern analysis (default: True)
        """
        self.ast_service = AnalysisService()
        self.joern_analyzer = SimplifiedJoernAnalyzer() if enable_joern else None
        self.enable_joern = enable_joern

        logger.info(
            f"HybridAnalysisService initialized (Joern: {'enabled' if enable_joern else 'disabled'})"
        )

    def analyze_repository_hybrid(
        self,
        repo_path: str,
        max_files: int = 100,
        languages: Optional[List[str]] = None,
        include_data_flow: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform hybrid repository analysis.

        Args:
            repo_path: Path to repository to analyze
            max_files: Maximum number of files to analyze
            languages: List of languages to include
            include_data_flow: Whether to include data flow analysis

        Returns:
            Combined analysis results with AST + Joern enhancements
        """
        start_time = time.time()

        try:
            logger.info(f"Starting hybrid analysis of {repo_path}")

            # Phase 1: AST Analysis (always runs - provides stable foundation)
            logger.info("Phase 1: AST structural analysis")
            ast_result = self._run_ast_analysis(repo_path, max_files, languages)

            # Phase 2: Joern Enhancement (if enabled and available)
            joern_enhancement = None
            if self.enable_joern and include_data_flow:
                logger.info("Phase 2: Joern CPG enhancement")
                joern_enhancement = self._run_joern_enhancement(repo_path, ast_result)

            # Phase 3: Merge results
            merged_result = self._merge_analysis_results(ast_result, joern_enhancement)

            analysis_time = time.time() - start_time
            logger.info(f"Hybrid analysis completed in {analysis_time:.2f} seconds")

            # Add metadata
            merged_result["metadata"] = {
                "analysis_time": analysis_time,
                "ast_functions": len(ast_result.get("nodes", {})),
                "joern_enhanced": joern_enhancement is not None,
                "data_flow_relationships": len(merged_result.get("data_flow_relationships", [])),
                "analysis_type": "hybrid_ast_joern",
            }

            return merged_result

        except Exception as e:
            logger.error(f"Hybrid analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Hybrid analysis failed: {str(e)}")

    def _run_ast_analysis(
        self, repo_path: str, max_files: int, languages: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Run AST-based analysis using existing AnalysisService.

        Args:
            repo_path: Repository path
            max_files: Maximum files to analyze
            languages: Language filters

        Returns:
            AST analysis results
        """
        try:
            result = self.ast_service.analyze_local_repository(
                repo_path=repo_path, max_files=max_files, languages=languages
            )

            logger.info(f"AST analysis found {result['summary']['total_nodes']} nodes")
            return result

        except Exception as e:
            logger.error(f"AST analysis failed: {str(e)}")
            # Return minimal result to allow analysis to continue
            return {
                "nodes": {},
                "relationships": [],
                "summary": {"total_nodes": 0, "total_relationships": 0},
                "status": "ast_failed",
                "error": str(e),
            }

    def _run_joern_enhancement(self, repo_path: str, ast_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Joern CPG analysis to enhance AST results.

        Args:
            repo_path: Repository path
            ast_result: Results from AST analysis

        Returns:
            Joern enhancement results
        """
        if not self.joern_analyzer:
            logger.warning("Joern analyzer not available, skipping enhancement")
            return {}

        try:
            # Get basic Joern analysis
            joern_result = self.joern_analyzer.analyze_repository_basic(repo_path)

            # Extract data flow for key functions
            data_flows = self._extract_data_flows_for_functions(
                repo_path, ast_result.get("nodes", {})
            )

            # Extract cross-module relationships
            cross_module_edges = self._extract_cross_module_relationships(
                ast_result
            )

            enhancement = {
                "joern_stats": joern_result,
                "data_flows": data_flows,
                "cross_module_edges": cross_module_edges,
                "enhanced_functions": len(data_flows),
                "cross_module_count": len(cross_module_edges),
            }

            logger.info(
                f"Joern enhancement: {enhancement['enhanced_functions']} functions, {enhancement['cross_module_count']} cross-module edges"
            )
            return enhancement

        except Exception as e:
            logger.warning(f"Joern enhancement failed: {str(e)}")
            return {"status": "joern_failed", "error": str(e)}

    def _extract_data_flows_for_functions(
        self, repo_path: str, ast_nodes: Dict[str, Any]
    ) -> List[DataFlowRelationship]:
        """
        Extract data flow relationships for key functions.

        Args:
            repo_path: Repository path
            ast_nodes: AST nodes from analysis

        Returns:
            List of data flow relationships
        """
        if not self.joern_analyzer:
            return []

        data_flows = []

        # Select a subset of important functions for data flow analysis
        important_functions = self._select_important_functions(ast_nodes, limit=20)

        for func_name, _func_info in important_functions.items():
            try:
                df_result = self.joern_analyzer.extract_data_flow_sample(repo_path, func_name)

                if df_result.get("status") != "error":
                    # Create data flow relationships
                    params = df_result.get("parameters", [])
                    locals = df_result.get("local_variables", [])

                    # Parameter to local variable flows
                    for param in params:
                        for local in locals:
                            data_flow = DataFlowRelationship(
                                source=f"{func_name}:{param}",
                                target=f"{func_name}:{local}",
                                flow_type="parameter_to_local",
                                variable_name=local,
                                confidence=0.8,
                            )
                            data_flows.append(data_flow)

            except Exception as e:
                logger.debug(f"Data flow analysis failed for {func_name}: {e}")
                continue

        return data_flows

    def _extract_cross_module_relationships(
        self, ast_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract cross-module relationships that AST might miss.

        Args:
            ast_result: Full AST analysis result

        Returns:
            List of cross-module relationship data
        """
        cross_module = []

        # Simple heuristic: identify calls that span different modules
        relationships = ast_result.get("relationships", [])

        for rel in relationships:
            caller = rel.get("caller", "")
            callee = rel.get("callee", "")

            if caller and callee:
                # Extract module paths from identifiers
                caller_module = ".".join(caller.split(".")[:-1]) if "." in caller else ""
                callee_module = ".".join(callee.split(".")[:-1]) if "." in callee else ""

                if caller_module and callee_module and caller_module != callee_module:
                    cross_module.append(
                        {
                            "caller": caller,
                            "callee": callee,
                            "caller_module": caller_module,
                            "callee_module": callee_module,
                            "relationship_type": "cross_module_call",
                            "confidence": 0.9,
                        }
                    )

        return cross_module

    def _select_important_functions(
        self, ast_nodes: Dict[str, Any], limit: int = 20
    ) -> Dict[str, Any]:
        """
        Select important functions for detailed analysis.

        Args:
            ast_nodes: AST nodes
            limit: Maximum number of functions to select

        Returns:
            Selected important functions
        """
        functions = {}

        # Prioritize functions by:
        # 1. Number of dependents (popularity)
        # 2. Presence of docstring (likely important)
        # 3. Name patterns (init, main, etc.)

        scored_functions = []

        for node_id, node_data in ast_nodes.items():
            if node_data.get("component_type") in ["function", "method"]:
                score = 0

                # Popularity score (number of dependents)
                dependents = node_data.get("dependents", [])
                score += len(dependents) * 2

                # Documentation score
                if node_data.get("has_docstring"):
                    score += 3

                # Name pattern scores
                name = node_data.get("name", "").lower()
                if name in ["init", "main", "run", "start", "process"]:
                    score += 5
                elif name.startswith("test_"):
                    score += 1

                # Complexity estimation (parameters)
                params = node_data.get("parameters", [])
                score += len(params) * 0.5

                scored_functions.append((score, node_id, node_data))

        # Sort by score and select top N
        scored_functions.sort(key=lambda x: x[0], reverse=True)

        for _, node_id, node_data in scored_functions[:limit]:
            functions[node_data.get("name", node_id)] = node_data

        return functions

    def _merge_analysis_results(
        self, ast_result: Dict[str, Any], joern_enhancement: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge AST and Joern results into comprehensive analysis.

        Args:
            ast_result: Results from AST analysis
            joern_enhancement: Results from Joern analysis

        Returns:
            Merged comprehensive results
        """
        merged = {
            "nodes": ast_result.get("nodes", {}),
            "relationships": ast_result.get("relationships", []),
            "data_flow_relationships": [],
            "cross_module_relationships": [],
            "summary": {
                **ast_result.get("summary", {}),
                "enhancement_applied": joern_enhancement is not None,
            },
        }

        if joern_enhancement and joern_enhancement.get("status") != "joern_failed":
            # Add data flow relationships
            data_flows = joern_enhancement.get("data_flows", [])
            merged["data_flow_relationships"] = [df.model_dump() for df in data_flows]

            # Add cross-module relationships
            cross_module = joern_enhancement.get("cross_module_edges", [])
            merged["cross_module_relationships"] = cross_module

            # Update summary
            merged["summary"].update(
                {
                    "data_flow_relationships": len(data_flows),
                    "cross_module_relationships": len(cross_module),
                    "joern_enhanced_functions": joern_enhancement.get("enhanced_functions", 0),
                }
            )

        return merged

    def analyze_single_function_with_data_flow(
        self, repo_path: str, function_name: str
    ) -> Dict[str, Any]:
        """
        Perform detailed analysis of a single function with data flow.

        Args:
            repo_path: Repository path
            function_name: Function name to analyze

        Returns:
            Detailed function analysis with data flow
        """
        try:
            # Get AST info for the function
            ast_result = self._run_ast_analysis(repo_path, 50, None)

            function_data = None
            for node_id, node in ast_result.get("nodes", {}).items():
                if node.get("name") == function_name:
                    function_data = node
                    break

            if not function_data:
                return {"error": f"Function {function_name} not found in AST analysis"}

            # Add data flow if available
            data_flow_result = {}
            if self.joern_analyzer:
                data_flow_result = self.joern_analyzer.extract_data_flow_sample(
                    repo_path, function_name
                )

            return {
                "function": function_data,
                "data_flow": data_flow_result,
                "analysis_type": "single_function_detailed",
            }

        except Exception as e:
            return {"error": f"Function analysis failed: {str(e)}"}


# Factory function for backward compatibility
def create_hybrid_analysis_service(enable_joern: bool = True) -> HybridAnalysisService:
    """
    Create hybrid analysis service with specified configuration.

    Args:
        enable_joern: Whether to enable Joern analysis

    Returns:
        HybridAnalysisService instance
    """
    return HybridAnalysisService(enable_joern=enable_joern)
