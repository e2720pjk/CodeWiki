"""
Call Graph Analyzer

Central orchestrator for multi-language call graph analysis.
Coordinates language-specific analyzers to build comprehensive call graphs
across different programming languages in a repository.
"""

from typing import Dict, List
import logging
import os
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

from codewiki.src.be.dependency_analyzer.models.core import Node, CallRelationship
from codewiki.src.be.dependency_analyzer.utils.patterns import CODE_EXTENSIONS
from codewiki.src.be.dependency_analyzer.utils.security import safe_open_text

logger = logging.getLogger(__name__)


class CallGraphAnalyzer:
    def __init__(self):
        """Initialize the call graph analyzer."""
        self.functions: Dict[str, Node] = {}
        self.call_relationships: List[CallRelationship] = []
        logger.debug("CallGraphAnalyzer initialized.")

    def analyze_code_files(
        self, code_files: List[Dict], base_dir: str, enable_parallel: bool = True
    ) -> Dict:
        """
        Complete analysis: Analyze all files to build complete call graph with all nodes.

        This approach:
        1. Analyzes all code files (sequentially or in parallel)
        2. Extracts all functions and relationships
        3. Builds complete call graph
        4. Returns all nodes and relationships

        Args:
            code_files: List of file information dictionaries
            base_dir: Base directory path
            enable_parallel: Whether to use parallel processing (default: True)
        """
        logger.debug(f"Starting analysis of {len(code_files)} files (parallel={enable_parallel})")

        self.functions = {}
        self.call_relationships = []

        if enable_parallel and len(code_files) > 1:
            return self._analyze_parallel(code_files, base_dir)
        else:
            return self._analyze_sequential(code_files, base_dir)

    def _analyze_parallel(self, code_files: List[Dict], base_dir: str) -> Dict:
        """Parallel file analysis implementation."""
        logger.debug(f"Starting parallel analysis of {len(code_files)} files")

        # Group files by language to reduce parser pool contention
        files_by_language = defaultdict(list)
        for file_info in code_files:
            files_by_language[file_info["language"]].append(file_info)

        # Initialize shared state (thread-safe collections needed)
        functions_lock = threading.Lock()
        relationships_lock = threading.Lock()

        self.functions = {}
        self.call_relationships = []

        # Process languages in parallel
        max_workers = min(os.cpu_count() or 4, len(files_by_language), 8)
        logger.debug(f"Using {max_workers} workers for parallel analysis (CPU cores: {os.cpu_count()}, language groups: {len(files_by_language)})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each language group
            future_to_language = {
                executor.submit(
                    self._analyze_language_files,
                    lang,
                    files,
                    base_dir,
                    functions_lock,
                    relationships_lock,
                ): lang
                for lang, files in files_by_language.items()
            }

            # Collect results
            for future in as_completed(future_to_language):
                lang = future_to_language[future]
                try:
                    result = future.result()
                    logger.debug(f"Completed analysis for {lang}: {result}")
                except Exception as e:
                    logger.error(f"Failed to analyze {lang} files: {e}")
                    # Continue with other languages

        # Continue with existing relationship resolution
        logger.debug("Resolving call relationships")
        self._resolve_call_relationships()
        self._deduplicate_relationships()
        viz_data = self._generate_visualization_data()

        return {
            "call_graph": {
                "total_functions": len(self.functions),
                "total_calls": len(self.call_relationships),
                "languages_found": list(set(f.get("language") for f in code_files)),
                "files_analyzed": len(code_files),
                "analysis_approach": "parallel",
            },
            "functions": [func.model_dump() for func in self.functions.values()],
            "relationships": [rel.model_dump() for rel in self.call_relationships],
            "visualization": viz_data,
        }

    def _analyze_sequential(self, code_files: List[Dict], base_dir: str) -> Dict:
        """Sequential file analysis implementation (fallback)."""
        logger.debug(f"Starting sequential analysis of {len(code_files)} files")

        files_analyzed = 0
        for file_info in code_files:
            logger.debug(f"Analyzing: {file_info['path']}")
            self._analyze_code_file(base_dir, file_info)
            files_analyzed += 1

        logger.debug("Resolving call relationships")
        self._resolve_call_relationships()
        self._deduplicate_relationships()
        viz_data = self._generate_visualization_data()

        return {
            "call_graph": {
                "total_functions": len(self.functions),
                "total_calls": len(self.call_relationships),
                "languages_found": list(set(f.get("language") for f in code_files)),
                "files_analyzed": files_analyzed,
                "analysis_approach": "sequential",
            },
            "functions": [func.model_dump() for func in self.functions.values()],
            "relationships": [rel.model_dump() for rel in self.call_relationships],
            "visualization": viz_data,
        }

    def _analyze_language_files(
        self,
        language: str,
        files: List[Dict],
        base_dir: str,
        functions_lock: threading.Lock,
        relationships_lock: threading.Lock,
    ) -> Dict:
        """Analyze files for a specific language."""
        logger.debug(f"Analyzing {len(files)} {language} files")

        local_functions = {}
        local_relationships = []

        for file_info in files:
            try:
                # Use existing _analyze_code_file logic but return results
                file_functions, file_relationships = self._analyze_code_file_safe(
                    base_dir, file_info
                )
                local_functions.update(file_functions)
                local_relationships.extend(file_relationships)
            except Exception as e:
                logger.error(f"Failed to analyze {file_info['path']}: {e}")

        # Thread-safe update of shared collections
        with functions_lock:
            self.functions.update(local_functions)

        with relationships_lock:
            self.call_relationships.extend(local_relationships)

        return {
            "language": language,
            "functions_count": len(local_functions),
            "relationships_count": len(local_relationships),
        }

    def _analyze_code_file_safe(
        self, base_dir: str, file_info: Dict
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version of _analyze_code_file that returns results."""
        base = Path(base_dir)
        file_path = base / file_info["path"]

        try:
            content = safe_open_text(base, file_path)
            language = file_info["language"]

            functions = {}
            relationships = []

            if language == "python":
                functions, relationships = self._analyze_python_file_safe(
                    file_path, content, base_dir
                )
            elif language == "javascript":
                functions, relationships = self._analyze_javascript_file_safe(
                    file_path, content, base_dir
                )
            elif language == "typescript":
                functions, relationships = self._analyze_typescript_file_safe(
                    file_path, content, base_dir
                )
            elif language == "java":
                functions, relationships = self._analyze_java_file_safe(
                    file_path, content, base_dir
                )
            elif language == "csharp":
                functions, relationships = self._analyze_csharp_file_safe(
                    file_path, content, base_dir
                )
            elif language == "c":
                functions, relationships = self._analyze_c_file_safe(file_path, content, base_dir)
            elif language == "cpp":
                functions, relationships = self._analyze_cpp_file_safe(file_path, content, base_dir)

            return functions, relationships

        except Exception as e:
            logger.error(f"⚠️ Error analyzing {file_path}: {str(e)}")
            return {}, []

    def extract_code_files(self, file_tree: Dict) -> List[Dict]:
        """
        Extract code files from file tree structure.

        Filters files based on supported extensions and excludes test/config files.

        Args:
            file_tree: Nested dictionary representing file structure

        Returns:
            List of code file information dictionaries
        """
        code_files = []

        def traverse(tree):
            if tree["type"] == "file":
                ext = tree.get("extension", "").lower()
                if ext in CODE_EXTENSIONS:
                    name = tree["name"].lower()
                    if not any(skip in name for skip in []):
                        code_files.append(
                            {
                                "path": tree["path"],
                                "name": tree["name"],
                                "extension": ext,
                                "language": CODE_EXTENSIONS[ext],
                            }
                        )
            elif tree["type"] == "directory" and tree.get("children"):
                for child in tree["children"]:
                    traverse(child)

        traverse(file_tree)
        return code_files

    def _analyze_code_file(self, repo_dir: str, file_info: Dict):
        """
        Analyze a single code file based on its language.

        Routes to appropriate language-specific analyzer.

        Args:
            repo_dir: Repository directory path
            file_info: File information dictionary
        """

        base = Path(repo_dir)
        file_path = base / file_info["path"]

        try:
            content = safe_open_text(base, file_path)
            language = file_info["language"]
            if language == "python":
                self._analyze_python_file(file_path, content, repo_dir)
            elif language == "javascript":
                self._analyze_javascript_file(file_path, content, repo_dir)
            elif language == "typescript":
                self._analyze_typescript_file(file_path, content, repo_dir)
            elif language == "java":
                self._analyze_java_file(file_path, content, repo_dir)
            elif language == "csharp":
                self._analyze_csharp_file(file_path, content, repo_dir)
            elif language == "c":
                self._analyze_c_file(file_path, content, repo_dir)
            elif language == "cpp":
                self._analyze_cpp_file(file_path, content, repo_dir)
            elif language == "php":
                self._analyze_php_file(file_path, content, repo_dir)
            # else:
            #     logger.warning(
            #         f"Unsupported language for call graph analysis: {language} for file {file_path}"
            #     )

        except Exception as e:
            logger.error(f"⚠️ Error analyzing {file_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _analyze_python_file(self, file_path: str, content: str, base_dir: str):
        """
        Analyze Python file using Python AST analyzer.

        Args:
            file_path: Relative path to Python file
            content: File content string
            base_dir: Repository base directory path
        """
        from codewiki.src.be.dependency_analyzer.analyzers.python import analyze_python_file

        try:
            functions, relationships = analyze_python_file(file_path, content, repo_path=base_dir)

            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                self.functions[func_id] = func

            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(f"Failed to analyze Python file {file_path}: {e}", exc_info=True)

    def _analyze_python_file_safe(
        self, file_path: str, content: str, base_dir: str
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version that returns results instead of modifying shared state."""
        from codewiki.src.be.dependency_analyzer.analyzers.python import analyze_python_file

        try:
            functions, relationships = analyze_python_file(file_path, content, repo_path=base_dir)

            function_dict = {}
            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                function_dict[func_id] = func

            return function_dict, relationships
        except Exception as e:
            logger.error(f"Failed to analyze Python file {file_path}: {e}", exc_info=True)
            return {}, []

    def _analyze_javascript_file(self, file_path: str, content: str, repo_dir: str):
        """
        Analyze JavaScript file using tree-sitter based AST analyzer

        Args:
            file_path: Relative path to JavaScript file
            content: File content string
            repo_dir: Repository base directory
        """
        try:
            from codewiki.src.be.dependency_analyzer.analyzers.javascript import (
                analyze_javascript_file_treesitter,
            )

            functions, relationships = analyze_javascript_file_treesitter(
                file_path, content, repo_path=repo_dir
            )

            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                self.functions[func_id] = func

            self.call_relationships.extend(relationships)

        except Exception as e:
            logger.error(f"Failed to analyze JavaScript file {file_path}: {e}", exc_info=True)

    def _analyze_javascript_file_safe(
        self, file_path: str, content: str, repo_dir: str
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version that returns results instead of modifying shared state."""
        try:
            from codewiki.src.be.dependency_analyzer.analyzers.javascript import (
                analyze_javascript_file_treesitter,
            )

            functions, relationships = analyze_javascript_file_treesitter(
                file_path, content, repo_path=repo_dir
            )

            function_dict = {}
            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                function_dict[func_id] = func

            return function_dict, relationships

        except Exception as e:
            logger.error(f"Failed to analyze JavaScript file {file_path}: {e}", exc_info=True)
            return {}, []

    def _analyze_typescript_file(self, file_path: str, content: str, repo_dir: str):
        """
        Analyze TypeScript file using tree-sitter based AST analyzer

        Args:
            file_path: Relative path to TypeScript file
            content: File content string
        """
        try:
            from codewiki.src.be.dependency_analyzer.analyzers.typescript import (
                analyze_typescript_file_treesitter,
            )

            functions, relationships = analyze_typescript_file_treesitter(
                file_path, content, repo_path=repo_dir
            )

            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                self.functions[func_id] = func

            self.call_relationships.extend(relationships)

        except Exception as e:
            logger.error(f"Failed to analyze TypeScript file {file_path}: {e}", exc_info=True)

    def _analyze_typescript_file_safe(
        self, file_path: str, content: str, repo_dir: str
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version that returns results instead of modifying shared state."""
        try:
            from codewiki.src.be.dependency_analyzer.analyzers.typescript import (
                analyze_typescript_file_treesitter,
            )

            functions, relationships = analyze_typescript_file_treesitter(
                file_path, content, repo_path=repo_dir
            )

            function_dict = {}
            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                function_dict[func_id] = func

            return function_dict, relationships

        except Exception as e:
            logger.error(f"Failed to analyze TypeScript file {file_path}: {e}", exc_info=True)
            return {}, []

    def _analyze_c_file(self, file_path: str, content: str, repo_dir: str):
        """
        Analyze C file using tree-sitter based analyzer.

        Args:
            file_path: Relative path to C file
            content: File content string
            repo_dir: Repository base directory
        """
        from codewiki.src.be.dependency_analyzer.analyzers.c import analyze_c_file

        functions, relationships = analyze_c_file(file_path, content, repo_path=repo_dir)

        for func in functions:
            func_id = func.id if func.id else f"{file_path}:{func.name}"
            self.functions[func_id] = func

        self.call_relationships.extend(relationships)

    def _analyze_c_file_safe(
        self, file_path: str, content: str, repo_dir: str
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version that returns results instead of modifying shared state."""
        from codewiki.src.be.dependency_analyzer.analyzers.c import analyze_c_file

        functions, relationships = analyze_c_file(file_path, content, repo_path=repo_dir)

        function_dict = {}
        for func in functions:
            func_id = func.id if func.id else f"{file_path}:{func.name}"
            function_dict[func_id] = func

        return function_dict, relationships

    def _analyze_cpp_file(self, file_path: str, content: str, repo_dir: str):
        """
        Analyze C++ file using tree-sitter based analyzer.

        Args:
            file_path: Relative path to C++ file
            content: File content string
        """
        from codewiki.src.be.dependency_analyzer.analyzers.cpp import analyze_cpp_file

        functions, relationships = analyze_cpp_file(file_path, content, repo_path=repo_dir)

        for func in functions:
            func_id = func.id if func.id else f"{file_path}:{func.name}"
            self.functions[func_id] = func

        self.call_relationships.extend(relationships)

    def _analyze_cpp_file_safe(
        self, file_path: str, content: str, repo_dir: str
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version that returns results instead of modifying shared state."""
        from codewiki.src.be.dependency_analyzer.analyzers.cpp import analyze_cpp_file

        functions, relationships = analyze_cpp_file(file_path, content, repo_path=repo_dir)

        function_dict = {}
        for func in functions:
            func_id = func.id if func.id else f"{file_path}:{func.name}"
            function_dict[func_id] = func

        return function_dict, relationships

    def _analyze_java_file(self, file_path: str, content: str, repo_dir: str):
        """
        Analyze Java file using tree-sitter based analyzer.

        Args:
            file_path: Relative path to Java file
            content: File content string
            repo_dir: Repository base directory
        """
        from codewiki.src.be.dependency_analyzer.analyzers.java import analyze_java_file

        try:
            functions, relationships = analyze_java_file(file_path, content, repo_path=repo_dir)
            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                self.functions[func_id] = func

            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(f"Failed to analyze Java file {file_path}: {e}", exc_info=True)

    def _analyze_java_file_safe(
        self, file_path: str, content: str, repo_dir: str
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version that returns results instead of modifying shared state."""
        from codewiki.src.be.dependency_analyzer.analyzers.java import analyze_java_file

        try:
            functions, relationships = analyze_java_file(file_path, content, repo_path=repo_dir)

            function_dict = {}
            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                function_dict[func_id] = func

            return function_dict, relationships
        except Exception as e:
            logger.error(f"Failed to analyze Java file {file_path}: {e}", exc_info=True)
            return {}, []

    def _analyze_csharp_file(self, file_path: str, content: str, repo_dir: str):
        """
        Analyze C# file using tree-sitter based analyzer.

        Args:
            file_path: Relative path to C# file
            content: File content string
            repo_dir: Repository base directory
        """
        from codewiki.src.be.dependency_analyzer.analyzers.csharp import analyze_csharp_file

        try:
            functions, relationships = analyze_csharp_file(file_path, content, repo_path=repo_dir)

            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                self.functions[func_id] = func

            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(f"Failed to analyze C# file {file_path}: {e}", exc_info=True)

    def _analyze_csharp_file_safe(
        self, file_path: str, content: str, repo_dir: str
    ) -> tuple[Dict[str, Node], List[CallRelationship]]:
        """Thread-safe version that returns results instead of modifying shared state."""
        from codewiki.src.be.dependency_analyzer.analyzers.csharp import analyze_csharp_file

        try:
            functions, relationships = analyze_csharp_file(file_path, content, repo_path=repo_dir)

            function_dict = {}
            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                function_dict[func_id] = func

            return function_dict, relationships
        except Exception as e:
            logger.error(f"Failed to analyze C# file {file_path}: {e}", exc_info=True)
            return {}, []

    def _analyze_php_file(self, file_path: str, content: str, repo_dir: str):
        """
        Analyze PHP file using tree-sitter based analyzer.

        Args:
            file_path: Relative path to the PHP file
            content: File content string
            repo_dir: Repository base directory
        """
        from codewiki.src.be.dependency_analyzer.analyzers.php import analyze_php_file

        try:
            functions, relationships = analyze_php_file(file_path, content, repo_path=repo_dir)

            for func in functions:
                func_id = func.id if func.id else f"{file_path}:{func.name}"
                self.functions[func_id] = func

            self.call_relationships.extend(relationships)
        except Exception as e:
            logger.error(f"Failed to analyze PHP file {file_path}: {e}", exc_info=True)

    def _resolve_call_relationships(self):
        """
        Resolve function call relationships across all languages.

        Attempts to match function calls to actual function definitions,
        handling cross-language calls where possible.
        """
        func_lookup = {}
        for func_id, func_info in self.functions.items():
            func_lookup[func_id] = func_id
            func_lookup[func_info.name] = func_id
            if func_info.component_id:
                func_lookup[func_info.component_id] = func_id
                method_name = func_info.component_id.split(".")[-1]
                if method_name not in func_lookup:
                    func_lookup[method_name] = func_id

        resolved_count = 0
        for relationship in self.call_relationships:
            callee_name = relationship.callee

            if callee_name in func_lookup:
                relationship.callee = func_lookup[callee_name]
                relationship.is_resolved = True
                resolved_count += 1
            elif "." in callee_name:
                if callee_name in func_lookup:
                    relationship.callee = func_lookup[callee_name]
                    relationship.is_resolved = True
                    resolved_count += 1
                else:
                    method_name = callee_name.split(".")[-1]
                    if method_name in func_lookup:
                        relationship.callee = func_lookup[method_name]
                        relationship.is_resolved = True
                        resolved_count += 1

    def _deduplicate_relationships(self):
        """
        Deduplicate call relationships based on caller-callee pairs.

        Removes duplicate relationships while preserving the first occurrence.
        This helps eliminate noise from multiple calls to the same function.
        """
        seen = set()
        unique_relationships = []

        for rel in self.call_relationships:
            key = (rel.caller, rel.callee)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        self.call_relationships = unique_relationships

    def _generate_visualization_data(self) -> Dict:
        """
        Generate visualization data for graph rendering.

        Creates Cytoscape.js compatible graph data with nodes and edges.

        Returns:
            Dict: Visualization data with cytoscape elements and summary
        """
        cytoscape_elements = []

        for func_id, func_info in self.functions.items():
            node_classes = []
            if func_info.node_type == "method":
                node_classes.append("node-method")
            else:
                node_classes.append("node-function")

            file_ext = Path(func_info.file_path).suffix.lower()
            if file_ext == ".py":
                node_classes.append("lang-python")
            elif file_ext == ".js":
                node_classes.append("lang-javascript")
            elif file_ext == ".ts":
                node_classes.append("lang-typescript")
            elif file_ext in [".c", ".h"]:
                node_classes.append("lang-c")
            elif file_ext in [".cpp", ".cc", ".cxx", ".hpp", ".hxx"]:
                node_classes.append("lang-cpp")
            elif file_ext in [".php", ".phtml", ".inc"]:
                node_classes.append("lang-php")

            cytoscape_elements.append(
                {
                    "data": {
                        "id": func_id,
                        "label": func_info.name,
                        "file": func_info.file_path,
                        "type": func_info.node_type or "function",
                        "language": CODE_EXTENSIONS.get(file_ext, "unknown"),
                    },
                    "classes": " ".join(node_classes),
                }
            )

        resolved_rels = [r for r in self.call_relationships if r.is_resolved]
        for rel in resolved_rels:
            cytoscape_elements.append(
                {
                    "data": {
                        "id": f"{rel.caller}->{rel.callee}",
                        "source": rel.caller,
                        "target": rel.callee,
                        "line": rel.call_line,
                    },
                    "classes": "edge-call",
                }
            )

        summary = {
            "total_nodes": len(self.functions),
            "total_edges": len(resolved_rels),
            "unresolved_calls": len(self.call_relationships) - len(resolved_rels),
        }

        return {
            "cytoscape": {"elements": cytoscape_elements},
            "summary": summary,
        }

    def generate_llm_format(self) -> Dict:
        """Generate clean format optimized for LLM consumption."""
        return {
            "functions": [
                {
                    "name": func.name,
                    "file": Path(func.file_path).name,
                    "purpose": (func.docstring.split("\n")[0] if func.docstring else None),
                    "parameters": func.parameters,
                    "is_recursive": func.name
                    in [
                        rel.callee
                        for rel in self.call_relationships
                        if rel.caller.endswith(func.name)
                    ],
                }
                for func in self.functions.values()
            ],
            "relationships": {
                func.name: {
                    "calls": [
                        rel.callee.split(":")[-1]
                        for rel in self.call_relationships
                        if rel.caller.endswith(func.name) and rel.is_resolved
                    ],
                    "called_by": [
                        rel.caller.split(":")[-1]
                        for rel in self.call_relationships
                        if rel.callee.endswith(func.name) and rel.is_resolved
                    ],
                }
                for func in self.functions.values()
            },
        }

    def _select_most_connected_nodes(self, target_count: int):
        """
        Select the most connected nodes from the call graph.

        Args:
            target_count: The number of nodes to select
        """
        if len(self.functions) <= target_count:
            return

        if not self.call_relationships:
            logger.warning("No call relationships found - keeping all functions by name")
            func_ids = list(self.functions.keys())[:target_count]
            self.functions = {fid: func for fid, func in self.functions.items() if fid in func_ids}
            return

        graph = {}
        for rel in self.call_relationships:
            if rel.caller in self.functions:
                if rel.caller not in graph:
                    graph[rel.caller] = set()
            if rel.callee in self.functions:
                if rel.callee not in graph:
                    graph[rel.callee] = set()

            if rel.caller in graph and rel.callee in graph:
                graph[rel.caller].add(rel.callee)
                graph[rel.callee].add(rel.caller)

        degree_centrality = {}
        for func_id in self.functions.keys():
            degree_centrality[func_id] = len(graph.get(func_id, set()))

        sorted_func_ids = sorted(
            degree_centrality.keys(), key=lambda x: degree_centrality.get(x, 0), reverse=True
        )

        selected_func_ids = sorted_func_ids[:target_count]

        self.functions = {
            fid: func for fid, func in self.functions.items() if fid in selected_func_ids
        }

        self.call_relationships = [
            rel
            for rel in self.call_relationships
            if rel.caller in selected_func_ids and rel.callee in selected_func_ids
        ]
