import logging
import os
import json
import asyncio
from typing import Dict, List, Any, Tuple
from copy import deepcopy
import traceback
from tqdm.asyncio import tqdm

# Configure logging and monitoring
logger = logging.getLogger(__name__)

# Local imports
from codewiki.src.be.dependency_analyzer import DependencyGraphBuilder
from codewiki.src.be.llm_services import call_llm, call_llm_async
from codewiki.src.be.prompt_template import (
    REPO_OVERVIEW_PROMPT,
    MODULE_OVERVIEW_PROMPT,
)
from codewiki.src.be.cluster_modules import cluster_modules
from codewiki.src.config import (
    Config,
    FIRST_MODULE_TREE_FILENAME,
    MODULE_TREE_FILENAME,
    OVERVIEW_FILENAME
)
from codewiki.src.utils import file_manager
from codewiki.src.be.agent_orchestrator import AgentOrchestrator
from codewiki.src.be.performance_metrics import performance_tracker


class DocumentationGenerator:
    """Main documentation generation orchestrator."""
    
    def __init__(self, config: Config, commit_id: str = None):
        self.config = config
        self.commit_id = commit_id
        self.graph_builder = DependencyGraphBuilder(config)
        self.agent_orchestrator = AgentOrchestrator(config)
    
    def create_documentation_metadata(self, working_dir: str, components: Dict[str, Any], num_leaf_nodes: int):
        """Create a metadata file with documentation generation information."""
        from datetime import datetime
        
        metadata = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "main_model": self.config.main_model,
                "generator_version": "1.0.0",
                "repo_path": self.config.repo_path,
                "commit_id": self.commit_id
            },
            "statistics": {
                "total_components": len(components),
                "leaf_nodes": num_leaf_nodes,
                "max_depth": self.config.max_depth
            },
            "files_generated": [
                "overview.md",
                "module_tree.json",
                "first_module_tree.json"
            ]
        }
        
        # Add generated markdown files to the metadata
        try:
            for file_path in os.listdir(working_dir):
                if file_path.endswith('.md') and file_path not in metadata["files_generated"]:
                    metadata["files_generated"].append(file_path)
        except Exception as e:
            logger.warning(f"Could not list generated files: {e}")
        
        metadata_path = os.path.join(working_dir, "metadata.json")
        file_manager.save_json(metadata, metadata_path)

    
    def get_processing_order(self, module_tree: Dict[str, Any], parent_path: List[str] = []) -> List[tuple[List[str], str]]:
        """Get the processing order using topological sort (leaf modules first)."""
        processing_order = []
        
        def collect_modules(tree: Dict[str, Any], path: List[str]):
            for module_name, module_info in tree.items():
                current_path = path + [module_name]
                
                # If this module has children, process them first
                if module_info.get("children") and isinstance(module_info["children"], dict) and module_info["children"]:
                    collect_modules(module_info["children"], current_path)
                    # Add this parent module after its children
                    processing_order.append((current_path, module_name))
                else:
                    # This is a leaf module, add it immediately
                    processing_order.append((current_path, module_name))
        
        collect_modules(module_tree, parent_path)
        return processing_order

    def is_leaf_module(self, module_info: Dict[str, Any]) -> bool:
        """Check if a module is a leaf module (has no children or empty children)."""
        children = module_info.get("children", {})
        return not children or (isinstance(children, dict) and len(children) == 0)

    def build_overview_structure(self, module_tree: Dict[str, Any], module_path: List[str],
                                 working_dir: str) -> Dict[str, Any]:
        """Build structure for overview generation with 1-depth children docs and target indicator."""
        
        processed_module_tree = deepcopy(module_tree)
        module_info = processed_module_tree
        for path_part in module_path:
            module_info = module_info[path_part]
            if path_part != module_path[-1]:
                module_info = module_info.get("children", {})
            else:
                module_info["is_target_for_overview_generation"] = True

        if "children" in module_info:
            module_info = module_info["children"]

        for child_name, child_info in module_info.items():
            if os.path.exists(os.path.join(working_dir, f"{child_name}.md")):
                child_info["docs"] = file_manager.load_text(os.path.join(working_dir, f"{child_name}.md"))
            else:
                logger.warning(f"Module docs not found at {os.path.join(working_dir, f"{child_name}.md")}")
                child_info["docs"] = ""

        return processed_module_tree

    async def process_leaf_modules_parallel(
        self, 
        leaf_modules: List[Tuple[List[str], str]], 
        components: Dict[str, Any], 
        working_dir: str
    ) -> Dict[str, Any]:
        """
        Process leaf modules in parallel using semaphore-controlled concurrency.
        
        Args:
            leaf_modules: List of (module_path, module_name) tuples for leaf modules
            components: Components dictionary
            working_dir: Working directory for output
            
        Returns:
            Updated module tree
        """
        if not self.config.enable_parallel_processing or len(leaf_modules) <= 1:
            # Fall back to sequential processing
            return await self.process_leaf_modules_sequential(leaf_modules, components, working_dir)
        
        logger.info(f"Processing {len(leaf_modules)} leaf modules in parallel with concurrency limit {self.config.concurrency_limit}")
        
        semaphore = asyncio.Semaphore(self.config.concurrency_limit)
        
        async def process_with_semaphore(module_path: List[str], module_name: str) -> Tuple[str, bool]:
            """Process a single leaf module with semaphore control."""
            async with semaphore:
                module_key = "/".join(module_path)
                try:
                    logger.info(f"📄 Processing leaf module: {module_key}")
                    
                    # Get module info
                    module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
                    module_tree = file_manager.load_json(module_tree_path)
                    module_info = module_tree
                    for path_part in module_path:
                        module_info = module_info[path_part]
                        if path_part != module_path[-1]:
                            module_info = module_info.get("children", {})
                    
                    # Process the module
                    start_time = asyncio.get_event_loop().time()
                    result = await self.agent_orchestrator.process_module(
                        module_name, components, module_info["components"], module_path, working_dir
                    )
                    processing_time = asyncio.get_event_loop().time() - start_time
                    
                    # Record metrics
                    performance_tracker.record_module_processing(True, processing_time)
                    
                    return module_key, True
                    
                except Exception as e:
                    logger.error(f"Failed to process leaf module {module_key}: {str(e)}")
                    performance_tracker.record_module_processing(False, 0)
                    return module_key, False
        
        # Process all modules in parallel with progress tracking
        with tqdm(total=len(leaf_modules), desc="Processing leaf modules") as pbar:
            async def process_with_progress(module_path: List[str], module_name: str):
                result = await process_with_semaphore(module_path, module_name)
                pbar.update(1)
                return result
            
            results = await asyncio.gather(
                *[process_with_progress(mp, mn) for mp, mn in leaf_modules],
                return_exceptions=True
            )
        
        # Check results and handle failures
        failed_modules = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel processing error: {result}")
                continue
            
            module_key, success = result
            if not success:
                failed_modules.append(module_key)
        
        # Retry failed modules sequentially
        if failed_modules:
            logger.warning(f"Retrying {len(failed_modules)} failed leaf modules sequentially")
            await self.retry_failed_modules_sequential(failed_modules, leaf_modules, components, working_dir)
        
        # Return updated module tree
        module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
        return file_manager.load_json(module_tree_path)
    
    async def process_leaf_modules_sequential(
        self, 
        leaf_modules: List[Tuple[List[str], str]], 
        components: Dict[str, Any], 
        working_dir: str
    ) -> Dict[str, Any]:
        """
        Process leaf modules sequentially (fallback method).
        
        Args:
            leaf_modules: List of (module_path, module_name) tuples for leaf modules
            components: Components dictionary
            working_dir: Working directory for output
            
        Returns:
            Updated module tree
        """
        logger.info(f"Processing {len(leaf_modules)} leaf modules sequentially")
        
        module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
        
        for module_path, module_name in leaf_modules:
            module_key = "/".join(module_path)
            try:
                logger.info(f"📄 Processing leaf module: {module_key}")
                
                # Get module info
                module_tree = file_manager.load_json(module_tree_path)
                module_info = module_tree
                for path_part in module_path:
                    module_info = module_info[path_part]
                    if path_part != module_path[-1]:
                        module_info = module_info.get("children", {})
                
                # Process the module
                start_time = asyncio.get_event_loop().time()
                await self.agent_orchestrator.process_module(
                    module_name, components, module_info["components"], module_path, working_dir
                )
                processing_time = asyncio.get_event_loop().time() - start_time
                
                # Record metrics
                performance_tracker.record_module_processing(True, processing_time)
                
            except Exception as e:
                logger.error(f"Failed to process leaf module {module_key}: {str(e)}")
                performance_tracker.record_module_processing(False, 0)
        
        return file_manager.load_json(module_tree_path)
    
    async def retry_failed_modules_sequential(
        self,
        failed_modules: List[str],
        all_leaf_modules: List[Tuple[List[str], str]],
        components: Dict[str, Any],
        working_dir: str
    ) -> None:
        """
        Retry failed modules sequentially as fallback.
        
        Args:
            failed_modules: List of module keys that failed
            all_leaf_modules: All leaf modules for reference
            components: Components dictionary
            working_dir: Working directory for output
        """
        # Create mapping from module key to (module_path, module_name)
        module_mapping = {"/".join(path): (path, name) for path, name in all_leaf_modules}
        
        for module_key in failed_modules:
            if module_key not in module_mapping:
                continue
                
            module_path, module_name = module_mapping[module_key]
            try:
                logger.info(f"🔄 Retrying leaf module: {module_key}")
                
                # Get module info
                module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
                module_tree = file_manager.load_json(module_tree_path)
                module_info = module_tree
                for path_part in module_path:
                    module_info = module_info[path_part]
                    if path_part != module_path[-1]:
                        module_info = module_info.get("children", {})
                
                # Process the module
                await self.agent_orchestrator.process_module(
                    module_name, components, module_info["components"], module_path, working_dir
                )
                
                logger.info(f"✓ Successfully retried module: {module_key}")
                
            except Exception as e:
                logger.error(f"❌ Retry failed for module {module_key}: {str(e)}")

    async def generate_module_documentation(self, components: Dict[str, Any], leaf_nodes: List[str]) -> str:
        """Generate documentation for all modules using dynamic programming approach."""
        # Prepare output directory
        working_dir = os.path.abspath(self.config.docs_dir)
        file_manager.ensure_directory(working_dir)

        module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
        first_module_tree_path = os.path.join(working_dir, FIRST_MODULE_TREE_FILENAME)
        module_tree = file_manager.load_json(module_tree_path)
        first_module_tree = file_manager.load_json(first_module_tree_path)
        
        # Get processing order (leaf modules first)
        processing_order = self.get_processing_order(first_module_tree)
        
        # Separate leaf and parent modules
        leaf_modules = []
        parent_modules = []
        
        for module_path, module_name in processing_order:
            # Get the module info from the tree
            module_info = module_tree
            for path_part in module_path:
                module_info = module_info[path_part]
                if path_part != module_path[-1]:  # Not the last part
                    module_info = module_info.get("children", {})
            
            if self.is_leaf_module(module_info):
                leaf_modules.append((module_path, module_name))
            else:
                parent_modules.append((module_path, module_name))
        
        # Start performance tracking
        total_modules = len(processing_order)
        leaf_count = len(leaf_modules)
        performance_tracker.start_tracking(total_modules, leaf_count, self.config.concurrency_limit)
        
        # Process modules in dependency order
        final_module_tree = module_tree
        processed_modules = set()

        if len(module_tree) > 0:
            # Process leaf modules in parallel if enabled
            if leaf_modules:
                if self.config.enable_parallel_processing and len(leaf_modules) > 1:
                    final_module_tree = await self.process_leaf_modules_parallel(
                        leaf_modules, components, working_dir
                    )
                else:
                    final_module_tree = await self.process_leaf_modules_sequential(
                        leaf_modules, components, working_dir
                    )
            
            # Process parent modules sequentially (must maintain order for context)
            for module_path, module_name in parent_modules:
                module_key = "/".join(module_path)
                try:
                    logger.info(f"📁 Processing parent module: {module_key}")
                    final_module_tree = await self.generate_parent_module_docs(
                        module_path, working_dir
                    )
                    processed_modules.add(module_key)
                    
                except Exception as e:
                    logger.error(f"Failed to process parent module {module_key}: {str(e)}")
                    continue

            # Generate repo overview
            logger.info(f"📚 Generating repository overview")
            final_module_tree = await self.generate_parent_module_docs(
                [], working_dir
            )
        else:
            logger.info(f"Processing whole repo because repo can fit in the context window")
            repo_name = os.path.basename(os.path.normpath(self.config.repo_path))
            final_module_tree = await self.agent_orchestrator.process_module(
                repo_name, components, leaf_nodes, [], working_dir
            )

            # save final_module_tree to module_tree.json
            file_manager.save_json(final_module_tree, os.path.join(working_dir, MODULE_TREE_FILENAME))

            # rename repo_name.md to overview.md
            repo_overview_path = os.path.join(working_dir, f"{repo_name}.md")
            if os.path.exists(repo_overview_path):
                os.rename(repo_overview_path, os.path.join(working_dir, OVERVIEW_FILENAME))
        
        return working_dir

    async def generate_parent_module_docs(self, module_path: List[str], 
                                        working_dir: str) -> Dict[str, Any]:
        """Generate documentation for a parent module based on its children's documentation."""
        module_name = module_path[-1] if len(module_path) >= 1 else os.path.basename(os.path.normpath(self.config.repo_path))

        logger.info(f"Generating parent documentation for: {module_name}")
        
        # Load module tree
        module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
        module_tree = file_manager.load_json(module_tree_path)

        # check if overview docs already exists
        overview_docs_path = os.path.join(working_dir, OVERVIEW_FILENAME)
        if os.path.exists(overview_docs_path):
            logger.info(f"✓ Overview docs already exists at {overview_docs_path}")
            return module_tree

        # check if parent docs already exists
        parent_docs_path = os.path.join(working_dir, f"{module_name if len(module_path) >= 1 else OVERVIEW_FILENAME.replace('.md', '')}.md")
        if os.path.exists(parent_docs_path):
            logger.info(f"✓ Parent docs already exists at {parent_docs_path}")
            return module_tree

        # Create repo structure with 1-depth children docs and target indicator
        repo_structure = self.build_overview_structure(module_tree, module_path, working_dir)

        prompt = MODULE_OVERVIEW_PROMPT.format(
            module_name=module_name,
            repo_structure=json.dumps(repo_structure, indent=4)
        ) if len(module_path) >= 1 else REPO_OVERVIEW_PROMPT.format(
            repo_name=module_name,
            repo_structure=json.dumps(repo_structure, indent=4)
        )
        
        try:
            parent_docs = call_llm(prompt, self.config)
            
            # Parse and save parent documentation
            parent_content = parent_docs.split("<OVERVIEW>")[1].split("</OVERVIEW>")[0].strip()
            # parent_content = prompt
            file_manager.save_text(parent_content, parent_docs_path)
            
            logger.debug(f"Successfully generated parent documentation for: {module_name}")
            return module_tree
            
        except Exception as e:
            logger.error(f"Error generating parent documentation for {module_name}: {str(e)}")
            raise
    
    async def run(self) -> None:
        """Run the complete documentation generation process using dynamic programming."""
        try:
            # Build dependency graph
            components, leaf_nodes = self.graph_builder.build_dependency_graph()

            logger.debug(f"Found {len(leaf_nodes)} leaf nodes")
            # logger.debug(f"Leaf nodes:\n{'\n'.join(sorted(leaf_nodes)[:200])}")
            # exit()
            
            # Cluster modules
            working_dir = os.path.abspath(self.config.docs_dir)
            file_manager.ensure_directory(working_dir)
            first_module_tree_path = os.path.join(working_dir, FIRST_MODULE_TREE_FILENAME)
            module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
            
            # Check if module tree exists
            if os.path.exists(first_module_tree_path):
                logger.debug(f"Module tree found at {first_module_tree_path}")
                module_tree = file_manager.load_json(first_module_tree_path)
            else:
                logger.debug(f"Module tree not found at {module_tree_path}, clustering modules")
                module_tree = cluster_modules(leaf_nodes, components, self.config)
                file_manager.save_json(module_tree, first_module_tree_path)
            
            file_manager.save_json(module_tree, module_tree_path)
            
            logger.debug(f"Grouped components into {len(module_tree)} modules")
            
            # Generate module documentation using dynamic programming approach
            # This processes leaf modules first, then parent modules
            working_dir = await self.generate_module_documentation(components, leaf_nodes)
            
            # Create documentation metadata
            self.create_documentation_metadata(working_dir, components, len(leaf_nodes))
            
            # Stop performance tracking and calculate metrics
            metrics = performance_tracker.stop_tracking()
            logger.info(f"Performance metrics: {metrics.total_time:.2f}s total, "
                       f"{metrics.successful_modules}/{metrics.total_modules} modules successful")
            
            logger.debug(f"Documentation generation completed successfully using dynamic programming!")
            logger.debug(f"Processing order: leaf modules → parent modules → repository overview")
            logger.debug(f"Documentation saved to: {working_dir}")
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise