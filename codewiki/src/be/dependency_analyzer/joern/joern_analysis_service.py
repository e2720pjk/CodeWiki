import logging
from typing import Dict, Any, Optional, List
from codewiki.src.be.dependency_analyzer.joern.client import JoernClient
from codewiki.src.be.dependency_analyzer.models.core import EnhancedNode, DataFlowRelationship

logger = logging.getLogger(__name__)

class JoernAnalysisService:
    """Service for Joern-based code analysis."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.client = JoernClient(timeout_seconds=timeout_seconds)
        self.is_available = self.client.is_available

    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Analyze a repository using Joern.
        
        Returns a dictionary compatible with the expected analysis result format.
        """
        if not self.is_available:
            raise RuntimeError("Joern not available")
            
        raw_result = self.client.generate_cpg(repo_path)
        
        # Convert Joern raw result to internal models
        nodes = {}
        relationships = []
        
        # Process methods as nodes
        for m in raw_result.get("methods", []):
            node_id = m.get("fullName") or m.get("name")
            nodes[node_id] = {
                "id": node_id,
                "name": m.get("name"),
                "file_path": m.get("file"),
                "start_line": m.get("line"),
                "component_type": "function"
            }
            
        # Process calls as relationships
        for c in raw_result.get("calls", []):
            caller = c.get("caller")
            callee = c.get("callee")
            if caller and callee:
                relationships.append({
                    "caller": caller,
                    "callee": callee,
                    "call_line": c.get("line"),
                    "type": "call"
                })
                
        return {
            "nodes": nodes,
            "relationships": relationships,
            "summary": {
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "total_files": raw_result.get("total_files", 0)
            },
            "joern_metadata": {
                "enhanced": True,
                "engine": "joern"
            }
        }
