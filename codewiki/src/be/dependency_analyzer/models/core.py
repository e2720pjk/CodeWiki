from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Set, Union, Literal
from datetime import datetime


class CFGNode(BaseModel):
    """Control Flow Graph node extracted from Joern CPG."""
    id: str
    label: str
    node_type: str  # "method", "block", "statement", etc.
    line: Optional[int] = None
    column: Optional[int] = None


class CFGEdge(BaseModel):
    """Edge in control flow graph."""
    source: str
    target: str
    label: str  # "true", "false", "default", etc.


class ControlFlowData(BaseModel):
    """Control flow analysis data from Joern CPG."""
    nodes: List[CFGNode] = Field(default_factory=list)
    edges: List[CFGEdge] = Field(default_factory=list)
    entry_point: Optional[str] = None
    exit_points: List[str] = Field(default_factory=list)


class DataDependency(BaseModel):
    """Single data dependency in DDG."""
    source: str  # Variable name
    target: str  # Variable name
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    dependency_type: Literal["direct", "indirect", "control"] = "direct"


class DataFlowData(BaseModel):
    """Data flow graph data from Joern CPG."""
    variables: Dict[str, List[int]] = {}  # variable -> line numbers
    dependencies: List[DataDependency] = Field(default_factory=list)
    tainted_flows: List[str] = Field(default_factory=list)


class JoernMetadata(BaseModel):
    """Language-specific Joern CPG metadata."""
    joern_version: str
    cpg_format_version: str
    language: str  # "python", "java", "javascript", etc.
    analysis_timestamp: str  # ISO 8601
    cpg_location: Optional[str] = None  # Path to CPG binary if saved


class Node(BaseModel):
    id: str
    name: str
    component_type: str
    file_path: str
    relative_path: str
    depends_on: List[str] = Field(default_factory=list)
    source_code: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    has_docstring: bool = False
    docstring: str = ""
    parameters: Optional[List[str]] = None
    node_type: Optional[str] = None
    base_classes: Optional[List[str]] = None
    class_name: Optional[str] = None
    display_name: Optional[str] = None
    component_id: Optional[str] = None

    def get_display_name(self) -> str:
        return self.display_name or self.name


class EnhancedNode(Node):
    """Extended Node with Joern CPG data."""
    cfg_data: Optional[ControlFlowData] = None
    ddg_data: Optional[DataFlowData] = None
    joern_metadata: Optional[JoernMetadata] = None

    @field_validator('cfg_data', 'ddg_data', mode='before')
    @classmethod
    def validate_graph_data(cls, v: Any) -> Any:
        """Allow dict or model instance."""
        if v is None:
            return None
        if isinstance(v, dict):
            try:
                if 'nodes' in v:  # Likely CFG
                    return ControlFlowData(**v)
                elif 'dependencies' in v:  # Likely DDG
                    return DataFlowData(**v)
            except Exception:
                return v  # Return as-is if validation fails
        return v


class CallRelationship(BaseModel):
    caller: str
    callee: str
    call_line: Optional[int] = None
    is_resolved: bool = False


class DataFlowRelationship(BaseModel):
    """Data flow relationship extracted from Joern CPG analysis."""
    source: str
    target: str
    flow_type: str  # "parameter", "local", "return", "field", "tainted"
    variable_name: Optional[str] = None
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    confidence: float = 1.0  # Confidence score 0.0-1.0
    file_path: Optional[str] = None
    source_var: Optional[str] = None
    target_var: Optional[str] = None


class InheritanceRelationship(BaseModel):
    """Inheritance/Implementation relationship."""
    source: str  # Child class ID
    target: str  # Parent class ID
    relationship_type: Literal["extends", "implements", "mixin"]


class Repository(BaseModel):
    url: str
    name: str
    clone_path: str
    analysis_id: str
