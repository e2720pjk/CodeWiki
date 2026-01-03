import logging
from typing import Optional, Union, Any
from enum import Enum
from codewiki.src.be.dependency_analyzer.analysis.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

class AnalyzerType(Enum):
    JOERN = "joern"
    HYBRID = "hybrid"
    AST = "ast"

class AnalyzerFactory:
    """Factory for creating appropriate analyzer with fallback."""
    
    @staticmethod
    def create_analyzer(analyzer_type: AnalyzerType = AnalyzerType.HYBRID, config: Any = None):
        """
        Create analyzer with automatic fallback.
        
        Args:
            analyzer_type: Preferred analyzer type
            config: Configuration object
        
        Returns:
            Working analyzer instance
        """
        if analyzer_type == AnalyzerType.HYBRID:
            try:
                from codewiki.src.be.dependency_analyzer.hybrid_analysis_service import HybridAnalysisService
                return HybridAnalysisService()
            except Exception as e:
                logger.warning(f"Hybrid analysis unavailable, falling back to AST: {e}")
                return AnalysisService()
        
        if analyzer_type == AnalyzerType.JOERN:
            # For now, Joern is used via Hybrid or Client. 
            # If someone explicitly wants Joern-only, we might return a Joern-specific wrapper.
            try:
                from codewiki.src.be.dependency_analyzer.joern.client import JoernClient
                client = JoernClient()
                if client.is_available:
                     return client
                else:
                    raise RuntimeError("Joern not available")
            except Exception as e:
                logger.warning(f"Joern unavailable, falling back to AST: {e}")
                return AnalysisService()

        # Default to AST
        return AnalysisService()
