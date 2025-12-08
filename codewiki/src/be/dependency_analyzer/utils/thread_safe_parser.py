"""
Thread-safe tree-sitter parser pool for parallel dependency analysis.
"""
import threading
from typing import Dict, Optional
from tree_sitter import Parser, Language
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_c_sharp

import logging

logger = logging.getLogger(__name__)


class ThreadSafeParserPool:
    """
    Thread-safe pool of tree-sitter parsers for parallel processing.
    
    Each thread gets its own parser instance, but Language objects are shared
    since they are thread-safe and expensive to create.
    """
    
    def __init__(self):
        self._language_cache: Dict[str, Language] = {}
        self._parser_cache: Dict[int, Dict[str, Parser]] = {}
        self._lock = threading.Lock()
        
        # Initialize language objects (thread-safe to share)
        self._init_languages()
    
    def _init_languages(self):
        """Initialize language objects (shared across threads)."""
        try:
            # JavaScript
            js_lang_capsule = tree_sitter_javascript.language()
            self._language_cache['javascript'] = Language(js_lang_capsule)
            
            # TypeScript
            ts_lang_capsule = tree_sitter_typescript.language_typescript()
            self._language_cache['typescript'] = Language(ts_lang_capsule)
            
            # Java
            java_lang_capsule = tree_sitter_java.language()
            self._language_cache['java'] = Language(java_lang_capsule)
            
            # C
            c_lang_capsule = tree_sitter_c.language()
            self._language_cache['c'] = Language(c_lang_capsule)
            
            # C++
            cpp_lang_capsule = tree_sitter_cpp.language()
            self._language_cache['cpp'] = Language(cpp_lang_capsule)
            
            # C#
            csharp_lang_capsule = tree_sitter_c_sharp.language()
            self._language_cache['csharp'] = Language(csharp_lang_capsule)
            
            logger.debug(f"Initialized {len(self._language_cache)} language parsers")
            
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter languages: {e}")
            raise
    
    def get_parser(self, language: str) -> Optional[Parser]:
        """
        Get a parser instance for the current thread.
        
        Args:
            language: Language name ('javascript', 'typescript', etc.)
            
        Returns:
            Parser instance for current thread, or None if language not supported
        """
        thread_id = threading.get_ident()
        
        with self._lock:
            # Initialize parser cache for this thread if needed
            if thread_id not in self._parser_cache:
                self._parser_cache[thread_id] = {}
            
            # Create parser for this language if needed
            if language not in self._parser_cache[thread_id]:
                if language not in self._language_cache:
                    logger.warning(f"Unsupported language: {language}")
                    return None
                
                try:
                    language_obj = self._language_cache[language]
                    parser = Parser(language_obj)
                    self._parser_cache[thread_id][language] = parser
                    logger.debug(f"Created {language} parser for thread {thread_id}")
                except Exception as e:
                    logger.error(f"Failed to create {language} parser: {e}")
                    return None
            
            return self._parser_cache[thread_id][language]
    
    def cleanup_thread(self):
        """Clean up parsers for the current thread."""
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id in self._parser_cache:
                del self._parser_cache[thread_id]
                logger.debug(f"Cleaned up parsers for thread {thread_id}")


# Global parser pool instance
parser_pool = ThreadSafeParserPool()


def get_thread_safe_parser(language: str) -> Optional[Parser]:
    """
    Get a thread-safe parser for the specified language.
    
    This is a convenience function that uses the global parser pool.
    
    Args:
        language: Language name
        
    Returns:
        Parser instance or None if not supported
    """
    return parser_pool.get_parser(language)


def cleanup_current_thread():
    """Clean up parsers for the current thread."""
    parser_pool.cleanup_thread()