"""
Simplified Joern Integration for Quick Win PoC

This provides a minimal Joern integration that focuses on the core functionality
without complex dependencies.
"""

import logging
import json
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class SimplifiedJoernAnalyzer:
    """
    Simplified Joern analyzer focusing on basic CPG extraction.
    """

    def __init__(self):
        self.joern_jar = None
        self.temp_dir = None

    def setup_joern(self) -> bool:
        """
        Set up Joern environment.

        Returns:
            True if setup successful, False otherwise
        """
        # Look for joern.jar in current directory and common paths
        possible_paths = [
            "joern.jar",
            "./joern.jar",
            os.path.expanduser("~/.joern/joern.jar"),
            "/opt/joern/joern.jar",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                self.joern_jar = path
                logger.info(f"Found Joern JAR at: {path}")
                return True

        logger.warning("Joern JAR not found. Please download joern.jar")
        return False

    def analyze_repository_basic(self, repo_path: str) -> Dict[str, Any]:
        """
        Perform basic repository analysis using Joern.

        Args:
            repo_path: Path to repository to analyze

        Returns:
            Basic analysis results
        """
        if not self.setup_joern():
            raise RuntimeError("Joern not available")

        start_time = time.time()
        self.temp_dir = tempfile.mkdtemp(prefix="joern_basic_")

        try:
            logger.info(f"Starting basic Joern analysis of {repo_path}")

            # Create a simple analysis script
            script_content = self._create_basic_script()
            script_file = os.path.join(self.temp_dir, "basic.sc")

            with open(script_file, "w") as f:
                f.write(script_content)

            # Run Joern
            cmd = ["java", "-jar", self.joern_jar, "--script", script_file, repo_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes timeout
                cwd=self.temp_dir,
            )

            if result.returncode != 0:
                logger.error(f"Joern analysis failed: {result.stderr}")
                # Try to return partial results from AST fallback
                return self._ast_fallback(repo_path)

            analysis_time = time.time() - start_time
            logger.info(f"Basic Joern analysis completed in {analysis_time:.2f} seconds")

            return self._parse_basic_results(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("Joern analysis timed out")
            return self._ast_fallback(repo_path)
        except Exception as e:
            logger.error(f"Joern analysis failed: {str(e)}")
            return self._ast_fallback(repo_path)
        finally:
            self._cleanup()

    def _create_basic_script(self) -> str:
        """Create a basic Joern analysis script."""
        return """
import joern._

// Load CPG
val cpg = loadCPG(args(0))

// Get basic statistics
val fileCount = cpg.namespaceBlock.size
val methodCount = cpg.method.size
val classCount = cpg.typeDecl.size

// Extract methods with basic info
val methods = cpg.method.take(100).map { method =>
    Map(
        "name" -> method.name,
        "file" -> method.filename.headOption.getOrElse(""),
        "line" -> method.lineNumber.headOption.getOrElse(-1),
        "signature" -> method.signature
    )
}.l

// Extract classes
val classes = cpg.typeDecl.take(50).map { cls =>
    Map(
        "name" -> cls.name,
        "file" -> cls.filename.headOption.getOrElse(""),
        "line" -> cls.lineNumber.headOption.getOrElse(-1)
    )
}.l

// Create result
val result = Map(
    "total_files" -> fileCount,
    "total_methods" -> methodCount,
    "total_classes" -> classCount,
    "methods" -> methods,
    "classes" -> classes
)

println(result.toJson.compactPrint)
"""

    def _parse_basic_results(self, output: str) -> Dict[str, Any]:
        """Parse basic Joern output."""
        try:
            if output.strip().startswith("{"):
                return json.loads(output)
        except json.JSONDecodeError:
            pass

        # Fallback parsing
        return {
            "status": "partial",
            "raw_output": output,
            "message": "Could not parse Joern output as JSON",
        }

    def _ast_fallback(self, repo_path: str) -> Dict[str, Any]:
        """Fallback to simple AST-style analysis when Joern fails."""
        logger.info("Using AST fallback analysis")

        try:
            # Simple file counting and basic extraction
            py_files = list(Path(repo_path).rglob("*.py"))
            js_files = list(Path(repo_path).rglob("*.js"))
            ts_files = list(Path(repo_path).rglob("*.ts"))
            java_files = list(Path(repo_path).rglob("*.java"))

            total_files = len(py_files) + len(js_files) + len(ts_files) + len(java_files)

            # Extract some basic function info from Python files
            functions = []
            for py_file in py_files[:10]:  # Limit to 10 files for speed
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Simple regex for function definitions
                        import re

                        for match in re.finditer(r"def\s+(\w+)\s*\(", content):
                            line_num = content[: match.start()].count("\\n") + 1
                            functions.append(
                                {
                                    "name": match.group(1),
                                    "file": str(py_file),
                                    "line": line_num,
                                    "type": "function",
                                }
                            )
                except Exception as e:
                    logger.warning(f"Could not read {py_file}: {e}")

            return {
                "status": "ast_fallback",
                "total_files": total_files,
                "total_methods": len(functions),
                "total_classes": 0,
                "methods": functions[:50],  # Limit to 50 methods
                "classes": [],
                "message": "Used AST fallback due to Joern unavailability",
            }

        except Exception as e:
            logger.error(f"AST fallback failed: {e}")
            return {"status": "error", "message": f"Both Joern and AST fallback failed: {str(e)}"}

    def extract_data_flow_sample(self, repo_path: str, function_name: str) -> Dict[str, Any]:
        """Extract sample data flow information."""
        if not self.setup_joern():
            return {"status": "error", "message": "Joern not available"}

        script_content = f"""
import joern._

val cpg = loadCPG(args(0))
val method = cpg.method.name("{function_name}").headOption

if (method.isDefined) {{
    val m = method.get
    val params = m.parameter.name.l
    val locals = m.local.name.l
    
    val result = Map(
        "function" -> "{function_name}",
        "parameters" -> params,
        "local_variables" -> locals,
        "param_count" -> params.size,
        "local_count" -> locals.size
    )
    
    println(result.toJson.compactPrint)
}} else {{
    println(Map("error" -> s"Function {function_name} not found").toJson.compactPrint)
}}
"""

        self.temp_dir = tempfile.mkdtemp(prefix="joern_df_")
        script_file = os.path.join(self.temp_dir, "dataflow.sc")

        try:
            with open(script_file, "w") as f:
                f.write(script_content)

            cmd = ["java", "-jar", self.joern_jar, "--script", script_file, repo_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return json.loads(result.stdout.strip())
            else:
                return {"status": "error", "message": result.stderr}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil

                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


def run_performance_baseline(repo_path: str) -> Dict[str, Any]:
    """
    Run performance baseline for AST vs Joern.

    Args:
        repo_path: Path to repository for baseline testing

    Returns:
        Performance metrics
    """
    analyzer = SimplifiedJoernAnalyzer()

    # Test AST fallback timing
    start_time = time.time()
    ast_result = analyzer._ast_fallback(repo_path)
    ast_time = time.time() - start_time

    # Test Joern if available
    joern_time = None
    joern_result = None

    if analyzer.setup_joern():
        try:
            start_time = time.time()
            joern_result = analyzer.analyze_repository_basic(repo_path)
            joern_time = time.time() - start_time
        except Exception as e:
            logger.warning(f"Joern baseline test failed: {e}")

    return {
        "ast_time": ast_time,
        "joern_time": joern_time,
        "ast_methods": ast_result.get("total_methods", 0),
        "joern_methods": joern_result.get("total_methods", 0) if joern_result else 0,
        "speedup": joern_time / ast_time if joern_time and ast_time > 0 else None,
        "status": "success",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with current directory as sample
    current_dir = os.getcwd()
    baseline = run_performance_baseline(current_dir)
    print("Performance Baseline Results:")
    print(json.dumps(baseline, indent=2))
