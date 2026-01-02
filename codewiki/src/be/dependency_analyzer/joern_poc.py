"""
Joern PoC - Direct Joern CLI Integration

This module provides direct integration with Joern CLI without pygraphviz dependency.
"""

import subprocess
import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import time

logger = logging.getLogger(__name__)


class JoernAnalyzer:
    """
    Direct Joern CLI integration for CPG analysis.

    This class uses subprocess to call Joern CLI directly,
    avoiding the pygraphviz dependency issues.
    """

    def __init__(self, joern_path: Optional[str] = None):
        """
        Initialize Joern analyzer.

        Args:
            joern_path: Path to Joern binary. If None, will try to find in PATH.
        """
        self.joern_path = joern_path or self._find_joern_binary()
        self.temp_dir = tempfile.mkdtemp(prefix="joern_analysis_")

    def _find_joern_binary(self) -> str:
        """Find Joern binary in PATH or default locations."""
        possible_paths = [
            "joern",
            "/usr/local/bin/joern",
            os.path.expanduser("~/.joern/joern"),
            "/opt/joern/joern",
        ]

        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"Found Joern at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        raise RuntimeError("Joern binary not found. Please install Joern manually.")

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Analyze a project using Joern CPG.

        Args:
            project_path: Path to the project to analyze

        Returns:
            Dict containing CPG analysis results
        """
        start_time = time.time()

        try:
            logger.info(f"Starting Joern analysis of {project_path}")

            # Create Joern script
            script = self._create_analysis_script()
            script_file = os.path.join(self.temp_dir, "analysis.sc")

            with open(script_file, "w") as f:
                f.write(script)

            # Run Joern
            cmd = [self.joern_path, "--script", script_file, project_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=self.temp_dir,
            )

            if result.returncode != 0:
                logger.error(f"Joern analysis failed: {result.stderr}")
                raise RuntimeError(f"Joern analysis failed: {result.stderr}")

            # Parse results
            analysis_time = time.time() - start_time
            logger.info(f"Joern analysis completed in {analysis_time:.2f} seconds")

            return self._parse_joern_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("Joern analysis timed out")
            raise RuntimeError("Joern analysis timed out after 5 minutes")
        except Exception as e:
            logger.error(f"Joern analysis failed: {str(e)}")
            raise

    def _create_analysis_script(self) -> str:
        """
        Create Joern script for CPG analysis.

        Returns:
            Joern script string
        """
        return """
import joern._
import java.io.{File, PrintWriter}

// Load CPG (will be loaded when running with project path)
val cpg = loadCPG(args(0))

// Extract functions
val functions = cpg.method.map { method =>
    Map(
        "id" -> method.id,
        "name" -> method.name,
        "signature" -> method.signature,
        "file" -> method.filename.headOption.getOrElse(""),
        "line_number" -> method.lineNumber.headOption.getOrElse(-1),
        "code" -> method.code.headOption.getOrElse(""),
        "is_external" -> method.isExternal,
        "num_parameters" -> method.parameter.size,
        "num_cfg_nodes" -> method.cfgNode.size,
        "num_ddg_edges" -> method.ddg.size
    )
}.l

// Extract call relationships
val calls = cpg.call.map { call =>
    Map(
        "caller_file" -> call.method.filename.headOption.getOrElse(""),
        "caller_name" -> call.method.name.headOption.getOrElse(""),
        "callee_name" -> call.methodFullName.headOption.getOrElse(""),
        "line_number" -> call.lineNumber.headOption.getOrElse(-1),
        "is_resolved" -> call.methodFullName.isDefined
    )
}.l

// Extract data flow information
val dataflows = cpg.identifier.map { ident =>
    Map(
        "name" -> ident.name,
        "type" -> ident.typeFullName.headOption.getOrElse(""),
        "file" -> ident.filename.headOption.getOrElse(""),
        "line_number" -> ident.lineNumber.headOption.getOrElse(-1),
        "is_parameter" -> ident.isParameter,
        "is_local" -> ident.isLocal
    )
}.l

// Export results
val result = Map(
    "functions" -> functions,
    "calls" -> calls,
    "dataflows" -> dataflows,
    "total_functions" -> functions.size,
    "total_calls" -> calls.size,
    "total_dataflows" -> dataflows.size
)

// Save to JSON
import spray.json._
import DefaultJsonProtocol._

implicit val mapFormat = jsonFormat2(Map.apply[String, Any])

val json = result.toJson.compactPrint
val writer = new PrintWriter("joern_results.json")
writer.write(json)
writer.close()

println(json)
"""

    def _parse_joern_output(self, output: str) -> Dict[str, Any]:
        """
        Parse Joern output.

        Args:
            output: Raw output from Joern

        Returns:
            Parsed results dictionary
        """
        try:
            # Try to parse as JSON
            if output.strip().startswith("{"):
                return json.loads(output)
            else:
                # Parse line by line
                lines = output.strip().split("\n")
                results = {}

                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        results[key.strip()] = value.strip()

                return results
        except json.JSONDecodeError:
            logger.warning("Could not parse Joern output as JSON")
            return {"raw_output": output}

    def extract_data_flow(self, project_path: str, function_name: str) -> Dict[str, Any]:
        """
        Extract data flow information for a specific function.

        Args:
            project_path: Path to the project
            function_name: Name of the function to analyze

        Returns:
            Data flow analysis results
        """
        script = (
            f"""
import joern._

val cpg = loadCPG("""
            + project_path
            + """)
val method = cpg.method.name("""
            + function_name
            + """).head

if (method != null) {{
    // Extract CFG
    val cfg_nodes = method.cfgNode.map { node =>
        Map(
            "id" -> node.id,
            "code" -> node.code.headOption.getOrElse(""),
            "line_number" -> node.lineNumber.headOption.getOrElse(-1)
        )
    }.l
    
    // Extract DDG (Data Dependencies)
    val ddg_edges = method.ddg.map { edge =>
        Map(
            "source" -> edge.inNode.head.id,
            "target" -> edge.outNode.head.id,
            "var" -> edge.variable.headOption.getOrElse("")
        )
    }.l
    
    // Extract PDG (Program Dependencies)
    val pdg_edges = method.pdg.map { edge =>
        Map(
            "source" -> edge.inNode.head.id,
            "target" -> edge.outNode.head.id,
            "edge_type" -> edge.edgeType.headOption.getOrElse("")
        )
    }.l
    
    val result = Map(
        "function_name" -> """
            " + function_name + "
            """,
        "cfg_nodes" -> cfg_nodes,
        "ddg_edges" -> ddg_edges,
        "pdg_edges" -> pdg_edges,
        "total_cfg_nodes" -> cfg_nodes.size,
        "total_ddg_edges" -> ddg_edges.size,
        "total_pdg_edges" -> pdg_edges.size
    )
    
    println(result.toJson.compactPrint)
}} else {{
    println(Map("error" -> s"Function """
            + function_name
            + """ not found").toJson.compactPrint)
}}
"""
        )

        script_file = os.path.join(self.temp_dir, "dataflow.sc")
        with open(script_file, "w") as f:
            f.write(script)

        cmd = [self.joern_path, "--script", script_file, project_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(f"Data flow analysis failed: {result.stderr}")

        return json.loads(result.stdout.strip())

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary directory: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def test_joern_analyzer():
    """Test function to verify Joern analyzer works."""
    try:
        analyzer = JoernAnalyzer()
        logger.info("✓ JoernAnalyzer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ JoernAnalyzer initialization failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_joern_analyzer()
