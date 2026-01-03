import subprocess
import json
import logging
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import tempfile
import pickle
import time

logger = logging.getLogger(__name__)

class JoernVersion:
    """Version specification for Joern."""
    MINIMUM_VERSION = (2, 0, 0)
    MAXIMUM_VERSION = (2, 999, 999)
    TESTED_VERSION = "2.1.0"  # Known working version

class JoernClient:
    """Wrapper around Joern CLI with caching support."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout_seconds: int = 300,
        joern_path: str = "joern"
    ):
        self.cache_dir = cache_dir or (Path.home() / ".codewiki" / "joern_cache")
        self.timeout_seconds = timeout_seconds
        self.joern_path = joern_path
        self.joern_jar = self._find_joern_jar()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate Joern is available
        self.is_available = self._validate_joern_installed()
    
    def _find_joern_jar(self) -> Optional[str]:
        """Look for joern.jar in common paths."""
        search_paths = [
            # 1. Direct path from config if relative
            Path(self.joern_path if self.joern_path.endswith(".jar") else "joern.jar"),
            # 2. Project root (CWD)
            Path.cwd() / "joern.jar",
            # 3. User home
            Path.home() / ".joern" / "joern.jar",
            Path.home() / "bin" / "joern" / "joern.jar",
            # 4. Global paths
            Path("/opt/joern/joern.jar"),
            Path("/usr/local/bin/joern.jar"),
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Detected Joern JAR at: {path}")
                return str(path.absolute())
        return None

    def _validate_joern_installed(self) -> bool:
        """Validate Joern is installed and return status."""
        import shutil
        
        # 1. Resolve binary path
        resolved_path = shutil.which(self.joern_path)
        if not resolved_path:
            # Try common Homebrew paths if not in PATH
            brew_paths = ["/opt/homebrew/bin/joern", "/usr/local/bin/joern"]
            for bp in brew_paths:
                if os.path.exists(bp):
                    resolved_path = bp
                    break
        
        if resolved_path:
            self.joern_path = resolved_path
            # Check if it actually runs. Use --help to avoid interactive shell.
            try:
                # Some Joern versions enter interactive mode with --version
                # --help is usually safe and returns 0 or 1 depending on version
                result = subprocess.run(
                    [self.joern_path, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                # We consider it found if it exists, but returncode 0 is a strong signal
                if result.returncode in [0, 1]:
                    logger.info(f"Joern binary validated at: {self.joern_path}")
                    return True
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                logger.debug(f"Joern binary validation failed: {e}. Checking path existence.")
                if os.path.exists(self.joern_path):
                    logger.info(f"Joern binary found at {self.joern_path}")
                    return True
        
        # Binary validation failed, try to return True if it just exists and is executable
        if resolved_path and os.path.exists(resolved_path):
             logger.info(f"Joern binary found at {resolved_path} (validation skipped)")
             return True

        # 2. Check JAR fallback
        if self.joern_jar:
            try:
                # JAR usually handles --version correctly without entering REPL
                result = subprocess.run(
                    ["java", "-jar", self.joern_jar, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"Joern JAR validated at {self.joern_jar}")
                    return True
            except (FileNotFoundError, subprocess.SubprocessError):
                if os.path.exists(self.joern_jar):
                    logger.info(f"Joern JAR found at {self.joern_jar} (validation skipped)")
                    return True

        logger.warning("Joern not found. Please install Joern or place joern.jar in the project root.")
        return False
    
    def _get_repo_cache_key(self, repo_path: str) -> str:
        """Generate cache key for repository."""
        repo_abs = Path(repo_path).resolve()
        content = f"{repo_abs.as_posix()}".encode()
        return hashlib.sha256(content).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path, repo_path: str) -> bool:
        """Check if cached CPG is still valid."""
        if not cache_file.exists():
            return False
        
        try:
            cache_mtime = cache_file.stat().st_mtime
            repo_mtime = self._get_repo_latest_change_time(repo_path)
            
            # Cache valid if it's newer than latest repo change
            return cache_mtime > repo_mtime
            
        except OSError:
            logger.warning(f"Could not validate cache for {repo_path}")
            return False
    
    @staticmethod
    def _get_repo_latest_change_time(repo_path: str) -> float:
        """Get timestamp of latest file change in repo."""
        repo_path = Path(repo_path)
        source_dirs = ['src', 'lib', 'app', 'main', '.']
        max_mtime = 0
        
        for source_dir in source_dirs:
            target = repo_path / source_dir
            if target.exists():
                for file_path in target.rglob('*'):
                    if file_path.is_file():
                        try:
                            mtime = file_path.stat().st_mtime
                            max_mtime = max(max_mtime, mtime)
                        except OSError:
                            pass
        return max_mtime
    
    def generate_cpg(
        self,
        repo_path: str,
        force: bool = False,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate or retrieve cached CPG."""
        if not self.is_available:
            raise RuntimeError("Joern not available")

        cache_key = self._get_repo_cache_key(repo_path)
        cache_file = self.cache_dir / f"{cache_key}.cpg.pkl"
        
        if not force and self._is_cache_valid(cache_file, repo_path):
            logger.info(f"Using cached CPG for {repo_path}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, regenerating")
        
        logger.info(f"Generating CPG for {repo_path} with Joern")
        try:
            # We use a script strategy similar to simplified_joern for initial implementation
            # as direct CPG export format might vary between Joern versions.
            cpg = self._run_joern_analysis(repo_path, language)
            
            # Save to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cpg, f)
                logger.debug(f"Cached CPG at {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache CPG: {e}")
            
            return cpg
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Joern CPG generation exceeded {self.timeout_seconds}s")

    def _run_joern_analysis(self, repo_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Execute Joern analysis script."""
        with tempfile.TemporaryDirectory(prefix="joern_client_") as temp_dir:
            repo_abs = str(Path(repo_path).resolve())
            script_content = self._get_default_analysis_script(repo_abs)
            script_file = Path(temp_dir) / "analyze.sc"
            script_file.write_text(script_content)

            # [CCR] Reason: Fix InaccessibleObjectException on Java 16+ (e.g. Java 25)
            # Joern/Scala's json4s library uses reflection on internal JDK classes.
            # IMPORTANT: For JAVA_TOOL_OPTIONS, we MUST use '=' syntax (e.g. --add-opens=...),
            # whereas CLI 'java' command often requires space separation.
            jvm_flags = [
                "--add-opens=java.base/java.util=ALL-UNNAMED",
                "--add-opens=java.base/java.lang=ALL-UNNAMED",
                "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
                "--add-opens=java.base/java.io=ALL-UNNAMED",
                "--add-opens=java.base/java.nio=ALL-UNNAMED",
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
            ]

            env = os.environ.copy()

            if self.joern_jar:
                # Inject flags before -jar
                # For direct CLI, we might need space separation depending on Java version?
                # But --add-opens=... worked in local test via JAVA_TOOL_OPTIONS.
                # Let's trust JAVA_TOOL_OPTIONS for JAR mode as well to be consistent.
                # Or keep passing args. Java CLI usually accepts '=' too?
                # My failed test `java --add-opens=...` suggests NO string support on CLI.
                # So we should use JAVA_TOOL_OPTIONS for BOTH cases to be safe!
                cmd = ["java", "-jar", self.joern_jar, "--script", str(script_file)]
                
                # Setup env for JAR case too
                existing_tool_opts = env.get("JAVA_TOOL_OPTIONS", "")
                flags_str = " ".join(jvm_flags)
                env["JAVA_TOOL_OPTIONS"] = f"{existing_tool_opts} {flags_str}".strip()
                
            else:
                # [CCR] Reason: Use JAVA_TOOL_OPTIONS for binary wrapper.
                # 'joern' (repl-bridge) logic for arguments is brittle.
                # JAVA_TOOL_OPTIONS is automatically picked up by any JVM.
                # We must also pass -no-version-check (single dash) because JAVA_TOOL_OPTIONS output confuses the version parser.
                cmd = [self.joern_path, "-no-version-check", "--script", str(script_file)]
                
                existing_tool_opts = env.get("JAVA_TOOL_OPTIONS", "")
                flags_str = " ".join(jvm_flags)
                env["JAVA_TOOL_OPTIONS"] = f"{existing_tool_opts} {flags_str}".strip()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=temp_dir,
                env=env
            )

            if result.returncode != 0:
                raise RuntimeError(f"Joern failed: {result.stderr or result.stdout}")

            return self._parse_joern_output(result.stdout)

    def _get_default_analysis_script(self, repo_path: str) -> str:
        """Get standard Joern analysis script for Joern 4.x."""
        # Escape backslashes for Scala string literal
        escaped_path = repo_path.replace("\\", "\\\\").replace("\"", "\\\"")
        
        # Use a more robust script format for Joern 4.x
        return f'''
// Use language specific frontend if possible, fallback to generic
val cpg = try {{
  importCode.python("{escaped_path}")
}} catch {{
  case _: Exception => importCode("{escaped_path}", "analysis", "")
}}

val mJson = cpg.method.internal.map {{ m =>
  Map(
    "name" -> m.name,
    "fullName" -> m.fullName,
    "file" -> m.filename,
    "line" -> m.lineNumber.headOption.getOrElse(-1)
  )
}}.toJson

val cJson = cpg.call.map {{ c =>
  Map(
    "name" -> c.name,
    "caller" -> c.method.fullName.headOption.getOrElse(""),
    "callee" -> c.methodFullName,
    "line" -> c.lineNumber.headOption.getOrElse(-1)
  )
}}.toJson

println("---JOERN_START---")
println(s"""{{
  "total_files": ${{cpg.file.size}},
  "total_methods": ${{cpg.method.size}},
  "total_classes": ${{cpg.typeDecl.size}},
  "methods": $mJson,
  "calls": $cJson
}}""")
println("---JOERN_END---")
'''

    def _parse_joern_output(self, output: str) -> Dict[str, Any]:
        """Parse Joern output with marker support."""
        try:
            # 1. Try to find content between markers
            if "---JOERN_START---" in output and "---JOERN_END---" in output:
                start = output.find("---JOERN_START---") + len("---JOERN_START---")
                end = output.find("---JOERN_END---")
                json_part = output[start:end].strip()
                return json.loads(json_part)

            # 2. Fallback: Find JSON block in output
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(output)
        except Exception as e:
            logger.error(f"Failed to parse Joern output: {str(e)}")
            logger.debug(f"Raw output snippet: {output[:1000]}")
            return {"status": "error", "message": f"Parse failed: {str(e)}"}
