"""
Repository Analyzer Module

This module provides functionality to analyze repository structures and generate
detailed file tree representations with filtering capabilities.
"""

import os
import fnmatch
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
from codewiki.src.be.dependency_analyzer.utils.patterns import (
    DEFAULT_IGNORE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS,
)
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern


class RepoAnalyzer:
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        respect_gitignore: bool = False,
        repo_path: Optional[str] = None,
    ) -> None:
        # Include patterns: if specified, use ONLY those patterns (replaces defaults)
        self.include_patterns = (
            include_patterns if include_patterns is not None else DEFAULT_INCLUDE_PATTERNS
        )
        # Exclude patterns: if specified, MERGE with default ignore patterns
        self.exclude_patterns = (
            list(DEFAULT_IGNORE_PATTERNS) + exclude_patterns
            if exclude_patterns is not None
            else list(DEFAULT_IGNORE_PATTERNS)
        )
        # User-specified exclude patterns only (separated from defaults)
        self._user_exclude_patterns = exclude_patterns if exclude_patterns is not None else []
        self.respect_gitignore = respect_gitignore
        self.repo_path = repo_path
        self.gitignore_spec = None
        self.include_spec = None
        self._git_path = shutil.which("git")
        self._git_available = self._check_git_availability()

        if include_patterns:
            try:
                self.include_spec = PathSpec.from_lines(GitWildMatchPattern, self.include_patterns)
            except ImportError:
                pass

        if self.respect_gitignore:
            self._load_gitignore_patterns()

    def analyze_repository_structure(self, repo_dir: str) -> Optional[Dict]:
        file_tree = self._build_file_tree(repo_dir)
        if file_tree is None:
            return None
        return {
            "file_tree": file_tree,
            "summary": {
                "total_files": self._count_files(file_tree),
                "total_size_kb": self._calculate_size(file_tree),
            },
        }

    def _load_gitignore_patterns(self):
        if not self.repo_path:
            self.gitignore_spec = None
            return

        gitignore_file = os.path.join(self.repo_path, ".gitignore")
        if os.path.exists(gitignore_file):
            with open(gitignore_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.gitignore_spec = PathSpec.from_lines(GitWildMatchPattern, lines)
        else:
            self.gitignore_spec = None

    def _check_git_availability(self) -> bool:
        if self._git_path is None:
            return False
        try:
            result = subprocess.run([self._git_path, "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def _is_ignored_by_git(self, path: str) -> Optional[bool]:
        if not self.repo_path:
            return None

        if not self._git_available:
            return None

        assert self._git_path is not None

        try:
            full_path = os.path.join(self.repo_path, path)
            result = subprocess.run(
                [self._git_path, "check-ignore", "--quiet", full_path],
                cwd=self.repo_path,
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _build_file_tree(self, repo_dir: str) -> Optional[Dict]:
        def build_tree(path: Path, base_path: Path) -> Optional[Dict]:
            relative_path = path.relative_to(base_path)
            relative_path_str = str(relative_path)

            # 🚫 Reject symlinks
            if path.is_symlink():
                return None

            # 🚫 Reject escaped paths (e.g., symlinks pointing outside)
            try:
                if not path.resolve().is_relative_to(base_path.resolve()):
                    return None
            except AttributeError:
                if not str(path.resolve()).startswith(str(base_path.resolve())):
                    return None

            if self._should_exclude_path(relative_path_str, path.name):
                return None

            if path.is_file():
                if not self._should_include_file(relative_path_str, path.name):
                    return None

                size = path.stat().st_size
                return {
                    "type": "file",
                    "name": path.name,
                    "path": relative_path_str,
                    "extension": path.suffix,
                    "_size_bytes": size,
                }

            elif path.is_dir():
                children = []
                try:
                    for child in sorted(path.iterdir()):
                        child_tree = build_tree(child, base_path)
                        if child_tree is not None:
                            children.append(child_tree)
                except PermissionError:
                    pass

                if children or str(relative_path) == ".":
                    return {
                        "type": "directory",
                        "name": path.name,
                        "path": relative_path_str,
                        "children": children,
                    }
                return None

            # Other types (sockets, devices, etc.)
            return None

        return build_tree(Path(repo_dir), Path(repo_dir))

    def _should_exclude_path(self, path: str, filename: str) -> bool:
        if self.respect_gitignore:
            git_ignored = self._is_ignored_by_git(path)

            if git_ignored is True:
                return True

            if git_ignored is False:
                for pattern in self._user_exclude_patterns:
                    if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
                        return True
                    if pattern.endswith("/") and path.startswith(pattern.rstrip("/")):
                        return True
                    if path.startswith(pattern + "/") or path == pattern:
                        return True
                    if pattern in path.split("/"):
                        return True
                # Allow default ignore patterns to be checked (remove early return)

            if (
                git_ignored is None
                and hasattr(self, "gitignore_spec")
                and self.gitignore_spec is not None
            ):
                if self.gitignore_spec.match_file(path):
                    return True
                if not path.endswith("/") and self.gitignore_spec.match_file(path + "/"):
                    return True

        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
            if pattern.endswith("/") and path.startswith(pattern.rstrip("/")):
                return True
            if path.startswith(pattern + "/") or path == pattern:
                return True
            if pattern in path.split("/"):
                return True

        return False

    def _should_include_file(self, path: str, filename: str) -> bool:
        if not self.include_patterns:
            return True

        if self.include_spec and self.include_spec.match_file(path):
            return True

        for pattern in self.include_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _count_files(self, tree: Dict) -> int:
        if tree["type"] == "file":
            return 1
        return sum(self._count_files(child) for child in tree.get("children", []))

    def _calculate_size(self, tree: Dict) -> float:
        if tree["type"] == "file":
            return tree.get("_size_bytes", 0) / 1024
        return sum(self._calculate_size(child) for child in tree.get("children", []))
