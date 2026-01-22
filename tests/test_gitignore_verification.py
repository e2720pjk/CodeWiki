import tempfile
import subprocess
import shutil
import sys
import os
from pathlib import Path


def main() -> None:
    if not Path("codewiki").exists():
        print("Please run this script from the project root directory.")
        sys.exit(1)
    sys.path.insert(0, os.getcwd())

    from codewiki.src.be.dependency_analyzer.analysis.repo_analyzer import RepoAnalyzer

    git_cmd = shutil.which("git")
    if git_cmd is None:
        print("Error: git command not found. Please install git.")
        sys.exit(1)

    temp_dir = Path(tempfile.mkdtemp(prefix="codewiki_verification_"))
    print(f"Creating test fixtures in: {temp_dir}")

    try:
        subprocess.run([git_cmd, "init", "-q"], cwd=temp_dir, check=True)
        # Use *.txt files instead of *.log since *.log is in DEFAULT_IGNORE_PATTERNS
        (temp_dir / ".gitignore").write_text("node_modules/\n*.txt\nbackend/config.ini")
        (temp_dir / "backend").mkdir()
        (temp_dir / "backend" / ".gitignore").write_text("secrets.py")

        print("-" * 60)
        print("TEST 1: Gitignore Logic (Basic & Nested) with respect_gitignore=True")
        analyzer = RepoAnalyzer(respect_gitignore=True, repo_path=str(temp_dir))

        check1 = analyzer._should_exclude_path("notes.txt", "notes.txt") is True
        check2 = analyzer._should_exclude_path("readme.txt", "readme.txt") is True
        check3 = analyzer._should_exclude_path("backend/secrets.py", "secrets.py") is True
        check4 = analyzer._should_exclude_path("backend/config.ini", "config.ini") is True

        print(f"  [{'✅' if check1 else '❌'}] Basic pattern (*.txt)")
        print(f"  [{'✅' if check2 else '❌'}] Another txt file (*.txt)")
        print(f"  [{'✅' if check3 else '❌'}] Nested .gitignore")
        print(f"  [{'✅' if check4 else '❌'}] Path pattern (backend/config.ini)")

        print("\nTEST 2: Priority Logic (CLI Override > Git)")
        analyzer_override = RepoAnalyzer(
            respect_gitignore=True, repo_path=str(temp_dir), exclude_patterns=["force_exclude.py"]
        )
        check4 = (
            analyzer_override._should_exclude_path("force_exclude.py", "force_exclude.py") is True
        )
        print(f"  [{'✅' if check4 else '❌'}] CLI --exclude overrides Git tracking")

        print("\nTEST 3: Gitignore Disabled (respect_gitignore=False)")
        analyzer_no_gitignore = RepoAnalyzer(respect_gitignore=False, repo_path=str(temp_dir))

        check5 = analyzer_no_gitignore._should_exclude_path("notes.txt", "notes.txt") is False
        check6 = analyzer_no_gitignore._should_exclude_path("readme.txt", "readme.txt") is False
        check7 = (
            analyzer_no_gitignore._should_exclude_path("backend/secrets.py", "secrets.py") is False
        )

        print(f"  [{'✅' if check5 else '❌'}] *.txt files NOT excluded by gitignore")
        print(f"  [{'✅' if check6 else '❌'}] readme.txt NOT excluded by gitignore")
        print(f"  [{'✅' if check7 else '❌'}] nested .gitignore NOT respected")

        if all([check1, check2, check3, check4, check5, check6, check7]):
            print("\n✨ ALL CHECKS PASSED")
            sys.exit(0)
        else:
            print("\n🚫 SOME CHECKS FAILED")
            sys.exit(1)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
