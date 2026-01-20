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
        (temp_dir / ".gitignore").write_text("node_modules/\n*.log\n!important.log")
        (temp_dir / "backend").mkdir()
        (temp_dir / "backend" / ".gitignore").write_text("secrets.py")

        (temp_dir / "app.log").touch()
        (temp_dir / "important.log").touch()
        (temp_dir / "backend" / "secrets.py").touch()
        (temp_dir / "backend" / "api.py").touch()
        (temp_dir / "force_exclude.py").touch()

        print("-" * 60)
        print("TEST 1: Gitignore Logic (Negation & Nested)")
        analyzer = RepoAnalyzer(respect_gitignore=True, repo_path=str(temp_dir))

        check1 = analyzer._should_exclude_path("app.log", "app.log") is True
        check2 = analyzer._should_exclude_path("important.log", "important.log") is False
        check3 = analyzer._should_exclude_path("backend/secrets.py", "secrets.py") is True

        print(f"  [{'✅' if check1 else '❌'}] Basic pattern (*.log)")
        print(f"  [{'✅' if check2 else '❌'}] Negation pattern (!important.log)")
        print(f"  [{'✅' if check3 else '❌'}] Nested .gitignore")

        print("\nTEST 2: Priority Logic (CLI Override > Git)")
        analyzer_override = RepoAnalyzer(
            respect_gitignore=True, repo_path=str(temp_dir), exclude_patterns=["force_exclude.py"]
        )
        check4 = (
            analyzer_override._should_exclude_path("force_exclude.py", "force_exclude.py") is True
        )
        print(f"  [{'✅' if check4 else '❌'}] CLI --exclude overrides Git tracking")

        if all([check1, check2, check3, check4]):
            print("\n✨ ALL CHECKS PASSED")
            sys.exit(0)
        else:
            print("\n🚫 SOME CHECKS FAILED")
            sys.exit(1)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
