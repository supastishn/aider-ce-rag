import os
from pathlib import Path

import pytest


def have_llama_index():
    try:
        import llama_index  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not have_llama_index(), reason="llama_index not installed")
def test_repo_rag_basic(tmp_path):
    # Create tiny repo tree
    proj = tmp_path / "proj"
    proj.mkdir()

    # Two small files
    (proj / "README.md").write_text("""
# Sample Project

Initialize the database using init_db() in db.py.
""".strip())
    (proj / "db.py").write_text(
        """
def init_db():
    return "ok"
""".strip()
    )

    # Optional .aiderignore to ensure it's honored (not excluding our files)
    (proj / ".aiderignore").write_text("# nothing ignored\n")

    # Build index under a temp cache dir to avoid touching user cache
    cache_dir = tmp_path / "cache"

    from aider.rag import RepoRAG

    rag = RepoRAG(project_root=proj, persist_dir=cache_dir)
    rag.build_index(force_reindex=True, quiet=True)

    # Query a known phrase
    ctx = rag.query("How do I initialize the database?", top_k=3)

    # Should reference db.py or include snippet
    assert "db.py" in ctx or "init_db" in ctx

