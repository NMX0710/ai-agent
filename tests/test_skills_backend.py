import sys
from pathlib import Path
from types import SimpleNamespace

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.recipe_app import RecipeApp, SKILLS_ROUTE


def test_skills_route_uses_filesystem_backend_and_lists_project_skills():
    runtime = SimpleNamespace(
        state={},
        store=None,
        config={},
        context=None,
        stream_writer=None,
    )

    backend = RecipeApp._build_backend(runtime, skills_root=Path(project_root) / "app/skills")

    infos = backend.ls_info(SKILLS_ROUTE)
    paths = sorted(item["path"] for item in infos if item.get("is_dir"))

    assert "/skills/meal-log-flow/" in paths
    assert "/skills/memory-policy/" in paths
    assert "/skills/nutrition-lookup/" in paths


def test_skills_route_reads_skill_markdown_from_project_filesystem():
    runtime = SimpleNamespace(
        state={},
        store=None,
        config={},
        context=None,
        stream_writer=None,
    )

    backend = RecipeApp._build_backend(runtime, skills_root=Path(project_root) / "app/skills")
    response = backend.download_files(["/skills/meal-log-flow/SKILL.md"])[0]

    assert response.error is None
    assert response.content is not None
    content = response.content.decode("utf-8")
    assert "name: meal-log-flow" in content
    assert "Follow lookup -> choose final estimate -> prepare -> show estimate -> explicit confirm -> commit flow." in content
