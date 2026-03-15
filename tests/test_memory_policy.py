import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.recipe_app import SYSTEM_PROMPT


def test_system_prompt_scopes_long_term_memory_to_core_files_and_exclusions():
    assert "/memories/users/<user_id>/" in SYSTEM_PROMPT
    assert "profile.md" in SYSTEM_PROMPT
    assert "preferences.md" in SYSTEM_PROMPT
    assert "goals.md and shopping.md may exist later" in SYSTEM_PROMPT
    assert "meal logs, weight history, Apple Health sync state" in SYSTEM_PROMPT
    assert "Do not write inferred medical conclusions" in SYSTEM_PROMPT


def test_memory_policy_skill_documents_phase_one_core_files():
    skill_path = Path(project_root) / "app/skills/memory-policy/SKILL.md"
    content = skill_path.read_text()

    assert "/memories/users/<user_id>/profile.md" in content
    assert "/memories/users/<user_id>/preferences.md" in content
    assert "Later expansion files may exist" in content
    assert "Meal logs" in content
    assert "Apple Health sync state" in content
