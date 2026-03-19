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
    assert "stable food likes or dislikes" in SYSTEM_PROMPT
    assert "/memories/users/<current user_id>/preferences.md" in SYSTEM_PROMPT
    assert "Never substitute placeholder ids such as default_user or anonymous" in SYSTEM_PROMPT
    assert "Do not invent new memory filenames such as dietary_preferences.md, dietary_habits.md" in SYSTEM_PROMPT
    assert "Write preference memory as short bullet summaries in preferences.md" in SYSTEM_PROMPT
    assert "do not ask a redundant confirmation question before updating long-term memory" in SYSTEM_PROMPT
    assert "meal events, weight history, Apple Health sync state" in SYSTEM_PROMPT
    assert "inferred medical conclusions" in SYSTEM_PROMPT


def test_memory_policy_skill_documents_phase_one_core_files():
    skill_path = Path(project_root) / "app/skills/memory-policy/SKILL.md"
    content = skill_path.read_text()

    assert "/memories/users/<user_id>/profile.md" in content
    assert "/memories/users/<user_id>/preferences.md" in content
    assert "Later expansion files may exist" in content
    assert "Meal logs" in content
    assert "Apple Health sync state" in content


def test_memory_policy_skill_description_and_examples_cover_stable_preference_triggers():
    skill_path = Path(project_root) / "app/skills/memory-policy/SKILL.md"
    content = skill_path.read_text()

    assert "even if the user does not explicitly ask you to save or remember it" in content
    assert "recurring eating habit" in content
    assert '"I usually skip breakfast."' in content
    assert '"I do not eat breakfast."' in content
    assert '"I skipped breakfast today."' in content
    assert "When this skill triggers:" in content


def test_memory_policy_skill_enforces_canonical_preferences_file_and_runtime_user_id():
    skill_path = Path(project_root) / "app/skills/memory-policy/SKILL.md"
    content = skill_path.read_text()

    assert "/memories/users/<user_id>/preferences.md" in content
    assert "do not invent new filenames for dietary preference memory" in content
    assert "/memories/users/<user_id>/dietary_preferences.md" in content
    assert "/memories/users/default_user/preferences.md" in content
    assert "Always use the current runtime `user_id`" in content


def test_memory_policy_skill_prefers_direct_write_for_strong_signals_and_bullet_format():
    skill_path = Path(project_root) / "app/skills/memory-policy/SKILL.md"
    content = skill_path.read_text()

    assert 'Statements such as "I usually ..."' in content
    assert "do not ask a redundant confirmation question before writing long-term memory" in content
    assert "Use a simple bullet-list structure" in content
    assert "- Usually skips breakfast." in content
    assert "- 平时通常不吃早餐。" in content
