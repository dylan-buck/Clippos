from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_skill_surface_files_exist() -> None:
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "commands" / "clip.md").exists()
    assert (ROOT / "commands" / "clip-config.md").exists()
    assert (ROOT / ".claude-plugin" / "plugin.json").exists()
    assert (ROOT / ".codex-plugin" / "plugin.json").exists()


def test_clip_command_invokes_clip_skill() -> None:
    command = (ROOT / "commands" / "clip.md").read_text(encoding="utf-8")

    assert "Invoke the `clip` skill" in command
    assert "$ARGUMENTS" in command


def test_skill_instructions_reference_helper_script() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "scripts/clip_skill.py" in skill
    assert "scoring-response.json" in skill
    assert "approved" in skill
