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


def test_skill_is_hermes_first_with_claude_codex_fallback() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    # Hermes is the first-class target…
    assert "${HERMES_SKILL_DIR}" in skill
    assert "/clip config" in skill
    assert "/clip package" in skill

    # …but the prologue must fall through to CLAUDE_PLUGIN_ROOT so the skill
    # still works as a Claude Code / Codex plugin, and HERMES_SKILL_DIR must
    # appear earlier in the chain than CLAUDE_PLUGIN_ROOT.
    assert "CLAUDE_PLUGIN_ROOT" in skill
    assert "/clip-config" in skill
    assert "/clip-package" in skill
    assert skill.index("${HERMES_SKILL_DIR}") < skill.index("CLAUDE_PLUGIN_ROOT")


def test_skill_routes_hermes_flow_through_hermes_clip_helper() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "scripts/hermes_clip.py" in skill
    assert "advance --source" in skill
    assert "next_action" in skill
    assert (ROOT / "scripts" / "hermes_clip.py").exists()


def test_skill_documents_messaging_attachment_and_creator_profile_hooks() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    # Attachment URL guidance so Hermes knows to pass Discord/Telegram CDN
    # URLs straight through to the skill.
    assert "Discord" in skill
    assert "Telegram" in skill
    assert "attachment" in skill.lower()

    # Creator-profile memory wiring at both model handoffs.
    assert "Creator Profile" in skill
    assert "creator profile" in skill.lower() or "Creator profile" in skill


def test_skill_documents_feedback_loop_and_creator_patterns() -> None:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    # The feedback command must be documented where Hermes can see it.
    assert "Feedback Loop" in skill
    assert "hermes_clip.py feedback" in skill
    assert "feedback-log.json" in skill
    assert "history.jsonl" in skill

    # creator_patterns must appear at the scoring + packaging handoffs so the
    # harness knows to read it from the advance payload.
    assert "creator_patterns" in skill
    assert skill.count("creator_patterns") >= 2
