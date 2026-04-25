from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_skill_surface_files_exist() -> None:
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "install.sh").exists()
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


def test_readme_documents_install_one_liner() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "install.sh | bash" in readme
    assert "CLIPPER_HARNESS" in readme


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


def test_pyproject_pins_python_and_engine_dep_set() -> None:
    """The engine extras have a known-coexisting version set verified by
    the 2026-04-25 dogfood. Unpinning any of these (especially torch /
    pyannote.audio / transformers / whisperx) breaks the install path
    via the dep-resolution cascade documented in pre-ship-fixes.md.
    Catch a future "let me bump this for flexibility" diff at CI rather
    than at the user's first install.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    # Python pin must close the upper bound while install.sh is intentionally
    # bootstrapping Python 3.12.
    assert 'requires-python = ">=3.12,<3.13"' in pyproject

    # Critical pins. Keep the torch / WhisperX / pyannote stack in sync;
    # partial upgrades tend to create resolver conflicts or import failures.
    required_pins = (
        '"torch==2.8.0"',
        '"torchaudio==2.8.0"',
        '"torchvision==0.23.0"',
        '"whisperx==3.8.5"',
        '"transformers>=4.48,<5"',
        '"pyannote.audio>=4.0,<5"',
        '"speechbrain>=1.0,<2"',
        '"matplotlib>=3.7"',
    )
    for pin in required_pins:
        assert pin in pyproject, (
            f"engine extras must pin {pin!r}; dropping this pin reopens the "
            "dep cascade documented in docs/pre-ship-fixes.md (F3)."
        )
