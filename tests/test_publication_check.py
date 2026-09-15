from pathlib import Path

from scripts.check_publication import scan


def test_publication_check_accepts_technical_documentation(tmp_path: Path) -> None:
    document = tmp_path / "architecture.md"
    document.write_text(
        "# Serving architecture\nThe API reads versioned evaluation artifacts.\n",
        encoding="utf-8",
    )

    assert scan([document], tmp_path) == []


def test_publication_check_reports_rule_without_content(tmp_path: Path) -> None:
    document = tmp_path / "notes.md"
    prohibited = "interview " + "preparation"
    document.write_text(f"Heading\n{prohibited}\n", encoding="utf-8")

    assert scan([document], tmp_path) == [
        ("notes.md", 2, "coaching.interview"),
    ]


def test_publication_check_flags_suspicious_document_name(tmp_path: Path) -> None:
    document = tmp_path / ("career-" + "coaching.md")
    document.write_text("generic content\n", encoding="utf-8")

    assert scan([document], tmp_path) == [
        (document.name, 0, "document.suspicious-name"),
    ]
