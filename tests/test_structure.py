from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_layered_app_structure_exists():
    expected = [
        ROOT / "app" / "config.py",
        ROOT / "app" / "services" / "chat_service.py",
        ROOT / "app" / "services" / "rag_service.py",
        ROOT / "app" / "ui" / "app.py",
        ROOT / "app" / "ui" / "styles.py",
        ROOT / "app" / "ui" / "sidebar.py",
    ]
    assert all(path.is_file() for path in expected)


def test_main_is_a_thin_launcher():
    assert (ROOT / "main.py").read_text(encoding="utf-8").count("\n") <= 6
