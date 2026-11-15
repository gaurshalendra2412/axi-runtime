"""quickstart_replay.py: the one-minute check of the hard test's published numbers. It must run with no key and no model, print the published figures (recomputed from the saved replies), and exit 0."""
import contextlib
import importlib.util
import io
import os

PATH = os.path.join(os.path.dirname(__file__), "..", "quickstart_replay.py")


def _run():
    spec = importlib.util.spec_from_file_location("quickstart_replay", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = mod.main()
    return code, buf.getvalue()


def test_quickstart_reproduces_the_published_figures():
    code, out = _run()
    assert code == 0, out
    for line in ("4,000 of 4,000 steps", "0 of 4,000 steps", "3,954 of 4,000 (98.85%)", "85 of 400 (21.2%)", "223 of 400", "84 of 400 (21.0%)", "400 of 400 (100.0%)", "4.5x", "4.9x",
                 "matches the published figures (4,000 of 4,000; 3,954 of 4,000; 85 / 84 / 400 / 400 of 400; 4.5x): yes"):
        assert line in out, line


def test_quickstart_needs_no_key_and_no_model():
    src = open(PATH).read()
    assert "API_KEY" not in src and "getpass" not in src and "generate(" not in src          # nothing here asks for a key or calls a model
    assert "open source" not in src.lower()
