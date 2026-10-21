"""tools/frameworks.py on a stand-in framework (a local git repository as a pinned submodule): record a change as
a patch, apply and check the patches, reset to the pin."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), "tools", "frameworks.py")


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True).stdout


def tool(root, *args, check=True):
    r = subprocess.run([sys.executable, TOOL, *args], env={**os.environ, "SHAKTI_AI_ROOT": str(root)},
                       text=True, capture_output=True)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_record_apply_check_reset(tmp_path):
    up = tmp_path / "upstream"
    up.mkdir()
    sh("git", "init", "-q", "-b", "main", cwd=up)
    (up / "kernel.c").write_text("int add(int a, int b) { return a + b; }\n")
    sh("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "x", "--allow-empty", cwd=up)
    sh("git", "add", "kernel.c", cwd=up)
    sh("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "kernel", cwd=up)
    pin = sh("git", "rev-parse", "HEAD", cwd=up).strip()

    root = tmp_path / "sdk"
    (root / "patches" / "tvm").mkdir(parents=True)
    sh("git", "init", "-q", "-b", "main", cwd=root)
    sh("git", "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(up), "frameworks/tvm", cwd=root)
    fw = root / "frameworks" / "tvm"
    assert tool(root, "status").stdout.split()[:3] == ["tvm", pin[:12], "fetched"]

    (fw / "kernel.c").write_text("int add(int a, int b) { return a + b; }  /* shakti */\n")
    tool(root, "record", "tvm", "Mark SHAKTI")
    patches = sorted(os.listdir(root / "patches" / "tvm"))
    assert patches == ["0001-mark-shakti.patch"]
    assert "shakti" not in (fw / "kernel.c").read_text() or sh("git", "status", "--porcelain", cwd=fw)
    tool(root, "reset", "tvm")
    assert "shakti" not in (fw / "kernel.c").read_text()

    tool(root, "apply", "tvm")
    assert "/* shakti */" in (fw / "kernel.c").read_text()
    (fw / "extra.h").write_text("#define X 1\n")                 # a second change on top: the next patch only
    tool(root, "record", "tvm", "extra header")
    assert sorted(os.listdir(root / "patches" / "tvm")) == ["0001-mark-shakti.patch", "0002-extra-header.patch"]
    assert "extra.h" in (root / "patches" / "tvm" / "0002-extra-header.patch").read_text()
    assert "kernel.c" not in (root / "patches" / "tvm" / "0002-extra-header.patch").read_text()

    assert tool(root, "check", "tvm").returncode == 0
    assert sh("git", "rev-parse", "HEAD", cwd=fw).strip() == pin and not sh("git", "status", "--porcelain", cwd=fw)
    (root / "patches" / "tvm" / "0003-broken.patch").write_text("--- a/nope.c\n+++ b/nope.c\n@@ -1 +1 @@\n-x\n+y\n")
    r = tool(root, "check", "tvm", check=False)
    assert r.returncode == 1 and "0003-broken.patch: does not apply" in r.stdout
