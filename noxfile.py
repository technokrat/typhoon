import nox


@nox.session
def build(session):
    """Build the Rust extension using maturin."""
    session.install("maturin")
    session.run("maturin", "build", "--release")


@nox.session
def develop(session):
    """Build and install in editable mode."""
    session.install("maturin")
    session.run("maturin", "develop")


@nox.session
def test(session):
    """Run Python tests using pytest."""
    session.install("maturin")
    session.run("maturin", "develop")
    session.install("pytest", "numpy", "pytest-benchmark")
    session.install("-e", ".")
    session.run("pytest", "tests")


@nox.session
def lint(session):
    """Lint Python code."""
    session.install("ruff", "black")
    session.run("ruff", "check")
    session.run("black", "--check", ".")
