# Contributing

Issues and pull requests are welcome on GitHub.

For a bug report, include the smallest dataset (or generator script) that reproduces the issue, the full receipt text or `receipt.to_json()` output, and the Python, NumPy, and Pillow versions involved.

For a code change:

1. Add or update tests for the behavior you changed.
2. Run `python -m pytest -q`.
3. Keep the public API small. A new check should answer a concrete pre-training question ("can X alone predict the label?" / "does X cross the split?") rather than expose another internal statistic.
4. Keep the dependency ceiling: NumPy and Pillow only. No model frameworks, no embedding models.
5. Do not add benchmark claims without the script, data description, and measured output needed to reproduce them.
