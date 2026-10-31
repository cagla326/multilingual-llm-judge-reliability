# Formatted reading copies

These copies make the Python source easier to inspect. Black 25.1.0 expands compressed suites and semicolon-separated statements and applies consistent indentation and line wrapping. All 40 copies have identical Python abstract syntax trees to their originals (including docstrings; location metadata excluded).

The original files elsewhere in this package remain the reproduction entry points and retain their historical bytes. Use these copies for reading, not as replacements for frozen execution: changing source bytes changes hash checks, and runtime-relative data/protocol paths are not duplicated here. No scientific decisions or result values have changed. FORMATTING_MAP.json records both byte identities.
