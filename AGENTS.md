# Repository Guidelines

## Project Structure

- `d2d_beamforming/` encodes the D2D interference graph with a numerical Transformer and trains an output head for continuous complex beam weights. Its README documents the DeepSeek experiment and local smoke command.
- `d2d_beam_selector/` contains the NumPy-based D2D beam selector. The CLI lives in `__main__.py`, CSI loading and weighted sum-rate (WSR) scoring in `core.py`, and API configuration and transport in `config.py` and `provider.py`.
- `swan_isac_forward/` and `swan_isac_repro/` each contain a PyTorch model in `model.py` and a standalone CLI in `forward.py`. Check both copies when changing shared behavior. The root `swan_isac_forward.zip` is a distribution archive.
- Each component has a README with input formats and usage examples. The root `d2d_provider.json` holds default provider settings.

## Local Development

Use Python 3.10 or later in a virtual environment. Run the following commands from the repository root.

- `python -m pip install numpy` installs the D2D dependency. Run `python -m d2d_beam_selector --demo --offline` to score the built-in scene without an API request.
- `python -m d2d_beam_selector --check-config` checks local provider settings without contacting the endpoint. Supply credentials through an environment variable or a private key file.
- `python -m pip install "torch>=2.4"` installs the SWAN dependency. Run `python swan_isac_forward/forward.py` or `python swan_isac_repro/forward.py` to exercise the default backbone on synthetic CSI. See the component README for optional GPT-2 dependencies.

## Coding Style

Use four-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and a leading underscore for internal helpers. Match nearby type annotations. Document tensor axis order, complex dtypes, and power units when changing numerical interfaces.

## Verification

Use the relevant smoke commands above. For D2D scoring changes, compare per-link rates and WSR against the exhaustive baseline on the deterministic demo. For SWAN changes, verify output shapes and complex beam weights in each affected copy. This checkout has no configured test suite, coverage target, formatter, or linter.

## Commits and Pull Requests

Git history is unavailable in this checkout. Use concise, imperative commit subjects such as `Validate CSI pairing order`. In pull requests, describe the behavior change, link any related issue, and list the commands and results used to verify it. Include a small input or output example when a CLI or numerical interface changes.
