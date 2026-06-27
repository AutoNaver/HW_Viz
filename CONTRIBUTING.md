# Contributing

## Principles

- Keep the MVP small and testable.
- Prefer pure functions in `hw_viz/` over logic embedded in the Streamlit layer.
- Document any modeling assumption that changes outputs or exported data.
- Add or update tests for any numerical or workflow change.

## Development Flow

1. Install the project with `python -m pip install -e .`
2. Run `pytest -q`
3. Run `streamlit run app.py`
4. Make the smallest coherent change
5. Update docs if behavior, interfaces, exports, or assumptions changed

## Pull Request Checklist

- The app still runs locally.
- The test suite passes.
- New public behavior is documented in `README.md` or `docs/`.
- Export formats remain stable, or the breaking change is documented clearly.
- Numerical assumptions are called out explicitly.
