# GitHub Copilot Custom Instructions for TurboEngine-HAL

TurboEngine-HAL is a physics-informed digital twin for gas turbine engines.

When writing or modifying code:
- Preserve thermodynamic cycle formulas and standard station numbering (Station 0 to Station 9).
- Ensure ML surrogates use scikit-learn compatible estimator interfaces.
- For FastAPI code, use typed Pydantic models for request and response validation.
- When working with PyVista or 3D CAD visualization, account for offscreen/headless mode support.
- All unit tests should use `pytest` fixtures and mock data where appropriate.
