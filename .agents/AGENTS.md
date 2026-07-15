# Environment Setup and Dependencies

- **Streamlit Dashboard Dependencies**: Always ensure `pyvista` (or the `[viz3d]` optional dependency group) is installed alongside the main package when setting up the python environment. The Streamlit dashboard imports `pyvista` on startup and will crash with a `ModuleNotFoundError` if it is missing.
- **Install Command**: Prefer running `pip install -e ".[all]"` or specifically including the `viz3d` group: `pip install -e ".[dev,api,dashboard,reports,ml,viz3d]"`.
