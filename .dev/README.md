# Shared group workspace (NOT part of the professor submission)

This folder is shared by all four group members for internal working files.
It is listed in `.gitignore`, so a git repo or a zip of the project can safely
exclude it without breaking anyone's local setup.

Contents:
- `instructions_professor.md` — the assignment prompt (shared reference).
- `presentation_outline.md` — the group presentation plan (shared editing).
- `archive/` — IDE / build artifacts moved out of the project root.

Convention for the shared repo:
- Do NOT commit `.dev/`, `.venv/`, `.vscode/`, `__pycache__/`, or generated
  result files; they are already ignored for every member.
- Each member runs their own `.venv` locally; the ignored path means your
  environment never leaks into the shared history.
- Keep professor-facing deliverables in the project root, not here.