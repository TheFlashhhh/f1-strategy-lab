# AGENTS.md

## Project Overview
- This repository is **F1 Strategy Lab**, a race-strategy analytics project.
- The project is currently complete through **Phase 2D**.
- The main user interface is `app/streamlit_app.py`.
- The main demo / walkthrough script is `app/demo_strategy.py`.

## Canonical File Rules
- Use canonical files only.
- Improve existing files directly unless the user explicitly asks for a parallel variant.
- Do not create parallel files such as `*_improved.py`, `README_IMPROVED.md`, `README_BETTER.md`, duplicate app entry points, or alternate docs that shadow the real source of truth.
- If a canonical file exists, update it instead of creating a sibling replacement.

## Repo Hygiene Rules
- Do not commit generated or local-only data.
- Do not commit `.venv`, caches, notebook noise, build artifacts, or temporary exports.
- Keep the repository root clean and focused on canonical project files.
- Reusable validation, audit, and debug scripts belong in `scripts/`.
- Documentation belongs in `docs/`.
- Do not leave one-off debug files, scratch notes, or temporary reports in the root directory.

## Workflow Rules
- Audit first, then implement.
- Make targeted edits only.
- Keep the repository runnable while changing it.
- Do not redesign the whole project unless the user explicitly asks for that.
- Preserve current working behavior unless the task is specifically to change it.
- Prefer the smallest change that satisfies the requested phase or fix.

## Verification Rules
- After meaningful code changes, always run:
- `python app/demo_strategy.py`
- `python -c "from app import streamlit_app; print('streamlit import ok')"`
- If the task is phase-specific, also run the relevant phase verification script if one exists.
- If something fails, fix minimally and rerun the affected verification.

## Scope-Control Rules
- Respect phase boundaries.
- Do not add unrelated features.
- If asked to implement a specific phase, do only that phase.
- Do not silently expand scope.
- Do not substitute a different feature for the requested phase.

## Project-Specific Roadmap Notes
- Phase 1A, Phase 1B, and Phase 1C are complete.
- Phase 2A, Phase 2B, Phase 2C, and Phase 2D are complete.
- UI theme and visual branding are deferred until later.
- Near-term work should prioritize modeling, validation, and refinement over custom theming unless the user explicitly requests UI work.

## Safety And Approval Rules
- Never commit to git or push to GitHub unless the user explicitly asks in that moment.
- The user handles commits manually unless they clearly delegate that task.
- Surface risky deletions before removing important files.
- Remove stale duplicates only when it is clearly safe and directly relevant.

## Response Format Guidance
- For most implementation tasks, structure the final response as:
- `1. Files read`
- `2. Issues found`
- `3. Changes made`
- `4. Commands run`
- `5. Verification results`
- `6. Remaining limitations`
- Keep responses concise, practical, and tied to the work actually performed.

