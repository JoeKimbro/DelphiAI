# Retired pre-yoyo migrations

The numbered SQL files in this directory (`001`–`006`) are **historical**
deltas that were hand-applied before yoyo-migrations existed. They are **already
folded into `../schemas.sql` / `../schemas_auth.sql`**, which the yoyo baseline
(`../../migrations/0001_initial.py`) reproduces in full.

**Do not run these against a fresh database** — the baseline already creates
everything they added. They are kept only as a change log.

Going forward, every schema change is a **new** numbered migration in
`DelphiAIApp/Models/migrations/` (e.g. `0002_*.py`). Stop editing `schemas.sql`
in place.
