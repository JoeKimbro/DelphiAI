# Design: Frontend + Backend Test Coverage for Important Functions

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan

## Goal

Add focused, deterministic tests for the important *untested* functions on both
sides of DelphiAI:

- **Backend** (Python / pytest) — extend the existing suite under
  `DelphiAIApp/Models/tests/`.
- **Frontend** (TypeScript) — stand up a Vitest setup in `DelphiAIApp/Views/`
  (no JS test runner exists today) and unit-test the `lib/` helpers and the
  bet-creation API route handler.

`pytest` cannot execute TypeScript, so "frontend pytests" is satisfied with
Vitest — the idiomatic test runner for a Next.js + TS app.

## Scope (Approach A + light route handlers)

Test **pure, deterministic logic** with no live DB or network dependency, plus a
small slice of route-handler logic where the validation lives in the handler
itself. This keeps the new tests fast, CI-friendly (no Docker/Postgres needed),
and free of the flakiness the existing DB/scraper integration tests carry.

### Explicitly out of scope
- React component render tests (no behavior-critical pure logic in components).
- Live-Postgres / running-FastAPI integration (already covered by the existing
  suite: `test_api_endpoints.py`, `test_event_service_db.py`, etc.).
- Scraper network tests.

## Frontend — new Vitest setup in `Views/`

### Tooling
- Add devDependencies: `vitest`, `@testing-library/react`, `@testing-library/jest-dom`,
  `jsdom`, `@vitejs/plugin-react`.
- Add `"test": "vitest run"` and `"test:watch": "vitest"` scripts to `package.json`.
- Add `vitest.config.ts`: `environment: "jsdom"`, `globals: true`, and resolve the
  `@/*` path alias (matches `tsconfig.json` → `"@/*": ["./*"]`) so handler imports
  like `@/lib/auth` work under test.

### Test files
1. `lib/__tests__/format.test.ts` — `lib/format.ts`:
   - `formatPercent`: `null`/`undefined`/`NaN` → `"—"`; rounding at default 0 digits
     and explicit digits.
   - `formatAmericanOdds`: positive → `+N`; negative → `-N`; nullish → `"—"`.
   - `formatElo`: rounds; `null` → `"—"`.
   - `formatEloDelta`: `0` → `"0"`; positive → `▲ +N`; negative → `▼ N`; `null` → `""`.
   - `slugifyFighter`: lowercases, strips punctuation, collapses spaces to `-`.
   - `shortDate`: valid ISO → formatted; `null` → `"—"`; unparseable → echoes input.
   - `confidenceColor`: each of `HIGH`/`MED`/`LOW`/`TOSS` returns the expected
     `{bg,text,border}` triple.
2. `lib/__tests__/utils.test.ts` — `cn()` merges classes and dedupes conflicting
   Tailwind utilities (e.g. `px-2` + `px-4` → `px-4`).
3. `lib/__tests__/api.test.ts` — `apiFetch` with a stubbed global `fetch`:
   - 200 → returns parsed JSON body.
   - non-OK with JSON `{detail}` → throws `ApiError` whose `status`/`message`
     match `detail`.
   - non-OK with non-JSON body → throws `ApiError` falling back to `statusText`.
4. `app/api/bets/__tests__/route.test.ts` — `POST` handler from
   `app/api/bets/route.ts`, mocking `@/lib/auth` (`auth`) and `@/lib/db` (`query`):
   - no session → `401`.
   - malformed JSON body → `400 "Invalid JSON"`.
   - zod validation failure (e.g. negative `units_wagered`) → `400`.
   - valid payload + authed session → `200`/`201` and `query` called with the
     inserted row. (Exact status asserted against the handler's actual return.)

## Backend — extend the pytest suite

1. `Models/tests/test_odds_math.py` — `DelphiAIApp/Services/odds_math.py`
   (the canonical "single source of truth"; the existing `test_odds_conversion.py`
   targets a different/older `ml/` module, so this is a genuine gap):
   - `american_to_decimal`: favorite (`-200` → `1.5`), underdog (`+150` → `2.5`),
     `0` raises `ValueError`, `None` raises `ValueError`.
   - `decimal_to_american`: round-trips with `american_to_decimal` for fav and dog;
     `decimal <= 1.0` and `None` raise `ValueError`.
   - `decimal_to_implied`: `2.0` → `0.5`; `<= 0`/`None` → `0.0`.
   - `remove_vig`: two-way market normalizes to sum `1.0`; non-positive total →
     `(0.5, 0.5)`.
   - `compute_edge_pct`: positive when model is more bullish than de-vigged market,
     negative when less; guard inputs (`None`, `decimal <= 1.0`) → `0.0`.
2. `Models/tests/test_performance_service.py` — `performance_service._jsonable`
   (pure recursive `Decimal` → `float` coercion, no DB):
   - bare `Decimal` → `float`.
   - nested `dict` and `list` containing `Decimal` are converted recursively.
   - non-`Decimal` scalars (`int`, `str`, `None`) pass through unchanged.

These import their targets directly (`from DelphiAIApp.Services.odds_math import ...`),
relying on the existing `DelphiAIApp/tests/conftest.py` sys.path setup. The
backend files needing it will live where conftest's path injection applies; if
`Models/tests/` is not covered by that conftest at run time, the new backend
tests add a minimal local sys.path shim consistent with the sibling test files.

## How to run

```bash
# Backend
python -m pytest DelphiAIApp/Models/tests/test_odds_math.py DelphiAIApp/Models/tests/test_performance_service.py -v

# Frontend
cd DelphiAIApp/Views && npm install && npm test
```

## Success criteria
- All new backend tests pass via `pytest` without Docker/Postgres running.
- `npm test` in `Views/` runs Vitest green with the new files.
- No changes to production code (tests + test tooling/config only).
