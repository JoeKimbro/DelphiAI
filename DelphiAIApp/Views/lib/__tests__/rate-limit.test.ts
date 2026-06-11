import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the db module before importing the unit under test. vi.mock is hoisted
// above normal declarations, so queryMock must be created via vi.hoisted to be
// available inside the (also-hoisted) factory.
const { queryMock } = vi.hoisted(() => ({ queryMock: vi.fn() }));
vi.mock("@/lib/db", () => ({ query: queryMock }));

import { clientIp, isLockedOut, recordAttempt, tooManySignups } from "@/lib/rate-limit";

beforeEach(() => queryMock.mockReset());

describe("clientIp", () => {
  it("reads the first x-forwarded-for hop", () => {
    const h = new Headers({ "x-forwarded-for": "1.2.3.4, 10.0.0.1" });
    expect(clientIp(h)).toBe("1.2.3.4");
  });
  it("falls back to 'unknown'", () => {
    expect(clientIp(new Headers())).toBe("unknown");
  });
});

describe("isLockedOut", () => {
  it("locks out at the failure threshold", async () => {
    queryMock.mockResolvedValueOnce([{ fails: "5" }]);
    expect(await isLockedOut("a@b.com", "1.2.3.4")).toBe(true);
  });
  it("allows below threshold", async () => {
    queryMock.mockResolvedValueOnce([{ fails: "4" }]);
    expect(await isLockedOut("a@b.com", "1.2.3.4")).toBe(false);
  });
});

describe("tooManySignups", () => {
  it("blocks past the per-IP signup cap", async () => {
    queryMock.mockResolvedValueOnce([{ n: "3" }]);
    expect(await tooManySignups("9.9.9.9")).toBe(true);
  });
});

describe("recordAttempt", () => {
  it("inserts a row (then opportunistically prunes)", async () => {
    // Both the INSERT and the best-effort cleanup DELETE go through query, so
    // give every call a catchable resolved value.
    queryMock.mockResolvedValue([]);
    await recordAttempt("a@b.com", "1.2.3.4", false);
    const sql = queryMock.mock.calls[0][0] as string;
    expect(sql).toMatch(/insert into auth_attempts/i);
  });
});
