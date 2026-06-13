import { describe, it, expect, vi, beforeEach } from "vitest";

const { authMock, queryMock } = vi.hoisted(() => ({
  authMock: vi.fn(),
  queryMock: vi.fn(),
}));
vi.mock("@/lib/auth", () => ({ auth: authMock }));
vi.mock("@/lib/db", () => ({ query: queryMock }));

import { POST } from "@/app/api/bets/route";

const VALID_BET = {
  event_name: "UFC 300",
  fighter_picked: "Jon Jones",
  opponent: "Stipe Miocic",
  bet_type: "moneyline",
  units_wagered: 1,
  american_odds: -200,
};

function makeReq(body: unknown) {
  return new Request("http://localhost/api/bets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

beforeEach(() => {
  authMock.mockReset();
  queryMock.mockReset();
});

describe("POST /api/bets", () => {
  it("returns 401 when no session", async () => {
    authMock.mockResolvedValue(null);
    const res = await POST(makeReq(VALID_BET));
    expect(res.status).toBe(401);
  });

  it("returns 400 for malformed JSON", async () => {
    authMock.mockResolvedValue({ user: { id: "u1" } });
    const req = new Request("http://localhost/api/bets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not-json{{{",
    });
    const res = await POST(req);
    expect(res.status).toBe(400);
    const json = await res.json();
    expect(json.error).toBe("Invalid JSON");
  });

  it("returns 400 for Zod validation failure (negative units_wagered)", async () => {
    authMock.mockResolvedValue({ user: { id: "u1" } });
    const res = await POST(makeReq({ ...VALID_BET, units_wagered: -5 }));
    expect(res.status).toBe(400);
  });

  it("returns 201 and calls query on valid payload", async () => {
    authMock.mockResolvedValue({ user: { id: "u1" } });
    const fakeRow = { id: "b1", user_id: "u1", ...VALID_BET };
    queryMock.mockResolvedValue([fakeRow]);

    const res = await POST(makeReq(VALID_BET));
    expect(res.status).toBe(201);
    const json = await res.json();
    expect(json.bet).toMatchObject({ id: "b1" });
    expect(queryMock).toHaveBeenCalledOnce();
  });
});
