import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { apiFetch, ApiError } from "@/lib/api";

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockFetch(status: number, body: unknown, statusText = "") {
  const isString = typeof body === "string";
  vi.mocked(fetch).mockResolvedValue(
    new Response(isString ? body : JSON.stringify(body), {
      status,
      statusText,
      headers: { "Content-Type": isString ? "text/plain" : "application/json" },
    })
  );
}

describe("apiFetch", () => {
  it("returns parsed JSON body on 200", async () => {
    mockFetch(200, { fights: [] });
    const result = await apiFetch<{ fights: unknown[] }>("/events/1");
    expect(result).toEqual({ fights: [] });
  });

  it("throws ApiError with detail from JSON on non-OK response", async () => {
    mockFetch(404, { detail: "Event not found" }, "Not Found");
    await expect(apiFetch("/events/missing")).rejects.toMatchObject({
      status: 404,
      message: "Event not found",
    });
  });

  it("throws ApiError falling back to statusText when body is not JSON", async () => {
    mockFetch(503, "Service Unavailable", "Service Unavailable");
    await expect(apiFetch("/events/1")).rejects.toMatchObject({
      status: 503,
      message: "Service Unavailable",
    });
  });

  it("thrown error is an instance of ApiError", async () => {
    mockFetch(500, { detail: "Internal error" }, "Internal Server Error");
    await expect(apiFetch("/fail")).rejects.toBeInstanceOf(ApiError);
  });
});
