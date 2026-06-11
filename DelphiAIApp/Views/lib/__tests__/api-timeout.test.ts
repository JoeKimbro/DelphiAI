import { describe, it, expect, vi, afterEach } from "vitest";
import { apiFetch, ApiError } from "@/lib/api";

afterEach(() => vi.restoreAllMocks());

describe("apiFetch timeout", () => {
  it("passes an AbortSignal to fetch", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    await apiFetch("/health");
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it("throws ApiError(504) when the request aborts", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      Object.assign(new Error("aborted"), { name: "AbortError" })
    );
    await expect(apiFetch("/health")).rejects.toBeInstanceOf(ApiError);
  });
});
