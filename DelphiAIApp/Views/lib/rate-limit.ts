import { query } from "./db";

// Tunables (see spec): 5 failed logins / 15 min → lockout; 3 signups / 15 min / IP.
const LOGIN_WINDOW_MIN = 15;
const LOGIN_MAX_FAILS = 5;
const SIGNUP_WINDOW_MIN = 15;
const SIGNUP_MAX = 3;

/**
 * Best-effort client IP for throttling. Prefers `x-real-ip`, which Vercel's
 * edge sets to the actual peer IP and a client cannot override. Only falls back
 * to the leftmost `x-forwarded-for` hop when `x-real-ip` is absent (non-Vercel
 * / local dev) — that hop IS client-spoofable, so it is a soft signal there and
 * the edge WAF is the real volumetric defense (see docs/SECURITY.md).
 */
export function clientIp(headers: Headers): string {
  const real = headers.get("x-real-ip")?.trim();
  if (real) return real;
  const xff = headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  return "unknown";
}

/** True when (email, ip) has ≥ LOGIN_MAX_FAILS failures since the last success in-window. */
export async function isLockedOut(email: string, ip: string): Promise<boolean> {
  const rows = await query<{ fails: string }>(
    `SELECT COUNT(*) AS fails
       FROM auth_attempts
      WHERE email = $1 AND ip = $2
        AND success = FALSE
        AND attempted_at > NOW() - ($3 || ' minutes')::interval
        AND attempted_at > COALESCE(
              (SELECT MAX(attempted_at) FROM auth_attempts
                WHERE email = $1 AND ip = $2 AND success = TRUE),
              '-infinity'::timestamptz)`,
    [email, ip, String(LOGIN_WINDOW_MIN)]
  );
  return Number(rows[0]?.fails ?? 0) >= LOGIN_MAX_FAILS;
}

/** True when this IP has created ≥ SIGNUP_MAX accounts in the window. */
export async function tooManySignups(ip: string): Promise<boolean> {
  const rows = await query<{ n: string }>(
    `SELECT COUNT(*) AS n
       FROM auth_attempts
      WHERE ip = $1 AND email = '__signup__' AND success = TRUE
        AND attempted_at > NOW() - ($2 || ' minutes')::interval`,
    [ip, String(SIGNUP_WINDOW_MIN)]
  );
  return Number(rows[0]?.n ?? 0) >= SIGNUP_MAX;
}

/** Log an attempt. Best-effort: also prunes rows older than 24h. */
export async function recordAttempt(email: string, ip: string, success: boolean): Promise<void> {
  await query(
    `INSERT INTO auth_attempts (email, ip, success) VALUES ($1, $2, $3)`,
    [email, ip, success]
  );
  // Cheap opportunistic cleanup; ignore failures.
  query(`DELETE FROM auth_attempts WHERE attempted_at < NOW() - interval '24 hours'`).catch(
    () => {}
  );
}
