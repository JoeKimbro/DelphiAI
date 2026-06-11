-- 006: brute-force lockout + signup-throttle support
CREATE TABLE IF NOT EXISTS auth_attempts (
    id              BIGSERIAL PRIMARY KEY,
    email           TEXT NOT NULL,
    ip              TEXT NOT NULL,
    success         BOOLEAN NOT NULL DEFAULT FALSE,
    attempted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_lookup
    ON auth_attempts (email, ip, attempted_at);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip
    ON auth_attempts (ip, attempted_at);
