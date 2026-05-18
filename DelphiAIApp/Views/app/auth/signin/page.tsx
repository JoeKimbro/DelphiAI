"use client";

import { useState, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { signIn } from "next-auth/react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

function SignInForm() {
  const router = useRouter();
  const params = useSearchParams();
  const callbackUrl = params.get("callbackUrl") || "/bets";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    const res = await signIn("credentials", {
      email,
      password,
      redirect: false,
    });
    setLoading(false);
    if (res?.error) {
      setError("Invalid email or password");
      return;
    }
    router.push(callbackUrl);
    router.refresh();
  }

  return (
    <motion.form
      onSubmit={onSubmit}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-4 rounded-2xl border border-border bg-surface p-8 shadow-xl"
    >
      <div>
        <h1 className="text-3xl font-black tracking-tight text-text">
          Sign in
        </h1>
        <p className="mt-1 text-sm text-muted">
          Access your bet log and personal ROI tracking.
        </p>
      </div>

      <Field label="Email" type="email" value={email} onChange={setEmail} required />
      <Field
        label="Password"
        type="password"
        value={password}
        onChange={setPassword}
        required
      />

      {error && (
        <div className="rounded-md border border-red/40 bg-red/10 px-3 py-2 text-sm text-red">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-md bg-red px-4 py-2.5 text-sm font-bold uppercase tracking-widest text-text transition-colors hover:bg-red/90 disabled:opacity-60"
      >
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        Sign in
      </button>

      <p className="text-center text-xs text-muted">
        New here?{" "}
        <Link href="/auth/signup" className="font-medium text-gold hover:underline">
          Create an account
        </Link>
      </p>
    </motion.form>
  );
}

export default function SignInPage() {
  return (
    <div className="mx-auto max-w-md px-6 py-16">
      <Suspense fallback={<div className="text-muted">Loading…</div>}>
        <SignInForm />
      </Suspense>
    </div>
  );
}

function Field({
  label,
  type,
  value,
  onChange,
  required,
}: {
  label: string;
  type: string;
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-[10px] font-bold uppercase tracking-widest text-muted">
        {label}
      </span>
      <input
        type={type}
        value={value}
        required={required}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-md border border-border bg-bg/60 px-3 py-2 text-sm text-text outline-none transition-colors focus:border-gold"
      />
    </label>
  );
}
