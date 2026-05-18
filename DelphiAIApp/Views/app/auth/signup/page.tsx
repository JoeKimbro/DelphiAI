"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { signIn } from "next-auth/react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

export default function SignUpPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setLoading(true);
    const res = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name: name || undefined }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setLoading(false);
      setError(body?.error ?? "Sign-up failed");
      return;
    }
    // Auto sign-in after registration.
    const signin = await signIn("credentials", {
      email,
      password,
      redirect: false,
    });
    setLoading(false);
    if (signin?.error) {
      setError("Account created but sign-in failed. Try signing in manually.");
      return;
    }
    router.push("/bets");
    router.refresh();
  }

  return (
    <div className="mx-auto max-w-md px-6 py-16">
      <motion.form
        onSubmit={onSubmit}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-4 rounded-2xl border border-border bg-surface p-8 shadow-xl"
      >
        <div>
          <h1 className="text-3xl font-black tracking-tight text-text">
            Create account
          </h1>
          <p className="mt-1 text-sm text-muted">
            Track your wagers and compare against the model.
          </p>
        </div>

        <Field label="Name (optional)" type="text" value={name} onChange={setName} />
        <Field label="Email" type="email" value={email} onChange={setEmail} required />
        <Field
          label="Password (min 8)"
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
          className="flex w-full items-center justify-center gap-2 rounded-md bg-gold px-4 py-2.5 text-sm font-bold uppercase tracking-widest text-red transition-colors hover:bg-gold-bright disabled:opacity-60"
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          Create Account
        </button>

        <p className="text-center text-xs text-muted">
          Already have an account?{" "}
          <Link href="/auth/signin" className="font-medium text-gold hover:underline">
            Sign in
          </Link>
        </p>
      </motion.form>
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
