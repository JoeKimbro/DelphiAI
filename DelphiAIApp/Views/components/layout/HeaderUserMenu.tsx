"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { signOut } from "next-auth/react";
import { LogIn, LogOut, User } from "lucide-react";

export function HeaderUserMenu({
  user,
}: {
  user: { name: string | null; email: string | null } | null;
}) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [open]);

  if (!user) {
    return (
      <Link
        href="/auth/signin"
        className="inline-flex items-center gap-1.5 rounded-md border border-gold/40 bg-gold/10 px-3 py-1.5 text-xs font-bold uppercase tracking-widest text-gold transition-colors hover:bg-gold hover:text-red"
      >
        <LogIn className="h-3.5 w-3.5" />
        Sign in
      </Link>
    );
  }

  const initial = (user.name || user.email || "?").charAt(0).toUpperCase();

  return (
    <div ref={wrapperRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-md border border-border bg-surface px-2 py-1.5 text-sm text-text transition-colors hover:border-gold/50"
      >
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-gold font-bold text-red">
          {initial}
        </span>
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-2 w-56 rounded-lg border border-border bg-surface p-1 shadow-xl">
          <div className="px-3 py-2">
            <div className="truncate text-sm font-medium text-text">
              {user.name || user.email}
            </div>
            {user.name && user.email && (
              <div className="truncate text-xs text-muted">{user.email}</div>
            )}
          </div>
          <div className="my-1 h-px bg-border" />
          <Link
            href="/bets"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-muted hover:bg-bg/40 hover:text-text"
          >
            <User className="h-3.5 w-3.5" />
            My bets
          </Link>
          <button
            type="button"
            onClick={() => signOut({ callbackUrl: "/" })}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-muted hover:bg-bg/40 hover:text-red"
          >
            <LogOut className="h-3.5 w-3.5" />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
