"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Loader2, Plus, X } from "lucide-react";
import { useCreateBet } from "@/lib/queries";

const BET_TYPES = ["moneyline", "method", "round", "parlay"] as const;

export function BetForm() {
  const [open, setOpen] = useState(false);
  const [eventName, setEventName] = useState("");
  const [fighter, setFighter] = useState("");
  const [opponent, setOpponent] = useState("");
  const [betType, setBetType] = useState<(typeof BET_TYPES)[number]>("moneyline");
  const [units, setUnits] = useState("1");
  const [odds, setOdds] = useState("-110");
  const [modelProb, setModelProb] = useState("");
  const [notes, setNotes] = useState("");

  const mutation = useCreateBet();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    await mutation.mutateAsync({
      event_name: eventName,
      event_id: null,
      fighter_picked: fighter,
      opponent,
      bet_type: betType,
      units_wagered: Number(units),
      american_odds: Number(odds),
      model_prob: modelProb ? Number(modelProb) : null,
      notes: notes || null,
    });
    // reset
    setEventName("");
    setFighter("");
    setOpponent("");
    setUnits("1");
    setOdds("-110");
    setModelProb("");
    setNotes("");
    setOpen(false);
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 rounded-md bg-gold px-4 py-2 text-sm font-bold uppercase tracking-widest text-red transition-colors hover:bg-gold-bright"
      >
        <Plus className="h-4 w-4" />
        Log new bet
      </button>
    );
  }

  return (
    <motion.form
      onSubmit={submit}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-border bg-surface p-6 shadow-xl"
    >
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-black tracking-tight text-text">New Bet</h2>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="rounded-md p-1 text-muted hover:bg-bg/40 hover:text-text"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Field label="Event" value={eventName} onChange={setEventName} required placeholder="UFC 326" />
        <Select
          label="Bet type"
          value={betType}
          onChange={(v) => setBetType(v as typeof betType)}
          options={BET_TYPES as unknown as string[]}
        />
        <Field
          label="Fighter (your pick)"
          value={fighter}
          onChange={setFighter}
          required
        />
        <Field label="Opponent" value={opponent} onChange={setOpponent} required />
        <Field label="Units wagered" type="number" value={units} onChange={setUnits} required />
        <Field label="American odds" type="number" value={odds} onChange={setOdds} required />
        <Field
          label="Model prob (0–1, optional)"
          type="number"
          value={modelProb}
          onChange={setModelProb}
          step="0.0001"
        />
        <Field label="Notes" value={notes} onChange={setNotes} />
      </div>

      {mutation.error && (
        <div className="mt-4 rounded-md border border-red/40 bg-red/10 px-3 py-2 text-sm text-red">
          {(mutation.error as Error).message}
        </div>
      )}

      <div className="mt-6 flex items-center gap-3">
        <button
          type="submit"
          disabled={mutation.isPending}
          className="inline-flex items-center gap-2 rounded-md bg-red px-4 py-2 text-sm font-bold uppercase tracking-widest text-text transition-colors hover:bg-red/90 disabled:opacity-60"
        >
          {mutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          Save Bet
        </button>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="rounded-md px-4 py-2 text-sm font-medium text-muted hover:text-text"
        >
          Cancel
        </button>
      </div>
    </motion.form>
  );
}

function Field({
  label,
  value,
  onChange,
  required,
  type = "text",
  placeholder,
  step,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
  type?: string;
  placeholder?: string;
  step?: string;
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
        placeholder={placeholder}
        step={step}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-md border border-border bg-bg/60 px-3 py-2 text-sm text-text outline-none transition-colors focus:border-gold"
      />
    </label>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <label className="block">
      <span className="text-[10px] font-bold uppercase tracking-widest text-muted">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-md border border-border bg-bg/60 px-3 py-2 text-sm capitalize text-text outline-none transition-colors focus:border-gold"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}
