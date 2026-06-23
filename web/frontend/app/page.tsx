"use client";

import { useState, useEffect, useCallback, useRef } from "react";

// ── Types ─────────────────────────────────────────────────────────────────

interface PhaseOutput {
  abduction_program: string | null;
  induction_program: string | null;
}

interface LearnResponse {
  success: boolean;
  hypothesis: string[];
  all_hypotheses: string[][];
  n_examples: number;
  n_background: number;
  runtime_ms: number;
  phases: PhaseOutput;
}

// ── Benchmark presets ─────────────────────────────────────────────────────

const BENCHMARKS = {
  penguins: {
    label: "Penguins",
    icon: "🐧",
    tag: "Defaults",
    description: "Birds fly — unless they're penguins",
    source: `% Background knowledge
bird(a). bird(b). bird(c).
penguin(d).
bird(X) :- penguin(X).

% Mode declarations
#modeh flies(+bird).
#modeb penguin(+bird).
#modeb not penguin(+bird).

% Examples
#example flies(a).
#example flies(b).
#example flies(c).
#example not flies(d).`,
  },
  animals: {
    label: "Animals",
    icon: "🐾",
    tag: "Classification",
    description: "Learn what makes an animal a mammal",
    source: `% Background knowledge
has_hair(dog).   produces_milk(dog).
has_hair(cat).   produces_milk(cat).
has_hair(bat).   produces_milk(bat).
has_hair(whale). produces_milk(whale).
lays_eggs(eagle). lays_eggs(salmon).

animal(dog). animal(cat). animal(bat).
animal(whale). animal(eagle). animal(salmon).

% Mode declarations
#modeh mammal(+animal).
#modeb has_hair(+animal).
#modeb produces_milk(+animal).

% Examples
#example mammal(dog).
#example mammal(cat).
#example mammal(bat).
#example mammal(whale).
#example not mammal(eagle).
#example not mammal(salmon).`,
  },
  traffic: {
    label: "Traffic",
    icon: "🚦",
    tag: "Rules",
    description: "Learn when a car should stop",
    source: `% Background knowledge
red(light1). red(light3).
green(light2). green(light4).

light(light1). light(light2).
light(light3). light(light4).

% Mode declarations
#modeh stop(+light).
#modeb red(+light).
#modeb green(+light).

% Examples
#example stop(light1).
#example stop(light3).
#example not stop(light2).
#example not stop(light4).`,
  },
  propositional: {
    label: "Propositional",
    icon: "⊕",
    tag: "Logic",
    description: "0-arity: learn that output holds given background facts",
    source: `% Background knowledge
input_a.
input_b.

% Mode declarations (0-arity — no type variables)
#modeh output.
#modeb input_a.
#modeb input_b.

% Examples
% With only a positive example, XHAIL finds the minimal
% hypothesis: output. (no body needed — Occam's razor).
% To force body conditions, add negative examples where
% input_a or input_b are absent.
#example output.`,
  },
  sugar: {
    label: "Sugar",
    icon: "🧪",
    tag: "Priority",
    description: "Learn priority rules: glucose first, lactose when glucose is gone",
    source: `% EC Axioms
holdsAt(F,T) :- time(T), time(S), S<T, happens(E,S), initiates(E,F,S), not clipped(S,F,T).
clipped(S,F,T) :- time(S), time(T), time(R), S<R, R<T, happens(E,R), terminates(E,F,R).
holdsAt(F,T) :- time(T), initially(F), not clipped(0,F,T).

% Time
time(0). time(1). time(2). time(3). time(4). time(5).

% Domain
sugar(lactose). sugar(glucose).

% Initiation / termination
initiates(add(G), available(G), T) :- sugar(G), time(T).
terminates(use(G), available(G), T) :- sugar(G), time(T).

% Can't use sugar that isn't available
:- happens(use(G), T), not holdsAt(available(G), T).

% Both sugars added at T=0
happens(add(lactose), 0).
happens(add(glucose), 0).

% Examples
#example holdsAt(available(lactose), 1).
#example holdsAt(available(lactose), 2).
#example not holdsAt(available(lactose), 3).
#example not holdsAt(available(glucose), 2).

% Mode declarations
% '#sugar' is a ground marker: lactose/glucose appear literally.
#modeh happens(use(#sugar), +time).
#modeb holdsAt(available(#sugar), +time).
#modeb not holdsAt(available(#sugar), +time).`,
  },
  event_calculus: {
    label: "Event Calculus",
    icon: "⏱",
    tag: "Temporal",
    description: "Learn temporal rules: work happens when an agent is awake",
    source: `% Event Calculus axioms
holdsAt(F,T) :- time(T), time(S), S<T, happens(E,S), initiates(E,F,S), not clipped(S,F,T).
clipped(S,F,T) :- time(S), time(T), time(R), S<R, R<T, happens(E,R), terminates(E,F,R).
holdsAt(F,T) :- time(T), initially(F), not clipped(0,F,T).

% Domain
time(0). time(1). time(2). time(3). time(4). time(5). time(6). time(7).
person(alice).

% Background events: alice wakes at T=2, sleeps at T=6
happens(wake_up(alice), 2).
happens(sleep(alice), 6).

initiates(wake_up(P), awake(P), T) :- person(P), time(T).
terminates(sleep(P),  awake(P), T) :- person(P), time(T).
initiates(work(P), productive(P), T) :- person(P), time(T).

% Observations
#example not holdsAt(awake(alice), 1).
#example holdsAt(awake(alice), 4).
#example holdsAt(productive(alice), 5).
#example not holdsAt(productive(alice), 2).

% Mode declarations
% '#person' is a ground marker: alice appears literally in the hypothesis.
#modeh happens(work(#person), +time).
#modeb holdsAt(awake(#person), +time).`,
  },
  trains: {
    label: "Trains",
    icon: "🚂",
    tag: "Classic",
    description: "The most cited ILP benchmark: learn what makes a train eastbound",
    source: `% Michalski's Trains — classic ILP benchmark (1969)
% Task: learn what makes a train eastbound from car properties.

train(t1). train(t2). train(t3).
train(t4). train(t5). train(t6).

has_car(t1,c11). has_car(t1,c12).
has_car(t2,c21). has_car(t2,c22).
has_car(t3,c31). has_car(t3,c32).
has_car(t4,c41). has_car(t4,c42).
has_car(t5,c51). has_car(t5,c52).
has_car(t6,c61). has_car(t6,c62).
car(C) :- has_car(_, C).

short(c11). short(c21). short(c31). short(c42). short(c61).
long(c12). long(c22). long(c32). long(c41). long(c51). long(c52). long(c62).

rectangle(c11). rectangle(c21). rectangle(c31). rectangle(c42). rectangle(c52).
ellipse(c12). ellipse(c22). ellipse(c32). ellipse(c41). ellipse(c51). ellipse(c61). ellipse(c62).

triangle_load(c11). triangle_load(c21). triangle_load(c31). triangle_load(c51).
circle_load(c12). circle_load(c22). circle_load(c32).
circle_load(c41). circle_load(c42). circle_load(c52). circle_load(c61). circle_load(c62).

#modeh eastbound(+train).
#modeb has_car(+train, -car).
#modeb short(+car).
#modeb triangle_load(+car).
#modeb rectangle(+car).

#example eastbound(t1).
#example eastbound(t2).
#example eastbound(t3).
#example not eastbound(t4).
#example not eastbound(t5).
#example not eastbound(t6).`,
  },
  grandfather: {
    label: "Grandfather",
    icon: "👴",
    tag: "Relational",
    description: "Learn grandfather from family facts — demonstrates chain reasoning",
    source: `% Learn grandfather from parent and father relationships.
% Demonstrates the '-' (output) variable marker for chain reasoning.

male(tom). male(bob). male(jim).
female(liz). female(ann). female(pat).
person(X) :- male(X).
person(X) :- female(X).

parent(tom, bob). parent(tom, liz).
parent(bob, ann). parent(bob, pat).
parent(ann, jim).

father(X, Y) :- parent(X, Y), male(X).

% '+' = input variable  '-' = newly introduced (enables chaining)
#modeh grandfather(+person, +person).
#modeb father(+person, -person).
#modeb parent(+person, +person).

#example grandfather(tom, ann).
#example grandfather(tom, pat).
#example grandfather(bob, jim).
#example not grandfather(tom, bob).
#example not grandfather(tom, liz).
#example not grandfather(bob, ann).
#example not grandfather(ann, jim).`,
  },
  blocks: {
    label: "Blocks World",
    icon: "🧱",
    tag: "Planning",
    description: "EC planning: learn when a block can be picked up",
    source: `% Blocks World — learn the pick-up precondition.
% A block can be picked up only when it is clear (nothing on top).

holdsAt(F,T) :- time(T), time(S), S<T, happens(E,S), initiates(E,F,S), not clipped(S,F,T).
clipped(S,F,T) :- time(S), time(T), time(R), S<R, R<T, happens(E,R), terminates(E,F,R).
holdsAt(F,T) :- time(T), initially(F), not clipped(0,F,T).

time(0). time(1). time(2). time(3).
block(a). block(b). block(c).

initiates(pick_up(B), holding(B), T) :- block(B), time(T).
terminates(pick_up(B), clear(B),   T) :- block(B), time(T).

% Can only pick up a clear block
:- happens(pick_up(B), T), not holdsAt(clear(B), T).

% Initial state: A on B; B on table; C on table
initially(clear(a)).
initially(clear(c)).
initially(on(a, b)).
initially(on(b, table)).
initially(on(c, table)).

#example holdsAt(holding(a), 1).
#example holdsAt(holding(c), 3).
#example not holdsAt(holding(b), 1).

% '#block' is a ground marker: a/b/c appear literally in the hypothesis.
#modeh happens(pick_up(#block), +time).
#modeb holdsAt(clear(#block), +time).`,
  },
  epidemic: {
    label: "Epidemic",
    icon: "🦠",
    tag: "Temporal",
    description: "EC disease spread: learn who infects whom via contact network",
    source: `% Epidemic Spread — Event Calculus model of disease propagation.
% Expected hypothesis:
%   happens(infect(bob),   T) :- holdsAt(ill(alice), T).
%   happens(infect(carol), T) :- holdsAt(ill(bob),   T).

% ── EC Axioms ──────────────────────────────────────────────────────
holdsAt(F,T) :- time(T), time(S), S<T, happens(E,S), initiates(E,F,S), not clipped(S,F,T).
clipped(S,F,T) :- time(S), time(T), time(R), S<R, R<T, happens(E,R), terminates(E,F,R).
holdsAt(F,T) :- time(T), initially(F), not clipped(0,F,T).

% ── Domain ─────────────────────────────────────────────────────────
time(0). time(1). time(2). time(3). time(4). time(5).
person(alice). person(bob). person(carol).

initiates(infect(P),   ill(P), T) :- person(P), time(T).
initiates(contract(P), ill(P), T) :- person(P), time(T).
terminates(recover(P), ill(P), T) :- person(P), time(T).

% ── Background: alice is patient zero ──────────────────────────────
happens(contract(alice), 0).
happens(recover(alice), 4).

% ── Examples ───────────────────────────────────────────────────────
#example holdsAt(ill(alice), 1).
#example holdsAt(ill(alice), 2).
#example holdsAt(ill(bob),   2).
#example holdsAt(ill(bob),   3).
#example holdsAt(ill(carol), 3).
#example not holdsAt(ill(bob),   1).
#example not holdsAt(ill(carol), 2).
#example not holdsAt(ill(alice), 5).

% ── Mode declarations ──────────────────────────────────────────────
#modeh happens(infect(#person), +time).
#modeb holdsAt(ill(#person), +time).`,
  },
} as const;

type BenchmarkKey = keyof typeof BENCHMARKS;
type Tab = "intro" | "ilp" | "advanced";

const TAB_BENCHMARKS: Record<Tab, readonly BenchmarkKey[]> = {
  intro:    ["penguins", "animals", "traffic", "propositional"],
  ilp:      ["trains", "grandfather", "sugar"],
  advanced: ["event_calculus", "blocks", "epidemic"],
};

// ── Clipboard hook ────────────────────────────────────────────────────────

function useClipboard(timeout = 1500) {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = useCallback(
    (text: string, id: string) => {
      navigator.clipboard.writeText(text).then(() => {
        setCopied(id);
        setTimeout(() => setCopied(null), timeout);
      });
    },
    [timeout]
  );
  return { copied, copy };
}

// ── Icons ─────────────────────────────────────────────────────────────────

function IconCopy({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function IconCheck({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconPlay({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  );
}

function IconGitHub({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

function IconSpinner({ size = 14 }: { size?: number }) {
  return (
    <svg className="spin" width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <path d="M21 12a9 9 0 11-6.219-8.56" />
    </svg>
  );
}

// ── CopyButton ────────────────────────────────────────────────────────────

function CopyButton({
  text, id, copied, copy, className = "",
}: {
  text: string; id: string; copied: string | null;
  copy: (t: string, id: string) => void; className?: string;
}) {
  const done = copied === id;
  return (
    <button
      onClick={() => copy(text, id)}
      title={done ? "Copied!" : "Copy"}
      className={`transition-colors ${className}`}
      style={{ color: done ? "#86efac" : undefined }}
    >
      {done ? <IconCheck /> : <IconCopy />}
    </button>
  );
}

// ── Idle state: animated pipeline diagram + stats + pip install ───────────

function PipelineIdleState() {
  const { copied, copy } = useClipboard();
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-6 py-8 px-4 animate-fade-in">

      {/* Pipeline SVG */}
      <svg viewBox="0 0 500 110" className="w-full max-w-[480px]">
        <rect x="2" y="18" width="126" height="68" rx="11"
          fill="rgba(56,189,248,.07)" stroke="rgba(56,189,248,.28)" strokeWidth="1.5" />
        <text x="65" y="49" textAnchor="middle" fill="#7dd3fc" fontSize="11.5"
          fontFamily="JetBrains Mono, monospace" fontWeight="600" letterSpacing="1">ABDUCE</text>
        <text x="65" y="68" textAnchor="middle" fill="rgba(56,189,248,.4)" fontSize="9.5"
          fontFamily="Inter, system-ui, sans-serif">Generate candidates</text>
        <line x1="128" y1="52" x2="178" y2="52" stroke="rgba(56,189,248,.35)"
          strokeWidth="1.5" strokeDasharray="5 4">
          <animate attributeName="stroke-dashoffset" values="0;-27" dur="1s" repeatCount="indefinite" />
        </line>
        <polygon points="178,47.5 186,52 178,56.5" fill="rgba(56,189,248,.45)" />
        <rect x="187" y="18" width="126" height="68" rx="11"
          fill="rgba(6,182,212,.06)" stroke="rgba(6,182,212,.22)" strokeWidth="1.5" />
        <text x="250" y="49" textAnchor="middle" fill="#67e8f9" fontSize="11.5"
          fontFamily="JetBrains Mono, monospace" fontWeight="600" letterSpacing="1">DEDUCE</text>
        <text x="250" y="68" textAnchor="middle" fill="rgba(6,182,212,.38)" fontSize="9.5"
          fontFamily="Inter, system-ui, sans-serif">Build kernel set</text>
        <line x1="313" y1="52" x2="363" y2="52" stroke="rgba(6,182,212,.3)"
          strokeWidth="1.5" strokeDasharray="5 4">
          <animate attributeName="stroke-dashoffset" values="0;-27" dur="1s" begin="0.33s" repeatCount="indefinite" />
        </line>
        <polygon points="363,47.5 371,52 363,56.5" fill="rgba(6,182,212,.4)" />
        <rect x="372" y="18" width="126" height="68" rx="11"
          fill="rgba(74,222,128,.06)" stroke="rgba(74,222,128,.25)" strokeWidth="1.5" />
        <text x="435" y="49" textAnchor="middle" fill="#86efac" fontSize="11.5"
          fontFamily="JetBrains Mono, monospace" fontWeight="600" letterSpacing="1">INDUCE</text>
        <text x="435" y="68" textAnchor="middle" fill="rgba(74,222,128,.4)" fontSize="9.5"
          fontFamily="Inter, system-ui, sans-serif">Search hypothesis</text>
      </svg>

      {/* Key stats */}
      <div className="flex gap-2 flex-wrap justify-center">
        {[
          { value: "2ms",  label: "penguins",   color: "#38bdf8" },
          { value: "10",   label: "benchmarks", color: "#4ade80" },
          { value: "v0.1", label: "on PyPI",    color: "#86efac" },
        ].map(({ value, label, color }) => (
          <div key={label}
            className="flex flex-col items-center px-4 py-2.5 rounded-lg border"
            style={{ borderColor: `${color}22`, background: `${color}06`, minWidth: 72 }}>
            <span className="text-sm font-bold font-mono leading-none" style={{ color }}>{value}</span>
            <span className="text-[9px] text-slate-600 mt-1 uppercase tracking-wider">{label}</span>
          </div>
        ))}
      </div>

      {/* pip install */}
      <div
        className="flex items-center gap-3 px-4 py-2.5 rounded-lg border font-mono text-sm"
        style={{ borderColor: "var(--border-2)", background: "var(--surface-2)", color: "#94a3b8" }}
      >
        <span className="text-slate-600 select-none">$</span>
        <code className="flex-1 text-slate-300">pip install xhail</code>
        <button
          onClick={() => copy("pip install xhail", "pip")}
          className="text-slate-600 hover:text-slate-300 transition-colors flex-shrink-0"
          title="Copy"
        >
          {copied === "pip" ? <IconCheck size={12} /> : <IconCopy size={12} />}
        </button>
      </div>

      {/* Instructions */}
      <p className="text-slate-500 text-sm text-center max-w-sm">
        Load an example or write your own program, then press{" "}
        <kbd className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded
          bg-[var(--surface-2)] border border-[var(--border-2)]
          text-[10px] font-mono text-slate-400">
          ⌘ ↵
        </kbd>{" "}
        to run.
      </p>

      {/* Feature pills */}
      <div className="flex gap-2 flex-wrap justify-center">
        {[
          { icon: "◈", label: "Symbolic & interpretable" },
          { icon: "⚡", label: "ASP-powered"             },
          { icon: "⬡", label: "ILP / Ray (2009)"        },
        ].map(({ icon, label }) => (
          <span key={label}
            className="text-[10px] font-mono px-3 py-1 rounded-full border text-slate-600"
            style={{ borderColor: "var(--border)" }}>
            {icon} {label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Loading skeleton ──────────────────────────────────────────────────────

function LoadingState() {
  const phases = [
    { label: "Abduction",  color: "#38bdf8", desc: "Generating hypothesis candidates…" },
    { label: "Deduction",  color: "#22d3ee", desc: "Constructing kernel set…"          },
    { label: "Induction",  color: "#4ade80", desc: "Searching hypothesis space…"       },
  ];
  return (
    <div className="flex flex-col gap-3">
      {phases.map((ph, i) => (
        <div key={ph.label} className="phase-card p-4 flex items-start gap-3 animate-fade-in"
          style={{ animationDelay: `${i * 0.12}s` }}>
          <div
            className="ping-dot mt-1 w-2 h-2 rounded-full flex-shrink-0"
            style={{ background: ph.color, boxShadow: `0 0 7px ${ph.color}`, animationDelay: `${i * 0.35}s` }}
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2.5">
              <span className="text-xs font-mono font-semibold" style={{ color: ph.color }}>
                {ph.label}
              </span>
              <div className="skeleton h-2 w-20" />
            </div>
            <div className="space-y-1.5">
              <div className="skeleton h-2 w-full" />
              <div className="skeleton h-2 w-4/5" />
              <div className="skeleton h-2 w-2/3" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── StatPill ──────────────────────────────────────────────────────────────

function StatPill({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat-pill flex flex-col items-center bg-[var(--surface)]
      border border-[var(--border)] rounded-lg px-3.5 py-2.5 min-w-[70px]">
      <span className="text-[9px] uppercase tracking-widest text-slate-600 mb-0.5 font-medium">
        {label}
      </span>
      <span className="text-sm font-semibold text-slate-200 font-mono leading-tight">
        {value}
      </span>
    </div>
  );
}

// ── PhaseCard ─────────────────────────────────────────────────────────────

function PhaseCard({
  title, badge, color, content, defaultOpen = false,
}: {
  title: string; badge: string; color: string;
  content: string | null; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const { copied, copy } = useClipboard();

  return (
    <div className="phase-card">
      <div className="phase-header" onClick={() => setOpen((o) => !o)}>
        <span
          className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md"
          style={{ background: `${color}18`, color }}
        >
          {badge}
        </span>
        <span className="text-sm font-medium text-slate-300">{title}</span>
        <span className="ml-auto text-slate-600 text-[10px] select-none">
          {open ? "▲" : "▼"}
        </span>
      </div>

      {open && (
        <div className="border-t border-[var(--border)] animate-fade-in">
          {content ? (
            <div className="relative group">
              <pre
                className="text-xs font-mono p-4 overflow-x-auto text-slate-400 leading-relaxed"
                style={{ maxHeight: "260px", overflowY: "auto" }}
              >
                {content.trim()}
              </pre>
              <button
                onClick={() => copy(content.trim(), badge)}
                className="absolute top-3 right-3 opacity-0 group-hover:opacity-100
                  transition-all p-1.5 rounded-md
                  bg-[var(--surface-2)] border border-[var(--border-2)]
                  text-slate-500 hover:text-slate-200"
                title="Copy"
              >
                {copied === badge ? <IconCheck /> : <IconCopy />}
              </button>
            </div>
          ) : (
            <p className="text-xs text-slate-600 p-4 italic font-mono">
              No output captured for this phase.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────

export default function Home() {
  const [program, setProgram]               = useState<string>(BENCHMARKS.penguins.source);
  const [depth, setDepth]                   = useState(10);
  const [allSolutions, setAllSolutions]     = useState(false);
  const [activeHypIdx, setActiveHypIdx]     = useState(0);
  const [activeBm, setActiveBm]             = useState<BenchmarkKey | null>("penguins");
  const [activeTab, setActiveTab]           = useState<Tab>("intro");
  const [result, setResult]                 = useState<LearnResponse | null>(null);
  const [loading, setLoading]               = useState(false);
  const [error, setError]                   = useState<string | null>(null);
  const textareaRef                         = useRef<HTMLTextAreaElement>(null);
  const { copied, copy }                    = useClipboard();

  const apiUrl   = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";
  const lineCount = program.split("\n").length;

  // ── Run ──
  const runLearn = useCallback(async () => {
    if (loading || !program.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setActiveHypIdx(0);
    try {
      const res = await fetch(`${apiUrl}/learn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ program, depth, all_solutions: allSolutions, timeout: 30 }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json() as LearnResponse);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [loading, program, depth, allSolutions, apiUrl]);

  // ── Keyboard shortcut: Cmd/Ctrl + Enter ──
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        runLearn();
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [runLearn]);

  function selectBenchmark(key: BenchmarkKey) {
    setActiveBm(key);
    setProgram(BENCHMARKS[key].source);
    setResult(null);
    setError(null);
  }

  // ── Render ──
  return (
    <>
      {/* ── Ambient background ── */}
      <div className="fixed inset-0 pointer-events-none" style={{ zIndex: 0 }}>
        {/* Dot grid */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              "radial-gradient(circle at 1px 1px, rgba(56,189,248,.06) 1px, transparent 0)",
            backgroundSize: "28px 28px",
          }}
        />
        {/* Top radial glow — neutral */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 80% 40% at 50% -5%, rgba(255,255,255,.04) 0%, transparent 60%)",
          }}
        />
        {/* Bottom — faint green only */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 100% 35% at 50% 110%, rgba(74,222,128,.04) 0%, transparent 60%)",
          }}
        />
      </div>

      {/* ── Shell ── */}
      <div className="relative min-h-screen flex flex-col" style={{ zIndex: 1 }}>

        {/* ══════════ HEADER ══════════ */}
        <header
          className="border-b border-[var(--border)] px-5 py-3 flex items-center gap-3"
          style={{ background: "rgba(8,11,20,.85)", backdropFilter: "blur(12px)" }}
        >
          {/* Logo */}
          <div
            className="logo-glow w-8 h-8 rounded-lg flex items-center justify-center
              font-bold text-sm flex-shrink-0"
            style={{
              background: "#1e2123",
              border: "1px solid #2e3336",
              color: "#c8d4da",
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: "-0.5px",
            }}
          >
            X
          </div>

          <div className="mr-1">
            <h1 className="gradient-text text-base font-bold tracking-tight leading-none">
              XHAIL
            </h1>
            <p className="text-[10px] text-slate-600 leading-none mt-[3px]">
              eXtended Hybrid Abductive Inductive Learning
            </p>
          </div>

          <div className="hidden sm:flex items-center gap-2 ml-1">
            <span className="w-px h-5 bg-[var(--border)]" />
            {["ILP", "ASP", "Symbolic AI"].map((tag) => (
              <span
                key={tag}
                className="text-[9px] font-mono px-2 py-0.5 rounded-full
                  border border-[var(--border)] text-slate-600"
              >
                {tag}
              </span>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-2">
            <a
              href={`${apiUrl}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="ghost-btn hidden sm:inline-flex"
            >
              API Docs
            </a>
            <a
              href="https://github.com/everettmakes/xhail"
              target="_blank"
              rel="noopener noreferrer"
              className="ghost-btn"
            >
              <IconGitHub size={12} />
              GitHub
            </a>
          </div>
        </header>

        {/* ══════════ HERO STRIP ══════════ */}
        <div
          className="flex items-center justify-center gap-4 px-5 py-2 border-b border-[var(--border)] flex-wrap"
          style={{ background: "rgba(56,189,248,.03)" }}
        >
          <span className="text-[10px] text-slate-600 font-mono hidden sm:inline">
            Symbolic ILP via Answer Set Programming
          </span>
          <span className="w-px h-3 bg-[var(--border)] hidden sm:inline-block" />
          {[
            { v: "2ms",  l: "penguins",      c: "#38bdf8" },
            { v: "10",   l: "benchmarks",    c: "#4ade80" },
          ].map(({ v, l, c }) => (
            <span key={l} className="text-[10px] font-mono">
              <span style={{ color: c }} className="font-semibold">{v}</span>
              <span className="text-slate-700 ml-1">{l}</span>
            </span>
          ))}
          <span className="w-px h-3 bg-[var(--border)] hidden sm:inline-block" />
          <a
            href="https://pypi.org/project/xhail/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] font-mono text-slate-600 hover:text-sky-400 transition-colors"
          >
            pip install xhail ↗
          </a>
        </div>

        {/* ══════════ BODY ══════════ */}
        <main className="flex-1 flex flex-col lg:flex-row overflow-hidden">

          {/* ─────── LEFT PANEL ─────── */}
          <div
            className="flex flex-col gap-4 p-5 lg:w-[46%] lg:border-r border-[var(--border)] overflow-y-auto"
            style={{ minHeight: 0 }}
          >

            {/* Benchmark selector */}
            <section>
              <p className="text-[10px] uppercase tracking-widest text-slate-600 font-semibold mb-2.5">
                Load example
              </p>

              {/* Tab bar */}
              <div className="flex items-center border-b border-[var(--border)] mb-3">
                {(["intro", "ilp", "advanced"] as const).map((tab) => {
                  const labels: Record<Tab, string> = {
                    intro: "Intro",
                    ilp: "ILP Benchmarks",
                    advanced: "Advanced",
                  };
                  const active = activeTab === tab;
                  return (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className="px-3 py-2 text-[10px] font-mono font-semibold uppercase tracking-widest transition-colors"
                      style={{
                        color: active ? "#38bdf8" : "#3d4347",
                        borderBottom: active ? "1px solid #38bdf8" : "1px solid transparent",
                        marginBottom: "-1px",
                      }}
                    >
                      {labels[tab]}
                    </button>
                  );
                })}
              </div>

              {/* Benchmark cards for active tab */}
              <div className="grid grid-cols-2 gap-2">
                {TAB_BENCHMARKS[activeTab].map((key) => {
                  const bm = BENCHMARKS[key];
                  const active = activeBm === key;
                  return (
                    <button
                      key={key}
                      onClick={() => selectBenchmark(key)}
                      className={`bench-btn ${active ? "active" : ""}`}
                    >
                      <div className="flex items-start gap-2">
                        <span className="text-lg leading-none mt-0.5">{bm.icon}</span>
                        <div className="flex-1 min-w-0 text-left">
                          <div className="flex items-center gap-1.5 mb-1">
                            <span
                              className="text-xs font-semibold"
                              style={{ color: active ? "#7dd3fa" : "#94a3b8" }}
                            >
                              {bm.label}
                            </span>
                            <span
                              className="text-[8px] font-mono px-1.5 py-px rounded-full border ml-auto"
                              style={{
                                color: active ? "#38bdf8" : "#3d4347",
                                borderColor: active ? "rgba(56,189,248,.35)" : "var(--border)",
                              }}
                            >
                              {bm.tag}
                            </span>
                          </div>
                          <p
                            className="text-[10px] leading-relaxed"
                            style={{ color: active ? "#4e8ba8" : "#333a3e" }}
                          >
                            {bm.description}
                          </p>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* Editor */}
            <section className="flex flex-col flex-1">
              {/* macOS-style top bar */}
              <div
                className="flex items-center gap-2 px-3 py-2 border border-[var(--border)]
                  border-b-0 rounded-t-lg"
                style={{ background: "var(--surface-2)" }}
              >
                {/* Traffic lights */}
                <div className="flex gap-1.5 flex-shrink-0">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ background: "#ff5f57" }} />
                  <div className="w-2.5 h-2.5 rounded-full" style={{ background: "#febc2e" }} />
                  <div className="w-2.5 h-2.5 rounded-full" style={{ background: "#28c840" }} />
                </div>
                <span className="font-mono text-[10px] text-slate-500 ml-1 flex-1">program.lp</span>
                <span className="text-[10px] text-slate-700 font-mono flex-shrink-0">
                  {lineCount} ln
                </span>
                <CopyButton
                  text={program}
                  id="editor"
                  copied={copied}
                  copy={copy}
                  className="flex items-center gap-1 text-[10px] text-slate-600 hover:text-slate-300 ml-1"
                />
              </div>

              <textarea
                ref={textareaRef}
                className="code-editor flex-1"
                style={{ minHeight: "260px" }}
                value={program}
                onChange={(e) => {
                  setProgram(e.target.value);
                  setActiveBm(null);
                }}
                spellCheck={false}
                placeholder={`% Write your XHAIL program here\n% ─────────────────────────────\nbird(tweety).\n\n#modeh flies(+bird).\n#modeb bird(+bird).\n\n#example flies(tweety).`}
              />
            </section>

            {/* Controls row */}
            <div className="flex items-end gap-4">
              {/* Depth slider */}
              <div className="flex flex-col gap-1.5 flex-1">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] uppercase tracking-widest text-slate-600 font-semibold">
                    Deduction depth
                  </label>
                  <span
                    className="text-xs font-mono text-slate-300 px-2 py-px rounded
                      bg-[var(--surface-2)] border border-[var(--border)]"
                  >
                    {depth}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={50}
                  value={depth}
                  onChange={(e) => setDepth(Number(e.target.value))}
                  className="w-full accent-slate-400"
                />
              </div>

              {/* All solutions toggle */}
              <button
                onClick={() => setAllSolutions((v) => !v)}
                title="Enumerate all optimal hypotheses"
                className="flex flex-col items-center gap-1 flex-shrink-0 pb-0.5"
              >
                <span className="text-[9px] uppercase tracking-widest font-semibold"
                  style={{ color: allSolutions ? "#38bdf8" : "#3d4347" }}>
                  All
                </span>
                <div
                  className="w-8 h-4 rounded-full transition-colors relative"
                  style={{
                    background: allSolutions ? "rgba(56,189,248,.25)" : "var(--surface-2)",
                    border: `1px solid ${allSolutions ? "rgba(56,189,248,.5)" : "var(--border-2)"}`,
                  }}
                >
                  <div
                    className="absolute top-0.5 w-3 h-3 rounded-full transition-all"
                    style={{
                      left: allSolutions ? "calc(100% - 14px)" : "2px",
                      background: allSolutions ? "#38bdf8" : "#2e3336",
                    }}
                  />
                </div>
              </button>

              {/* Run button */}
              <button
                onClick={runLearn}
                disabled={loading || !program.trim()}
                className="run-btn px-5 py-2.5 rounded-lg text-sm font-semibold
                  flex items-center gap-2 flex-shrink-0"
              >
                {loading ? (
                  <>
                    <IconSpinner />
                    Running…
                  </>
                ) : (
                  <>
                    <IconPlay />
                    Run
                    <kbd className="text-[9px] opacity-50 font-mono ml-0.5">⌘↵</kbd>
                  </>
                )}
              </button>
            </div>

            {/* Syntax reference (collapsible) */}
            <details className="border border-[var(--border)] rounded-lg overflow-hidden">
              <summary
                className="px-3.5 py-2.5 cursor-pointer flex items-center justify-between
                  text-[10px] uppercase tracking-widest text-slate-600 font-semibold
                  hover:bg-[var(--surface)] transition-colors"
              >
                <span>Syntax reference</span>
                <span className="text-[8px]">▼</span>
              </summary>
              <div
                className="px-4 pt-3 pb-4 border-t border-[var(--border)] flex flex-col gap-2"
                style={{ background: "var(--surface)" }}
              >
                {[
                  { code: "#modeh flies(+bird).",            color: "#38bdf8", desc: "Head mode — predicate to learn" },
                  { code: "#modeb penguin(+bird).",          color: "#22d3ee", desc: "Body mode — available conditions" },
                  { code: "#modeh flies(+bird) :1-3 =2 @1.", color: "#0ea5e9", desc: "Mode with min-max, weight, priority" },
                  { code: "#example flies(a).",              color: "#4ade80", desc: "Positive example" },
                  { code: "#example not flies(d).",          color: "#86efac", desc: "Negative example" },
                  { code: "#example flies(a) =5 @2.",        color: "#34d399", desc: "Weighted/prioritised example" },
                  { code: "#display flies/1.",               color: "#a78bfa", desc: "Filter output to this predicate" },
                ].map(({ code, color, desc }) => (
                  <div key={code}
                    className="flex items-center gap-3 px-3 py-2 rounded-md border"
                    style={{ borderColor: `${color}1a`, background: `${color}06` }}>
                    <code className="font-mono text-[11px] whitespace-nowrap flex-shrink-0" style={{ color }}>
                      {code}
                    </code>
                    <span className="text-[10px] text-slate-600 ml-auto text-right">{desc}</span>
                  </div>
                ))}
                <div
                  className="flex items-center gap-3 px-3 py-2 rounded-md border mt-1"
                  style={{ borderColor: "var(--border)", background: "var(--surface-2)" }}>
                  <code className="font-mono text-[11px] whitespace-nowrap text-slate-500">
                    fact(a).&nbsp;&nbsp;<span style={{ color: "#475569" }}>r(X) :- body(X).</span>
                  </code>
                  <span className="text-[10px] text-slate-600 ml-auto text-right">Background knowledge (ASP)</span>
                </div>
                <p className="text-[10px] text-slate-700 pt-1">
                  <span className="text-slate-600">+type</span> = input variable &nbsp;·&nbsp;
                  <span className="text-slate-600">#type</span> = ground constant &nbsp;·&nbsp;
                  <span className="text-slate-600">-type</span> = output variable
                </p>
              </div>
            </details>
          </div>

          {/* ─────── RIGHT PANEL ─────── */}
          <div
            className="flex flex-col gap-4 p-5 lg:flex-1 overflow-y-auto"
            style={{ minHeight: 0 }}
          >
            {/* Panel header */}
            <div className="flex items-center justify-between flex-shrink-0">
              <p className="text-[10px] uppercase tracking-widest text-slate-600 font-semibold">
                Pipeline output
              </p>
              {result && (
                <span
                  className="animate-fade-up text-[10px] font-mono px-2.5 py-0.5 rounded-full border"
                  style={result.success
                    ? { color: "#86efac", borderColor: "rgba(74,222,128,.3)", background: "rgba(74,222,128,.07)" }
                    : { color: "#f87171", borderColor: "rgba(248,113,113,.3)", background: "rgba(248,113,113,.06)" }
                  }
                >
                  {result.success ? "✓ Hypothesis found" : "✗ No hypothesis"}
                </span>
              )}
            </div>

            {/* ── Error ── */}
            {error && (
              <div
                className="animate-fade-up border rounded-lg p-4"
                style={{
                  borderColor: "rgba(248,113,113,.35)",
                  background: "rgba(127,29,29,.18)",
                }}
              >
                <p className="text-xs font-semibold text-red-400 mb-1.5 uppercase tracking-wide">
                  Error
                </p>
                <pre className="text-sm text-red-300 font-mono whitespace-pre-wrap leading-relaxed">
                  {error}
                </pre>
              </div>
            )}

            {/* ── Idle ── */}
            {!result && !error && !loading && <PipelineIdleState />}

            {/* ── Loading ── */}
            {loading && <LoadingState />}

            {/* ── Results ── */}
            {result && (
              <div className="flex flex-col gap-5 animate-fade-in">

                {/* Stats row */}
                <div className="flex gap-2 flex-wrap">
                  <StatPill label="Runtime"    value={`${result.runtime_ms.toFixed(0)} ms`} />
                  <StatPill label="Rules"      value={result.hypothesis.length}              />
                  <StatPill label="Examples"   value={result.n_examples}                     />
                  <StatPill label="Background" value={result.n_background}                   />
                  {result.all_hypotheses.length > 1 && (
                    <StatPill label="Optimal"  value={result.all_hypotheses.length}          />
                  )}
                </div>

                {/* Learned hypothesis */}
                <section>
                  <div className="flex items-center justify-between mb-2.5">
                    <p className="text-[10px] uppercase tracking-widest text-slate-600 font-semibold">
                      {result.all_hypotheses.length > 1
                        ? `Hypothesis ${activeHypIdx + 1} of ${result.all_hypotheses.length}`
                        : "Learned hypothesis"}
                    </p>
                    {/* Hypothesis navigator */}
                    {result.all_hypotheses.length > 1 && (
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => setActiveHypIdx((i) => Math.max(0, i - 1))}
                          disabled={activeHypIdx === 0}
                          className="w-5 h-5 rounded flex items-center justify-center text-[10px]
                            border border-[var(--border)] text-slate-500 hover:text-slate-200
                            disabled:opacity-25 transition-colors"
                        >‹</button>
                        {result.all_hypotheses.map((_, i) => (
                          <button
                            key={i}
                            onClick={() => setActiveHypIdx(i)}
                            className="w-5 h-5 rounded flex items-center justify-center text-[9px] font-mono
                              border transition-colors"
                            style={{
                              borderColor: i === activeHypIdx ? "rgba(56,189,248,.5)" : "var(--border)",
                              background: i === activeHypIdx ? "rgba(56,189,248,.1)" : "transparent",
                              color: i === activeHypIdx ? "#38bdf8" : "#475569",
                            }}
                          >{i + 1}</button>
                        ))}
                        <button
                          onClick={() => setActiveHypIdx((i) => Math.min(result.all_hypotheses.length - 1, i + 1))}
                          disabled={activeHypIdx === result.all_hypotheses.length - 1}
                          className="w-5 h-5 rounded flex items-center justify-center text-[10px]
                            border border-[var(--border)] text-slate-500 hover:text-slate-200
                            disabled:opacity-25 transition-colors"
                        >›</button>
                      </div>
                    )}
                  </div>

                  {(() => {
                    const displayRules = result.all_hypotheses.length > 1
                      ? (result.all_hypotheses[activeHypIdx] ?? [])
                      : result.hypothesis;
                    return result.success ? (
                      <div className="flex flex-col gap-2">
                        {displayRules.map((rule, i) => (
                          <div key={i} className="hypothesis-rule" style={{ animationDelay: `${i * 0.06}s` }}>
                            <code className="flex-1 min-w-0">{rule}</code>
                            <button
                              onClick={() => copy(rule, `rule-${activeHypIdx}-${i}`)}
                              className="flex-shrink-0 text-green-700 hover:text-green-300 transition-colors"
                              title="Copy rule"
                            >
                              {copied === `rule-${activeHypIdx}-${i}` ? <IconCheck /> : <IconCopy />}
                            </button>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div
                        className="animate-fade-up border rounded-lg p-4 text-sm font-mono"
                        style={{
                          borderColor: "rgba(251,191,36,.25)",
                          background: "rgba(120,53,15,.15)",
                          color: "rgba(251,191,36,.75)",
                        }}
                      >
                        No hypothesis found — try loosening mode declarations or increasing depth.
                      </div>
                    );
                  })()}
                </section>

                {/* Pipeline trace */}
                <section>
                  <p className="text-[10px] uppercase tracking-widest text-slate-600 font-semibold mb-2.5">
                    Pipeline trace
                  </p>
                  <div className="flex flex-col gap-2">
                    <PhaseCard
                      title="Abduction — minimal grounding"
                      badge="Phase 1"
                      color="#38bdf8"
                      content={result.phases.abduction_program}
                      defaultOpen
                    />
                    <PhaseCard
                      title="Deduction — kernel construction"
                      badge="Phase 2"
                      color="#22d3ee"
                      content={
                        "% Deduction builds the kernel set in working memory.\n" +
                        "% The intermediate representation is not a serialisable ASP program.\n" +
                        "% See Ray (2009) §4.2 for the formal definition of the kernel set."
                      }
                    />
                    <PhaseCard
                      title="Induction — hypothesis search"
                      badge="Phase 3"
                      color="#4ade80"
                      content={result.phases.induction_program}
                    />
                  </div>
                </section>
              </div>
            )}
          </div>
        </main>

        {/* ══════════ FOOTER ══════════ */}
        <footer
          className="border-t border-[var(--border)] px-5 py-2.5 flex items-center gap-4 flex-shrink-0 flex-wrap"
          style={{ background: "rgba(8,11,20,.8)", backdropFilter: "blur(8px)" }}
        >
          <span className="text-[10px] text-slate-700 font-mono">
            XHAIL · ILP via ASP
          </span>
          <span className="w-px h-3 bg-[var(--border)] hidden sm:inline-block" />
          <a href="https://doi.org/10.1007/s10994-008-5083-8" target="_blank" rel="noopener noreferrer"
            className="text-[10px] text-slate-700 hover:text-slate-400 transition-colors">
            Ray (2009)
          </a>
          <span className="w-px h-3 bg-[var(--border)] hidden sm:inline-block" />
          <a href="https://pypi.org/project/xhail/" target="_blank" rel="noopener noreferrer"
            className="text-[10px] text-slate-700 hover:text-sky-400 transition-colors font-mono">
            pypi · xhail
          </a>
          <span className="w-px h-3 bg-[var(--border)] hidden sm:inline-block" />
          <a href="https://github.com/everettmakes/xhail" target="_blank" rel="noopener noreferrer"
            className="text-[10px] text-slate-700 hover:text-slate-400 transition-colors">
            GitHub
          </a>
          <span className="ml-auto text-[10px] text-slate-800 font-mono">v0.1.1</span>
        </footer>
      </div>
    </>
  );
}
