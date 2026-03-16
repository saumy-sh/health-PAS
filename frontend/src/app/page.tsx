"use client";

import Link from "next/link";
import { ArrowRight, Clock, FileX, Zap, Brain, ShieldCheck, BarChart3, RefreshCw, Lock, CheckCircle } from "lucide-react";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

// ── Animated reveal wrapper ───────────────────────────────────────────────────
function FadeIn({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 28 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({
  value,
  label,
  source,
  accent = false,
}: {
  value: string;
  label: string;
  source: string;
  accent?: boolean;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      className="group relative p-8 border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-blue-400 dark:hover:border-blue-500 transition-all duration-300 overflow-hidden"
    >
      <div
        className={`absolute top-0 left-0 h-[3px] w-0 group-hover:w-full transition-all duration-500 ${
          accent ? "bg-blue-500" : "bg-slate-300 dark:bg-slate-700"
        }`}
      />
      <div
        className={`text-5xl lg:text-[3.5rem] font-black tracking-tighter mb-3 leading-none ${
          accent
            ? "text-blue-600 dark:text-blue-400"
            : "text-slate-900 dark:text-slate-100"
        }`}
      >
        {value}
      </div>
      <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 leading-snug mb-3">
        {label}
      </div>
      <div className="text-[10px] font-bold text-slate-400 dark:text-slate-600 uppercase tracking-widest">
        {source}
      </div>
    </motion.div>
  );
}

// ── Benefit card ──────────────────────────────────────────────────────────────
function BenefitCard({
  icon: Icon,
  title,
  body,
  index,
}: {
  icon: any;
  title: string;
  body: string;
  index: number;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 16 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{
        duration: 0.5,
        delay: (index % 3) * 0.07,
        ease: [0.22, 1, 0.36, 1],
      }}
      className="flex gap-5 p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-blue-300 dark:hover:border-blue-700 hover:shadow-lg hover:shadow-blue-500/5 transition-all duration-300 group"
    >
      <div className="shrink-0 mt-0.5">
        <div className="h-10 w-10 rounded-lg bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center group-hover:bg-blue-600 transition-colors duration-300">
          <Icon
            size={19}
            className="text-blue-600 dark:text-blue-400 group-hover:text-white transition-colors duration-300"
          />
        </div>
      </div>
      <div>
        <div className="font-bold text-slate-900 dark:text-slate-100 mb-1.5">
          {title}
        </div>
        <div className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
          {body}
        </div>
      </div>
    </motion.div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function Home() {
  return (
    <div className="min-h-screen bg-white dark:bg-slate-950">

      {/* ── Sticky nav ── */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-slate-200/70 dark:border-slate-800/70 bg-white/90 dark:bg-slate-950/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="h-5 w-5 text-blue-600" strokeWidth={2.5} />
            <span className="font-black text-base tracking-tight text-slate-900 dark:text-white">
              InsuranceHelper
            </span>
          </div>
          <Link
            href="/analyse"
            className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold rounded-full transition-colors"
          >
            Start Analysis <ArrowRight size={14} />
          </Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="relative pt-32 pb-24 px-6 overflow-hidden">
        {/* Grid texture */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f1f5f9_1px,transparent_1px),linear-gradient(to_bottom,#f1f5f9_1px,transparent_1px)] dark:bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:56px_56px] opacity-60 dark:opacity-40" />
        {/* Glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-blue-500/6 dark:bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative max-w-7xl mx-auto lg:flex lg:items-center lg:gap-16">
          {/* Left */}
          <div className="flex-1">
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-blue-200 dark:border-blue-900 bg-blue-50 dark:bg-blue-950/50 mb-8"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" />
              <span className="text-[11px] font-black text-blue-700 dark:text-blue-400 uppercase tracking-[0.18em]">
                AI-Powered · Six-Agent Pipeline · v2.0
              </span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
              className="text-5xl sm:text-6xl lg:text-7xl font-black tracking-tighter text-slate-900 dark:text-white leading-[0.93] mb-6"
            >
              Prior auth,{" "}
              <br />
              <span className="text-blue-600 dark:text-blue-400">finally</span>{" "}
              automated.
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.18 }}
              className="text-lg text-slate-500 dark:text-slate-400 leading-relaxed max-w-xl mb-10"
            >
              Upload patient documents once. Six specialized AI agents extract,
              validate, and reason through clinical eligibility — delivering a
              complete pre-authorization report in under two minutes.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: 0.28 }}
              className="flex flex-wrap items-center gap-4"
            >
              <Link
                href="/analyse"
                className="group inline-flex items-center gap-2.5 px-8 py-4 bg-slate-900 dark:bg-white text-white dark:text-slate-900 font-black text-sm rounded-full hover:bg-blue-600 dark:hover:bg-blue-600 dark:hover:text-white transition-all shadow-xl shadow-slate-900/20"
              >
                Analyse Documents
                <ArrowRight
                  size={15}
                  className="group-hover:translate-x-1 transition-transform"
                />
              </Link>
              <span className="text-sm text-slate-400 dark:text-slate-600">
                No login · HIPAA compliant
              </span>
            </motion.div>
          </div>

          {/* Right — terminal mockup */}
          <motion.div
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="flex-1 mt-16 lg:mt-0 max-w-lg"
          >
            <div className="bg-slate-950 rounded-xl border border-slate-800 shadow-2xl shadow-slate-900/50 overflow-hidden">
              <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800 bg-slate-900/60">
                <div className="h-2.5 w-2.5 rounded-full bg-red-500/60" />
                <div className="h-2.5 w-2.5 rounded-full bg-yellow-500/60" />
                <div className="h-2.5 w-2.5 rounded-full bg-green-500/60" />
                <span className="ml-2 text-[11px] font-mono text-slate-600">
                  pipeline.log
                </span>
              </div>
              <div className="p-5 font-mono text-[12px] space-y-2.5 leading-relaxed">
                <div className="text-blue-400">
                  ▶ [Agent 1] OCR — 7 documents identified
                </div>
                <div className="text-slate-500 pl-3">
                  Clinical Note · Insurance ID · Lab Report · +4
                </div>
                <div className="text-emerald-400">
                  ✓ [Agent 2] BlueCross BlueShield PPO · BCBS-99001122
                </div>
                <div className="text-emerald-400">
                  ✓ [Agent 3] MRI Lumbar Spine (CPT 72148) — requirements loaded
                </div>
                <div className="text-amber-400">
                  ⚠ [Agent 4] Conflict: symptom duration
                </div>
                <div className="text-slate-500 pl-3">
                  Clinical Note: 2 wks · Estimate: 7 wks → conservative: 2 wks
                </div>
                <div className="text-blue-400">
                  ▶ [Agent 5] Eligibility · probability: 62% PENDING_REVIEW
                </div>
                <div className="text-emerald-400">
                  ✓ [Agent 6] Report generated
                </div>
                <div className="h-px bg-slate-800 my-1" />
                <div className="text-slate-600 text-[10px] uppercase tracking-widest">
                  Total elapsed: 1m 47s
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Section divider ── */}
      <div className="max-w-7xl mx-auto px-6">
        <div className="border-t border-slate-200 dark:border-slate-800" />
      </div>

      {/* ── Why AI: Stats ── */}
      <section className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <FadeIn>
            <span className="text-[11px] font-black uppercase tracking-[0.2em] text-blue-600 dark:text-blue-500">
              The Problem
            </span>
          </FadeIn>
          <FadeIn delay={0.06} className="mt-3 mb-4">
            <h2 className="text-4xl sm:text-5xl font-black tracking-tighter text-slate-900 dark:text-white leading-tight">
              Why AI for prior authorization?
            </h2>
          </FadeIn>
          <FadeIn delay={0.1} className="mb-14">
            <p className="text-lg text-slate-500 dark:text-slate-400 max-w-2xl leading-relaxed">
              The prior-authorization system isn't broken because the medicine is
              wrong. It's broken because the paperwork takes longer than the
              treatment. These numbers tell the story.
            </p>
          </FadeIn>

          {/* Stats — borderless grid joined together */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-px bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-800">
            <StatCard
              value="94%"
              label="of physicians say prior auth delays necessary patient care"
              source="AMA Prior Authorization Survey, 2023"
              accent
            />
            <StatCard
              value="33%"
              label="of patients experience a serious adverse medical event while waiting for authorization"
              source="AMA Prior Authorization Survey, 2023"
            />
            <StatCard
              value="45 hrs"
              label="spent by physicians and staff per week on prior authorization requests"
              source="AMA Physician Practice Benchmark Survey"
            />
            <StatCard
              value="1 in 4"
              label="patients abandon recommended treatment entirely due to prior auth delays"
              source="AMA Prior Authorization Survey, 2023"
            />
            <StatCard
              value="$528M"
              label="spent annually by U.S. physicians solely on prior authorization administration"
              source="JAMA Internal Medicine"
              accent
            />
            <StatCard
              value="82%"
              label="of physicians report prior authorization negatively impacts clinical outcomes"
              source="AMA Prior Authorization Physician Report, 2022"
            />
          </div>

          {/* Pull quote */}
          <FadeIn delay={0.1} className="mt-14">
            <blockquote className="pl-7 border-l-[3px] border-blue-500">
              <p className="text-xl font-semibold italic text-slate-600 dark:text-slate-300 leading-relaxed max-w-3xl">
                "Prior authorization is the single biggest administrative burden
                in medicine today. Every hour spent on paperwork is an hour not
                spent on patients."
              </p>
              <footer className="mt-3 text-xs font-bold text-slate-400 dark:text-slate-600 uppercase tracking-wider">
                American Medical Association · 2023 Prior Authorization Physician Survey
              </footer>
            </blockquote>
          </FadeIn>
        </div>
      </section>

      {/* ── Section divider ── */}
      <div className="max-w-7xl mx-auto px-6">
        <div className="border-t border-slate-200 dark:border-slate-800" />
      </div>

      {/* ── What makes us different ── */}
      <section className="py-24 px-6 bg-slate-50 dark:bg-slate-950/60">
        <div className="max-w-7xl mx-auto">
          <FadeIn>
            <span className="text-[11px] font-black uppercase tracking-[0.2em] text-blue-600 dark:text-blue-500">
              The Solution
            </span>
          </FadeIn>
          <FadeIn delay={0.06} className="mt-3 mb-4">
            <h2 className="text-4xl sm:text-5xl font-black tracking-tighter text-slate-900 dark:text-white leading-tight">
              What makes InsuranceHelper different
            </h2>
          </FadeIn>
          <FadeIn delay={0.1} className="mb-14">
            <p className="text-lg text-slate-500 dark:text-slate-400 max-w-2xl leading-relaxed">
              Most tools just fill forms. We run a full clinical reasoning
              pipeline that reads your documents the way an expert reviewer
              would — catching conflicts, verifying criteria, and producing a
              complete determination before you submit.
            </p>
          </FadeIn>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <BenefitCard
              icon={Brain}
              index={0}
              title="Six-agent pipeline — fully end-to-end"
              body="Documents flow through six specialized agents: OCR extraction, policy field parsing, requirements retrieval, document verification, clinical eligibility assessment, and final report generation. Each agent does one job and passes clean structured data to the next."
            />
            <BenefitCard
              icon={RefreshCw}
              index={1}
              title="Conflict detection across documents"
              body="When two documents contradict each other — say a clinical note records 2 weeks of symptoms while a cost estimate says 7 weeks — the pipeline flags the conflict and uses the most conservative value for threshold checks. No cherry-picking evidence in favour of approval."
            />
            <BenefitCard
              icon={FileX}
              index={2}
              title="Smart document gap identification"
              body="Agent 4 checks every submitted document against the insurer's specific requirements for the requested procedure, giving you a precise checklist — satisfied, incomplete, or missing — before you hit submit."
            />
            <BenefitCard
              icon={BarChart3}
              index={3}
              title="Approval probability with evidence trail"
              body="Agent 5 produces an approval probability with per-requirement status (met / partial / not met), document citations for each finding, and specific denial reasons where criteria aren't satisfied. No black box — full auditability."
            />
            <BenefitCard
              icon={Zap}
              index={4}
              title="Under 2 minutes from upload to report"
              body="The full pipeline — OCR through final JSON report — runs in under two minutes for a typical patient file. No queuing, no human reviewer scheduling, no waiting days for a response."
            />
            <BenefitCard
              icon={Lock}
              index={5}
              title="HIPAA-compliant, no data retention"
              body="Documents are processed in isolated temporary sessions and never stored beyond the active session. No patient data is persisted to external databases. All processing runs on AWS infrastructure with encryption in transit and at rest."
            />
          </div>
        </div>
      </section>

      {/* ── Section divider ── */}
      <div className="max-w-7xl mx-auto px-6">
        <div className="border-t border-slate-200 dark:border-slate-800" />
      </div>

      {/* ── Pipeline walkthrough ── */}
      <section className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <FadeIn>
            <span className="text-[11px] font-black uppercase tracking-[0.2em] text-blue-600 dark:text-blue-500">
              How It Works
            </span>
          </FadeIn>
          <FadeIn delay={0.06} className="mt-3 mb-16">
            <h2 className="text-4xl sm:text-5xl font-black tracking-tighter text-slate-900 dark:text-white">
              Six agents. One report.
            </h2>
          </FadeIn>

          <div className="relative">
            {/* Vertical spine */}
            <div className="absolute left-[27px] top-7 bottom-7 w-px bg-slate-200 dark:bg-slate-800 hidden sm:block" />

            <div className="space-y-2">
              {[
                {
                  n: "01",
                  label: "Document Intelligence",
                  desc: "All uploaded files — PDF, PNG, JPG, WEBP — are converted to images and sent to Amazon Nova Pro for full OCR. Every value, code, name, and date is extracted as structured prose regardless of document type.",
                },
                {
                  n: "02",
                  label: "Policy Field Extraction",
                  desc: "Extracts the four required fields — insurer name, policy number, plan type, and member ID — from the combined document set. If any are missing the pipeline pauses and asks for an insurance card before proceeding.",
                },
                {
                  n: "03",
                  label: "Requirements Retrieval",
                  desc: "Looks up the insurer's specific pre-authorization requirements for the requested procedure from a local policy knowledge base, or via LLM reasoning when not in the KB. Returns two lists: required documents and required clinical criteria.",
                },
                {
                  n: "04",
                  label: "Document Verification",
                  desc: "Checks every submitted document against the requirements list. Each item is marked satisfied, incomplete, or missing — shown as an informational checklist. The pipeline always continues regardless of result.",
                },
                {
                  n: "05",
                  label: "Clinical Eligibility",
                  desc: "Checks every clinical requirement (symptom duration, step therapy, specialist consult, lab values) against actual document content. Detects cross-document conflicts and applies the conservative rule — always using the lowest value for threshold comparisons.",
                },
                {
                  n: "06",
                  label: "Final Report",
                  desc: "Synthesizes all findings into a structured JSON report: patient info, insurance details, requested procedure, document status, clinical assessment, approval probability, and recommended next steps.",
                },
              ].map((step, i) => (
                <FadeIn key={step.n} delay={i * 0.05}>
                  <div className="flex gap-6 group">
                    <div className="shrink-0 h-14 w-14 rounded-full border-2 border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 flex items-center justify-center font-black text-[11px] text-slate-400 dark:text-slate-600 group-hover:border-blue-500 group-hover:text-blue-600 dark:group-hover:text-blue-500 transition-all z-10">
                      {step.n}
                    </div>
                    <div className="flex-1 pt-3.5 pb-8 border-b border-slate-100 dark:border-slate-900 last:border-0">
                      <div className="font-black text-slate-900 dark:text-slate-100 mb-1.5">
                        {step.label}
                      </div>
                      <div className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed max-w-2xl">
                        {step.desc}
                      </div>
                    </div>
                  </div>
                </FadeIn>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-24 px-6 bg-slate-900 dark:bg-slate-950 relative overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:48px_48px] opacity-50" />
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-blue-900/20 rounded-full blur-3xl pointer-events-none" />

        <div className="relative max-w-3xl mx-auto text-center">
          <FadeIn>
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-slate-700 bg-slate-800/60 mb-8">
              <CheckCircle size={12} className="text-emerald-400" />
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.18em]">
                No setup · No account · No waiting
              </span>
            </div>

            <h2 className="text-4xl sm:text-5xl font-black tracking-tighter text-white mb-5 leading-tight">
              Stop waiting.
              <br />
              Start authorizing.
            </h2>

            <p className="text-lg text-slate-400 mb-10 max-w-lg mx-auto leading-relaxed">
              Upload patient documents now and receive a complete prior-authorization
              assessment in under two minutes.
            </p>

            <Link
              href="/analyse"
              className="group inline-flex items-center gap-3 px-10 py-5 bg-blue-600 hover:bg-blue-500 text-white font-black text-base rounded-full transition-colors shadow-2xl shadow-blue-600/25"
            >
              Analyse Documents Now
              <ArrowRight
                size={17}
                className="group-hover:translate-x-1 transition-transform"
              />
            </Link>

            <div className="mt-8 flex items-center justify-center gap-8 text-xs text-slate-600">
              <span className="flex items-center gap-1.5">
                <ShieldCheck size={12} className="text-slate-500" /> HIPAA Compliant
              </span>
              <span className="flex items-center gap-1.5">
                <Clock size={12} className="text-slate-500" /> Under 2 minutes
              </span>
              <span className="flex items-center gap-1.5">
                <Lock size={12} className="text-slate-500" /> No data retained
              </span>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="py-8 px-6 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-blue-600" strokeWidth={2.5} />
            <span className="font-black text-sm text-slate-900 dark:text-white">
              InsuranceHelper
            </span>
          </div>
          <p className="text-[11px] text-slate-400 text-center leading-relaxed">
            Built with Amazon Nova · AWS Bedrock · Next.js · FastAPI
            <span className="mx-2">·</span>
            Statistics from AMA Prior Authorization Surveys 2022–2023
          </p>
          <Link
            href="/analyse"
            className="text-xs font-bold text-blue-600 hover:underline"
          >
            Start Analysis →
          </Link>
        </div>
      </footer>
    </div>
  );
}
