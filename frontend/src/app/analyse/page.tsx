"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  ArrowLeft,
  CheckCircle2,
  AlertCircle,
  FileText,
  FileCheck,
  Bot,
  User,
  Loader2,
  ChevronRight,
  ShieldCheck,
  X,
  Eye,
  ExternalLink,
  Image as ImageIcon,
  Plus,
  ClipboardList,
} from "lucide-react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import * as api from "@/lib/api";

type Message = {
  id: string;
  role: "agent" | "user";
  type: "text" | "document_request" | "result" | "upload";
  content: string;
  agentName?: string;
  data?: any;
  loading?: boolean;
};

// ─── Pipeline step definitions ────────────────────────────────────────────────
const STEPS = [
  { id: 1, label: "OCR & Extraction",    agent: "Agent 1" },
  { id: 2, label: "Policy Fields",       agent: "Agent 2" },
  { id: 3, label: "Policy Requirements", agent: "Agent 3" },
  { id: 4, label: "Doc Verification",    agent: "Agent 4" },
  { id: 5, label: "Eligibility",         agent: "Agent 5" },
  { id: 6, label: "Final Report",        agent: "Agent 6" },
];

export default function AnalysePage() {
  const [messages, setMessages]           = useState<Message[]>([]);
  const [isAnalysing, setIsAnalysing]     = useState(false);
  const [hasUploaded, setHasUploaded]     = useState(false);   // lock after first upload
  const [needsReUpload, setNeedsReUpload] = useState(false);   // only for agent2 missing policy
  const [currentStep, setCurrentStep]     = useState(0);       // 0 = idle
  const [uploadedDocs, setUploadedDocs]   = useState<{ name: string; url: string; type: string }[]>([]);
  const [viewerDoc, setViewerDoc]         = useState<{ name: string; url: string; type: string } | null>(null);
  const [finalReport, setFinalReport]     = useState<any>(null);

  // Accumulates ALL Agent 1 document objects across every upload (initial + re-upload).
  // Using a ref so async pipeline functions always read the latest value without
  // stale closure issues.
  const allDocumentsRef = useRef<any[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef   = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Message helpers ──────────────────────────────────────────────────────
  const addMessage = (msg: Omit<Message, "id">) => {
    const id = Math.random().toString(36).substring(7);
    setMessages((prev) => [...prev, { ...msg, id }]);
    return id;
  };

  const updateMessage = (id: string, updates: Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...updates } : m)));
  };

  // ── File upload handler ───────────────────────────────────────────────────
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    // Reset input so same files can be re-selected if needed
    e.target.value = "";

    if (!hasUploaded) {
      // ── FIRST upload: run full pipeline ───────────────────────────────────
      setHasUploaded(true);
      setIsAnalysing(true);
      setCurrentStep(1);

      addMessage({
        role: "user",
        type: "upload",
        content: `Uploaded ${files.length} document${files.length > 1 ? "s" : ""}`,
        data: { files: files.map((f) => f.name) },
      });

      // Add to gallery
      setUploadedDocs(
        files.map((f) => ({ name: f.name, url: URL.createObjectURL(f), type: f.type }))
      );

      try {
        const { files: savedPaths } = await api.uploadFiles(files);
        await runFullPipeline(savedPaths);
      } catch (err: any) {
        addMessage({ role: "agent", type: "text", content: `Error: ${err.message}`, agentName: "System" });
      } finally {
        setIsAnalysing(false);
        setCurrentStep(0);
      }
    } else if (needsReUpload) {
      // ── RE-UPLOAD: only allowed when agent2 couldn't find policy info ──────
      setNeedsReUpload(false);
      setIsAnalysing(true);

      addMessage({
        role: "user",
        type: "upload",
        content: `Uploaded ${files.length} additional document${files.length > 1 ? "s" : ""} (insurance info)`,
        data: { files: files.map((f) => f.name) },
      });

      setUploadedDocs((prev) => [
        ...prev,
        ...files.map((f) => ({ name: f.name, url: URL.createObjectURL(f), type: f.type })),
      ]);

      try {
        const { files: newPaths } = await api.uploadFiles(files);

        // Run Agent 1 on the newly uploaded files only
        const loaderId = addMessage({
          role: "agent", type: "text", content: "Processing new documents...", loading: true, agentName: "Agent 1",
        });
        const a1New = await api.runAgent1(newPaths);

        // MERGE: append new Agent 1 documents into the accumulated list
        // so Agent 2 sees ALL documents from every upload combined
        allDocumentsRef.current = [
          ...allDocumentsRef.current,
          ...(a1New.documents ?? []),
        ];

        updateMessage(loaderId, {
          content: `New document(s) processed. Total documents in context: ${allDocumentsRef.current.length}.`,
          loading: false,
          agentName: "Agent 1",
          data: { documents: allDocumentsRef.current },
        });

        // Resume from Agent 2 with the full merged document set
        await runFromAgent2();
      } catch (err: any) {
        addMessage({ role: "agent", type: "text", content: `Error: ${err.message}`, agentName: "System" });
      } finally {
        setIsAnalysing(false);
        setCurrentStep(0);
      }
    }
  };

  // ── Full pipeline: Agent 1 → 6 ────────────────────────────────────────────
  const runFullPipeline = async (savedPaths: string[]) => {
    // Agent 1
    setCurrentStep(1);
    const a1LoaderId = addMessage({
      role: "agent", type: "text", content: "Extracting document content via OCR...", loading: true, agentName: "Agent 1",
    });
    const a1 = await api.runAgent1(savedPaths);

    // Store this batch as the canonical document list
    allDocumentsRef.current = a1.documents ?? [];

    updateMessage(a1LoaderId, {
      content: `Identified ${allDocumentsRef.current.length} document(s).`,
      data: a1, loading: false, agentName: "Agent 1",
    });

    await runFromAgent2();
  };

  // ── Pipeline from Agent 2 onward — always reads from allDocumentsRef ────────
  const runFromAgent2 = async () => {
    // Always build the Agent 1 shaped input from the accumulated document list.
    // This means Agent 2 sees every document from every upload combined.
    const a1Input = { documents: allDocumentsRef.current };

    // ── Agent 2 ─────────────────────────────────────────────────────────────
    setCurrentStep(2);
    const a2LoaderId = addMessage({
      role: "agent", type: "text", content: "Extracting policy & patient fields...", loading: true, agentName: "Agent 2",
    });
    const a2 = await api.runAgent2(a1Input);
    updateMessage(a2LoaderId, {
      content: a2.ready
        ? `Policy identified: ${a2.policy_search_fields?.insurer_name ?? "Unknown insurer"}`
        : "Some policy fields are missing — pipeline paused.",
      data: a2, loading: false, agentName: "Agent 2",
    });

    // If agent2 is not ready, give user ONE chance to upload insurance card
    if (!a2.ready) {
      addMessage({
        role: "agent",
        type: "document_request",
        content:
          "Insurance policy information could not be fully identified from the submitted documents. " +
          "Please upload your insurance card. The pipeline will resume with all previously " +
          "submitted documents plus the new one.",
        data: { missing: a2.missing_critical, isAgent2: true },
        agentName: "Agent 2",
      });
      setNeedsReUpload(true);
      return; // pause — handleFileUpload will resume from Agent 2 with merged docs
    }

    await runFromAgent3(a2);
  };

  // ── Agent 3 → 6 (always runs straight through, no interruptions) ─────────
  const runFromAgent3 = async (a2: any) => {
    // ── Agent 3 ─────────────────────────────────────────────────────────────
    setCurrentStep(3);
    const a3LoaderId = addMessage({
      role: "agent", type: "text", content: "Retrieving policy requirements...", loading: true, agentName: "Agent 3",
    });
    const a3 = await api.runAgent3(a2);
    updateMessage(a3LoaderId, {
      content: `Requirements loaded for: ${a3.procedure_identified ?? "requested procedure"}`,
      data: a3, loading: false, agentName: "Agent 3",
    });

    // ── Agent 4 ─────────────────────────────────────────────────────────────
    setCurrentStep(4);
    const a4LoaderId = addMessage({
      role: "agent", type: "text", content: "Verifying submitted documents against requirements...", loading: true, agentName: "Agent 4",
    });
    const a4 = await api.runAgent4(a3);
    // Show results but NEVER block — always continue
    const satisfied  = a4.satisfied?.length ?? 0;
    const missing    = a4.missing_documents?.length ?? 0;
    const partial    = a4.partial_documents?.length ?? 0;
    updateMessage(a4LoaderId, {
      content: `Verification complete: ${satisfied} satisfied, ${missing} missing, ${partial} incomplete.`,
      data: a4, loading: false, agentName: "Agent 4",
    });

    // ── Agent 5 ─────────────────────────────────────────────────────────────
    setCurrentStep(5);
    const a5LoaderId = addMessage({
      role: "agent", type: "text", content: "Performing clinical eligibility analysis...", loading: true, agentName: "Agent 5",
    });
    const a5 = await api.runAgent5(a4);
    updateMessage(a5LoaderId, {
      content: `Eligibility assessed. Determination: ${a5.determination ?? "PENDING_REVIEW"}`,
      data: { ...a5, agent: 5 }, loading: false, type: "result", agentName: "Agent 5",
    });

    // ── Agent 6 ─────────────────────────────────────────────────────────────
    setCurrentStep(6);
    const a6LoaderId = addMessage({
      role: "agent", type: "text", content: "Generating final pre-authorization report...", loading: true, agentName: "Agent 6",
    });
    const a6 = await api.runAgent6(a5);
    setFinalReport(a6);
    updateMessage(a6LoaderId, {
      content: "Final report generated.",
      data: { ...a6, agent: 6 }, loading: false, type: "result", agentName: "Agent 6",
    });
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-zinc-50 dark:bg-zinc-950">

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md z-10">
        <div className="flex items-center gap-4">
          <Link href="/" className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full transition-colors">
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div>
            <h1 className="text-xl font-bold dark:text-white flex items-center gap-2">
              <ShieldCheck className="text-blue-600 h-6 w-6" />
              InsuranceHelper
              <span className="text-xs font-normal text-zinc-400">Analysis Session</span>
            </h1>
          </div>
        </div>

        {/* Step progress strip */}
        <div className="hidden md:flex items-center gap-1">
          {STEPS.map((step) => {
            const done    = currentStep > step.id;
            const active  = currentStep === step.id;
            const waiting = currentStep < step.id;
            return (
              <div key={step.id} className="flex items-center gap-1">
                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold transition-all ${
                  done    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400" :
                  active  ? "bg-blue-600 text-white animate-pulse" :
                            "bg-zinc-100 text-zinc-400 dark:bg-zinc-800"
                }`}>
                  {done ? <CheckCircle2 size={11} /> : active ? <Loader2 size={11} className="animate-spin" /> : <span>{step.id}</span>}
                  {step.label}
                </div>
                {step.id < STEPS.length && <ChevronRight size={12} className="text-zinc-300 dark:text-zinc-700" />}
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-3">
          {isAnalysing && <Loader2 className="h-4 w-4 animate-spin text-blue-600" />}
          <div className={`h-2 w-2 rounded-full ${isAnalysing ? "bg-blue-500 animate-pulse" : "bg-green-500"}`} />
          <span className="text-sm font-medium text-zinc-500">{isAnalysing ? "Running" : "Ready"}</span>
        </div>
      </header>

      {/* Chat */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6">

        {/* Empty state */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-6 max-w-lg mx-auto">
            <div className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-3xl">
              <Upload className="h-12 w-12 text-blue-600 mx-auto" />
            </div>
            <div>
              <h2 className="text-2xl font-bold dark:text-white">Upload Your Documents</h2>
              <p className="text-zinc-500 mt-2">
                Upload all your medical documents at once — clinical notes, lab reports, insurance card,
                cost estimates — and the pipeline will run end-to-end automatically, producing a final
                pre-authorization report.
              </p>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalysing}
              className="flex items-center gap-2 px-8 py-4 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 rounded-full font-bold shadow-xl hover:scale-105 transition-transform disabled:opacity-50"
            >
              <Plus className="h-5 w-5" /> Upload All Documents
            </button>
            <p className="text-xs text-zinc-400">Supports PDF, PNG, JPG, WEBP</p>
          </div>
        )}

        <AnimatePresence>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className={`max-w-[85%] sm:max-w-2xl flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
                <div className={`flex-shrink-0 h-10 w-10 rounded-full flex items-center justify-center ${msg.role === "user" ? "bg-zinc-200 dark:bg-zinc-800" : "bg-blue-600"}`}>
                  {msg.role === "user" ? <User size={20} /> : <Bot size={20} className="text-white" />}
                </div>

                <div className={`space-y-2 ${msg.role === "user" ? "text-right" : ""}`}>
                  {msg.agentName && (
                    <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">{msg.agentName}</div>
                  )}

                  <div className={`p-4 rounded-2xl shadow-sm border ${
                    msg.role === "user"
                      ? "bg-zinc-900 text-white dark:bg-white dark:text-zinc-900 border-zinc-900/10"
                      : "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 text-zinc-800 dark:text-zinc-100"
                  }`}>
                    {msg.loading ? (
                      <div className="flex items-center gap-3">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>{msg.content}</span>
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
                    )}

                    {/* ── Upload list ── */}
                    {msg.type === "upload" && msg.data?.files && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {msg.data.files.map((f: string, i: number) => (
                          <span key={i} className="flex items-center gap-1 px-2 py-1 bg-white/10 dark:bg-black/20 rounded text-xs font-mono">
                            <FileText size={10} /> {f}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* ── Agent 1 doc chips ── */}
                    {msg.agentName?.includes("Agent 1") && msg.data?.documents && !msg.loading && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {msg.data.documents.map((d: any, i: number) => (
                          <div key={i} className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg text-xs font-bold border border-blue-100 dark:border-blue-900/30">
                            <FileText size={12} /> {d.document_type}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* ── Agent 2 policy summary ── */}
                    {msg.agentName?.includes("Agent 2") && msg.data?.policy_search_fields && !msg.loading && (
                      <div className="mt-4 p-4 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-100 dark:border-zinc-800 space-y-3">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-[10px] font-bold text-zinc-400 uppercase">Insurer</div>
                            <div className="text-sm font-bold">{msg.data.policy_search_fields.insurer_name ?? "—"}</div>
                          </div>
                          <div>
                            <div className="text-[10px] font-bold text-zinc-400 uppercase">Plan</div>
                            <div className="text-sm font-bold">{msg.data.policy_search_fields.plan_type ?? "—"}</div>
                          </div>
                          <div>
                            <div className="text-[10px] font-bold text-zinc-400 uppercase">Policy #</div>
                            <div className="text-sm font-bold">{msg.data.policy_search_fields.policy_number ?? "—"}</div>
                          </div>
                          <div>
                            <div className="text-[10px] font-bold text-zinc-400 uppercase">Member ID</div>
                            <div className="text-sm font-bold">{msg.data.policy_search_fields.member_id ?? "—"}</div>
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] font-bold text-zinc-400 uppercase">Procedure Requested</div>
                          <div className="text-sm font-medium text-blue-600 dark:text-blue-400">
                            {msg.data.policy_search_fields.procedure ?? "—"}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* ── Agent 3 requirements ── */}
                    {msg.agentName?.includes("Agent 3") && msg.data && !msg.loading && (
                      <div className="mt-4 space-y-3">
                        {msg.data.document_requirements?.length > 0 && (
                          <div className="space-y-2">
                            <div className="text-[10px] font-bold text-zinc-400 uppercase flex items-center gap-1"><FileCheck size={12} className="text-blue-500" /> Required Documents</div>
                            {msg.data.document_requirements.map((r: any, i: number) => (
                              <div key={i} className="p-3 bg-white dark:bg-zinc-900 rounded-xl border border-zinc-100 dark:border-zinc-800 text-xs">
                                <div className="font-bold">{r.document_type}</div>
                                <div className="text-zinc-500 mt-0.5">{r.info_needed}</div>
                              </div>
                            ))}
                          </div>
                        )}
                        {msg.data.medical_requirements?.length > 0 && (
                          <div className="space-y-2">
                            <div className="text-[10px] font-bold text-zinc-400 uppercase flex items-center gap-1"><ShieldCheck size={12} className="text-emerald-500" /> Clinical Criteria</div>
                            {msg.data.medical_requirements.map((r: any, i: number) => (
                              <div key={i} className="p-3 bg-zinc-50 dark:bg-zinc-950 rounded-xl border border-zinc-100 dark:border-zinc-800 text-xs">
                                <div className="flex justify-between items-start">
                                  <span className="font-bold">{r.requirement}</span>
                                  <span className={`text-[9px] px-1.5 py-0.5 rounded uppercase font-bold ${r.importance === "required" ? "bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-400" : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800"}`}>
                                    {r.importance}
                                  </span>
                                </div>
                                <div className="text-zinc-500 mt-0.5">{r.description}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* ── Agent 4 verification checklist ── */}
                    {msg.agentName?.includes("Agent 4") && msg.data && !msg.loading && (
                      <div className="mt-4 space-y-2">
                        {msg.data.satisfied?.map((s: any, i: number) => (
                          <div key={i} className="flex items-center gap-3 p-3 bg-emerald-50/50 dark:bg-emerald-900/10 rounded-xl border border-emerald-100 dark:border-emerald-900/20 text-xs">
                            <CheckCircle2 size={14} className="text-emerald-500 shrink-0" />
                            <span className="font-bold text-emerald-800 dark:text-emerald-300">{s.document_type}</span>
                            <span className="text-emerald-600/60 text-[10px]">satisfied by {s.satisfied_by}</span>
                          </div>
                        ))}
                        {msg.data.partial_documents?.map((p: any, i: number) => (
                          <div key={i} className="flex items-start gap-3 p-3 bg-amber-50/50 dark:bg-amber-900/10 rounded-xl border border-amber-100 dark:border-amber-900/20 text-xs">
                            <AlertCircle size={14} className="text-amber-500 shrink-0 mt-0.5" />
                            <div>
                              <div className="font-bold text-amber-800 dark:text-amber-300">{p.document_type} (incomplete)</div>
                              {p.info_missing && <div className="text-amber-600/70 mt-0.5">Missing: {p.info_missing}</div>}
                            </div>
                          </div>
                        ))}
                        {msg.data.missing_documents?.map((m: any, i: number) => (
                          <div key={i} className="flex items-center gap-3 p-3 bg-red-50/50 dark:bg-red-900/10 rounded-xl border border-red-100 dark:border-red-900/20 text-xs">
                            <X size={14} className="text-red-400 shrink-0" />
                            <span className="font-bold text-red-700 dark:text-red-400">{m.document_type} (missing)</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* ── Agent 2 document request (only pipeline blocker) ── */}
                    {msg.type === "document_request" && msg.data?.isAgent2 && (
                      <div className="mt-4 space-y-3">

                        {/* Header */}
                        <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/10 rounded-xl border border-red-200 dark:border-red-900/30">
                          <AlertCircle className="h-5 w-5 text-red-500 shrink-0" />
                          <div>
                            <div className="text-sm font-bold text-red-700 dark:text-red-400">
                              Insurance Card Required
                            </div>
                            <div className="text-xs text-red-600/70 dark:text-red-400/60 mt-0.5">
                              The following {msg.data.missing?.length === 1 ? "field is" : `${msg.data.missing?.length} fields are`} missing from your submitted documents.
                              Please upload your insurance card to continue.
                            </div>
                          </div>
                        </div>

                        {/* Per-field breakdown with reasons */}
                        <div className="space-y-2">
                          {msg.data.missing?.map((doc: any, i: number) => (
                            <div key={i} className="p-3 bg-white dark:bg-zinc-900 rounded-xl border border-red-100 dark:border-red-900/20 text-sm">
                              <div className="flex items-center gap-2 mb-1">
                                <div className="h-5 w-5 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center shrink-0">
                                  <span className="text-[10px] font-black text-red-600 dark:text-red-400">{i + 1}</span>
                                </div>
                                <div className="font-bold text-zinc-800 dark:text-zinc-100">
                                  {doc.info_needed}
                                </div>
                              </div>
                              <div className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed ml-7">
                                {doc.reason}
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Where to find this */}
                        <div className="px-3 py-2.5 bg-blue-50 dark:bg-blue-900/10 rounded-xl border border-blue-100 dark:border-blue-900/20 text-xs text-blue-700 dark:text-blue-400">
                          💡 You can find all of these on your <strong>insurance card</strong> or{" "}
                          <strong>member ID card</strong> issued by your insurer.
                        </div>

                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="w-full flex items-center justify-center gap-2 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 active:scale-95 transition-all"
                        >
                          <Upload size={16} /> Upload Insurance Card
                        </button>
                      </div>
                    )}

                    {/* ── Agent 5 result card ── */}
                    {msg.type === "result" && msg.data?.agent === 5 && (
                      <div className="mt-4 p-5 rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border border-blue-100 dark:border-blue-900/30">
                        <div className="flex items-center justify-between mb-4">
                          <div className="text-sm font-bold text-blue-900 dark:text-blue-300 uppercase tracking-widest">Eligibility Assessment</div>
                          <div className={`text-2xl font-black ${
                            msg.data.approval_probability > 0.7 ? "text-emerald-500" :
                            msg.data.approval_probability > 0.4 ? "text-amber-500" : "text-red-500"
                          }`}>
                            {((msg.data.approval_probability ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="space-y-3">
                          {msg.data.clinical_summary && (
                            <div className="p-3 bg-white/50 dark:bg-white/5 rounded-xl border border-white dark:border-white/10">
                              <div className="text-xs text-zinc-400 mb-1 uppercase tracking-widest font-bold">Clinical Summary</div>
                              <div className="text-sm text-zinc-700 dark:text-zinc-300 italic">"{msg.data.clinical_summary}"</div>
                            </div>
                          )}
                          <div className="space-y-1.5">
                            {msg.data.requirements_checked?.map((r: any, i: number) => (
                              <div key={i} className="flex flex-col gap-0.5 p-2 rounded-lg bg-white/30 dark:bg-black/20 border border-black/5 dark:border-white/5">
                                <div className="flex items-center gap-2 text-xs font-bold">
                                  {r.status === "met" ? <CheckCircle2 size={13} className="text-emerald-500" /> : <AlertCircle size={13} className="text-amber-500" />}
                                  <span className={r.status === "met" ? "text-emerald-700 dark:text-emerald-400" : "text-amber-700 dark:text-amber-400"}>{r.requirement}</span>
                                </div>
                                {r.evidence && <div className="text-[10px] text-zinc-500 ml-5">{r.evidence}</div>}
                              </div>
                            ))}
                          </div>
                          <div className={`text-sm font-bold p-3 rounded-xl border ${
                            ["APPROVED","LIKELY_APPROVED"].includes(msg.data.determination)
                              ? "bg-emerald-50 text-emerald-700 border-emerald-100 dark:bg-emerald-950/20 dark:border-emerald-900/30"
                              : "bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-950/20 dark:border-amber-900/30"
                          }`}>
                            {msg.data.recommendation}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* ── Agent 6 final report card ── */}
                    {msg.type === "result" && msg.data?.agent === 6 && (
                      <FinalReportCard report={msg.data} />
                    )}

                    {/* ── Technical details expander ── */}
                    {msg.data && !msg.loading && msg.type !== "document_request" && (
                      <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800">
                        <details className="group">
                          <summary className="flex items-center gap-2 text-[10px] font-bold text-zinc-400 uppercase tracking-widest cursor-pointer hover:text-blue-500 transition-colors list-none">
                            <ChevronRight size={11} className="group-open:rotate-90 transition-transform" />
                            Technical Details
                          </summary>
                          <pre className="mt-2 p-3 bg-zinc-50 dark:bg-zinc-950 rounded-xl text-[10px] font-mono text-zinc-500 overflow-auto max-h-64">
                            {JSON.stringify(msg.data, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={chatEndRef} />
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5 z-10">
        <div className="max-w-6xl mx-auto space-y-4">

          {/* Document gallery */}
          {uploadedDocs.length > 0 && (
            <div className="flex gap-3 overflow-x-auto pb-1">
              {uploadedDocs.map((doc, i) => (
                <motion.div
                  key={i}
                  whileHover={{ scale: 1.05 }}
                  onClick={() => setViewerDoc(doc)}
                  className="flex-shrink-0 w-28 h-18 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800 overflow-hidden cursor-pointer relative group"
                >
                  {doc.type.includes("image") ? (
                    <img src={doc.url} alt={doc.name} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity" />
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center p-2">
                      <FileText size={18} className="text-zinc-400" />
                      <span className="text-[8px] text-zinc-500 truncate w-full text-center mt-1 font-mono">{doc.name}</span>
                    </div>
                  )}
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <Eye size={14} className="text-white" />
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-zinc-500">
              {isAnalysing ? (
                <><Loader2 size={16} className="animate-spin text-blue-600" /> Running pipeline…</>
              ) : finalReport ? (
                <><CheckCircle2 size={16} className="text-emerald-500" /> Pipeline complete — report ready</>
              ) : needsReUpload ? (
                <><AlertCircle size={16} className="text-amber-500" /> Waiting for insurance card upload</>
              ) : hasUploaded ? (
                <><ShieldCheck size={16} className="text-zinc-400" /> Analysis finished</>
              ) : (
                <><ShieldCheck size={16} className="text-emerald-500" /> Ready — upload documents to begin</>
              )}
            </div>

            <div className="flex gap-2">
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} multiple className="hidden" accept=".pdf,.png,.jpg,.jpeg,.webp" />

              {/* Show upload button only if: not yet uploaded, OR waiting for re-upload (agent2) */}
              {(!hasUploaded || needsReUpload) && (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isAnalysing}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-2xl font-bold shadow-lg shadow-blue-500/20 hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:scale-100"
                >
                  <Upload size={18} />
                  {!hasUploaded ? "Upload Documents" : "Upload Insurance Card"}
                </button>
              )}

              {/* Download report button once agent6 is done */}
              {finalReport && (
                <button
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(finalReport, null, 2)], { type: "application/json" });
                    const url  = URL.createObjectURL(blob);
                    const a    = document.createElement("a");
                    a.href = url; a.download = "preauth_report.json"; a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="flex items-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-2xl font-bold hover:bg-emerald-700 transition-colors"
                >
                  <ClipboardList size={18} /> Download Report
                </button>
              )}
            </div>
          </div>

          <p className="text-center text-[10px] text-zinc-400 uppercase tracking-[0.2em]">
            HIPAA Compliant · Amazon Nova · AI Medical Pipeline
          </p>
        </div>
      </footer>

      {/* Document viewer modal */}
      <AnimatePresence>
        {viewerDoc && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-xl bg-black/60"
            onClick={() => setViewerDoc(null)}
          >
            <motion.div
              initial={{ scale: 0.92, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.92, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              className="relative w-full max-w-4xl max-h-[90vh] bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl overflow-hidden flex flex-col"
            >
              <div className="flex items-center justify-between p-4 border-b border-zinc-100 dark:border-zinc-800">
                <div className="flex items-center gap-3 px-2">
                  {viewerDoc.type.includes("image") ? <ImageIcon size={18} className="text-blue-600" /> : <FileText size={18} className="text-blue-600" />}
                  <span className="font-bold text-sm truncate max-w-sm">{viewerDoc.name}</span>
                </div>
                <div className="flex gap-2">
                  <a href={viewerDoc.url} target="_blank" rel="noopener noreferrer" className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg text-zinc-500">
                    <ExternalLink size={18} />
                  </a>
                  <button onClick={() => setViewerDoc(null)} className="p-2 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg text-zinc-500 hover:text-red-600">
                    <X size={18} />
                  </button>
                </div>
              </div>
              <div className="flex-1 bg-zinc-50 dark:bg-zinc-950 overflow-auto p-4 flex items-center justify-center">
                {viewerDoc.type.includes("image") ? (
                  <img src={viewerDoc.url} alt={viewerDoc.name} className="max-w-full h-auto rounded-lg shadow-lg" />
                ) : (
                  <iframe src={`${viewerDoc.url}#toolbar=0`} className="w-full h-full min-h-[60vh] rounded-lg border-0" title={viewerDoc.name} />
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Final Report Card Component ──────────────────────────────────────────────
function FinalReportCard({ report }: { report: any }) {
  const det = report.approval_assessment?.determination ?? "PENDING_REVIEW";
  const prob = report.approval_assessment?.probability ?? 0;
  const isApproved = ["APPROVED", "LIKELY_APPROVED"].includes(det);

  return (
    <div className="mt-4 rounded-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800">
      {/* Header band */}
      <div className={`px-5 py-4 flex items-center justify-between ${isApproved ? "bg-emerald-600" : "bg-amber-500"}`}>
        <div>
          <div className="text-white/70 text-xs font-bold uppercase tracking-widest">Pre-Authorization Report</div>
          <div className="text-white text-lg font-black">{det.replace(/_/g, " ")}</div>
        </div>
        <div className="text-white text-3xl font-black">{(prob * 100).toFixed(0)}%</div>
      </div>

      <div className="p-5 space-y-4 bg-white dark:bg-zinc-900">
        {/* Patient + Insurance */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <div className="text-[10px] font-bold text-zinc-400 uppercase">Patient</div>
            <div className="text-sm font-bold">{report.patient_info?.name ?? "—"}</div>
            <div className="text-xs text-zinc-500">DOB: {report.patient_info?.dob ?? "—"}</div>
          </div>
          <div className="space-y-1">
            <div className="text-[10px] font-bold text-zinc-400 uppercase">Insurance</div>
            <div className="text-sm font-bold">{report.insurance_info?.insurer ?? "—"}</div>
            <div className="text-xs text-zinc-500">Policy: {report.insurance_info?.policy_number ?? "—"}</div>
          </div>
        </div>

        {/* Procedure */}
        <div className="p-3 bg-zinc-50 dark:bg-zinc-950 rounded-xl border border-zinc-100 dark:border-zinc-800">
          <div className="text-[10px] font-bold text-zinc-400 uppercase mb-1">Requested Procedure</div>
          <div className="text-sm font-bold text-blue-600 dark:text-blue-400">{report.requested_procedure?.name ?? "—"}</div>
          <div className="text-xs text-zinc-500 mt-0.5">
            CPT: {report.requested_procedure?.cpt_codes?.join(", ") || "—"} · ICD-10: {report.requested_procedure?.icd10_codes?.join(", ") || "—"}
          </div>
        </div>

        {/* Medical requirements */}
        {report.medical_requirements_check?.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-bold text-zinc-400 uppercase">Clinical Requirements</div>
            {report.medical_requirements_check.map((r: any, i: number) => (
              <div key={i} className={`flex items-start gap-2 p-2 rounded-lg text-xs border ${
                r.status === "met"
                  ? "bg-emerald-50 border-emerald-100 dark:bg-emerald-900/10 dark:border-emerald-900/20"
                  : "bg-amber-50 border-amber-100 dark:bg-amber-900/10 dark:border-amber-900/20"
              }`}>
                {r.status === "met"
                  ? <CheckCircle2 size={13} className="text-emerald-500 shrink-0 mt-0.5" />
                  : <AlertCircle size={13} className="text-amber-500 shrink-0 mt-0.5" />}
                <div>
                  <span className="font-bold">{r.requirement}</span>
                  {r.evidence && <div className="text-zinc-500 mt-0.5">{r.evidence}</div>}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Next steps */}
        {report.next_steps?.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-bold text-zinc-400 uppercase">Next Steps</div>
            {report.next_steps.map((s: string, i: number) => (
              <div key={i} className="flex items-start gap-2 text-xs text-zinc-600 dark:text-zinc-400">
                <ChevronRight size={12} className="text-blue-500 shrink-0 mt-0.5" />
                {s}
              </div>
            ))}
          </div>
        )}

        {/* Recommendation */}
        {report.approval_assessment?.recommendation && (
          <div className={`text-sm font-bold p-3 rounded-xl border ${
            isApproved
              ? "bg-emerald-50 text-emerald-700 border-emerald-100 dark:bg-emerald-950/20 dark:border-emerald-900/30"
              : "bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-950/20 dark:border-amber-900/30"
          }`}>
            {report.approval_assessment.recommendation}
          </div>
        )}
      </div>
    </div>
  );
}
