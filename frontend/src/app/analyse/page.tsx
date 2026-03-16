"use client";

import React, { useState, useRef, useEffect } from "react";
import { 
  Upload, 
  Plus, 
  Send, 
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
  Image as ImageIcon
} from "lucide-react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import * as api from "@/lib/api";

type Message = {
  id: string;
  role: 'agent' | 'user';
  type: 'text' | 'document_request' | 'result' | 'upload';
  content: string;
  agentName?: string;
  data?: any;
  loading?: boolean;
};

export default function AnalysePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAnalysing, setIsAnalysing] = useState(false);
  const [sessionFiles, setSessionFiles] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [allDocData, setAllDocData] = useState<any>(null); // Agent 1 merged results
  const [currentStep, setCurrentStep] = useState<number>(0); // 0: idle, 1..5: agents
  const [lastAgentData, setLastAgentData] = useState<any>(null);
  const [uploadedDocs, setUploadedDocs] = useState<{name: string, url: string, type: string}[]>([]);
  const [viewerDoc, setViewerDoc] = useState<{name: string, url: string, type: string} | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (msg: Omit<Message, 'id'>) => {
    const id = Math.random().toString(36).substring(7);
    setMessages((prev) => [...prev, { ...msg, id }]);
    return id;
  };

  const updateMessage = (id: string, updates: Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...updates } : m)));
  };

  // Initial upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    if (currentStep === 0) {
      // Starting new analysis
      setIsAnalysing(true);
      addMessage({
        role: 'user',
        type: 'upload',
        content: `Uploaded ${files.length} document(s)`,
        data: { files: files.map(f => f.name) }
      });

      try {
        const { session_id, files: savedPaths } = await api.uploadFiles(files);
        setSessionId(session_id);
        setSessionFiles(savedPaths);
        
        // Add to gallery
        const newDocs = files.map(f => ({
          name: f.name,
          url: URL.createObjectURL(f),
          type: f.type
        }));
        setUploadedDocs(prev => [...prev, ...newDocs]);
        
        const loaderId = addMessage({
          role: 'agent',
          type: 'text',
          content: "Starting document analysis...",
          loading: true,
          agentName: "System"
        });

        // Step 1: Agent 1
        const a1Result = await api.runAgent1(savedPaths);
        setAllDocData(a1Result);
        updateMessage(loaderId, {
          content: "Documents processed and clinical summaries generated.",
          data: a1Result,
          loading: false,
          agentName: "Agent 1 (OCR)"
        });

        setLastAgentData(a1Result);
        setCurrentStep(1);
        runPipeline(1, a1Result);
      } catch (err: any) {
        addMessage({
          role: 'agent',
          type: 'text',
          content: "Error: " + err.message,
          agentName: "System"
        });
      } finally {
        setIsAnalysing(false);
      }
    } else {
      // Mid-pipeline upload (requested docs)
      const loaderId = addMessage({
        role: 'user',
        type: 'upload',
        content: `Submitting requested documents: ${files.length} file(s)`,
        data: { files: files.map(f => f.name) }
      });

      try {
        const { files: newPaths } = await api.uploadFiles(files);
        
        // Add to gallery
        const newDocs = files.map(f => ({
          name: f.name,
          url: URL.createObjectURL(f),
          type: f.type
        }));
        setUploadedDocs(prev => [...prev, ...newDocs]);

        const a1NewResult = await api.runAgent1(newPaths);
        
        // Merge with existing docs
        const updatedDocs = [...allDocData.documents, ...a1NewResult.documents];
        const mergedResult = { ...allDocData, documents: updatedDocs };
        setAllDocData(mergedResult);

        addMessage({
          role: 'agent',
          type: 'text',
          content: "New documents processed. Re-evaluating criteria...",
          agentName: "Agent 1"
        });

        // Go back to Agent 2 to refresh the whole context with new docs
        runPipeline(2, mergedResult);
      } catch (err: any) {
         addMessage({
          role: 'agent',
          type: 'text',
          content: "Error processing new documents: " + err.message,
          agentName: "System"
        });
      }
    }
  };

  const runPipeline = async (startStep: number, inputData: any) => {
    setIsAnalysing(true);
    let currentInput = inputData;

    try {
      // Step 2: Policy Checker
      if (startStep <= 2) {
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Extracting policy details...", loading: true, agentName: "Agent 2" });
        const res = await api.runAgent2(currentInput);
        setLastAgentData(res);

        if (!res.ready) {
           updateMessage(loaderId, { 
             content: "Critical policy information is missing.",
             data: res,
             loading: false 
           });
           addMessage({
             role: 'agent',
             type: 'document_request',
             content: "I'm unable to uniquely identify your insurance policy. Please upload an insurance card or a document containing the following missing details:",
             data: { 
               missing: res.missing_critical,
               isAgent2: true
             },
             agentName: "Agent 2"
           });
           setIsAnalysing(false);
           return;
        }

        updateMessage(loaderId, { 
          content: `Policy and Procedure details extracted.`, 
          data: res,
          loading: false 
        });
        currentInput = res;
      }

      // Step 3: Policy Retriever
      if (startStep <= 3) {
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Retrieving policy requirements...", loading: true, agentName: "Agent 3" });
        const res = await api.runAgent3(currentInput);
        updateMessage(loaderId, { 
          content: `Policy requirements retrieved for ${res.procedure_identified || 'requested procedure'}.`,
          data: res,
          loading: false 
        });
        currentInput = res;
        setLastAgentData(res);
      }

      // Step 4: Document Checker
      if (startStep <= 4) {
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Verifying document completeness...", loading: true, agentName: "Agent 4" });
        const res = await api.runAgent4(currentInput);
        setLastAgentData(res);
        
        if (!res.can_proceed) {
          updateMessage(loaderId, { 
            content: "Requirement verification incomplete.",
            data: res,
            loading: false 
          });
          
          addMessage({
            role: 'agent',
            type: 'document_request',
            content: "The insurance policy requires specific clinical evidence that isn't fully present in the current documents:",
            data: { 
              missing: res.missing_documents,
              partial: res.partial_documents
            },
            agentName: "Agent 4"
          });
          setIsAnalysing(false);
          return;
        } else {
          updateMessage(loaderId, { 
            content: "Clinical documentation verification complete.",
            data: res,
            loading: false 
          });
          currentInput = res;
        }
      }

      // Step 5: Eligibility Reasoning
      if (startStep <= 5) {
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Performing final clinical analysis...", loading: true, agentName: "Agent 5" });
        const res = await api.runAgent5(currentInput);
        console.log("[Pipeline] Agent 5 Success:", res);
        
        updateMessage(loaderId, { 
          type: 'result',
          content: `Clinical Analysis Complete. Determination: ${res.determination}`,
          data: { ...res, agent: 5 },
          loading: false,
          agentName: "Agent 5"
        });
        currentInput = res;
        setLastAgentData(res);
      }

      // Step 6: Form Filler
      if (startStep <= 6) {
        console.log("[Pipeline] Starting Agent 6 (Form Filler)...");
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Generating pre-authorization PDF form...", loading: true, agentName: "Agent 6" });
        const res = await api.runAgent6(currentInput);
        console.log("[Pipeline] Agent 6 Success:", res);
        
        updateMessage(loaderId, { 
          type: 'result',
          content: `Form filling complete. Generated ${res.fields_filled} fields across the document.`,
          data: { ...res, agent: 6 },
          loading: false,
          agentName: "Agent 6 (Final)"
        });
      }

    } catch (err: any) {
      console.error("[Pipeline] CRITICAL ERROR:", err);
      addMessage({ role: 'agent', type: 'text', content: "Error in pipeline: " + err.message, agentName: "System" });
    } finally {
      setIsAnalysing(false);
    }
  };

  const handleProceedManual = () => {
    if (lastAgentData) {
      addMessage({ role: 'user', type: 'text', content: "Proceed with current information." });
      // If we are skipping Agent 4's request, we go to Agent 5.
      // If we are skipping Agent 2's request, we also go to Agent 5 (since A3/A4 need A2's policy info).
      runPipeline(5, lastAgentData); 
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <Link href="/" className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full transition-colors">
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div>
            <h1 className="text-xl font-bold dark:text-white flex items-center gap-2">
              <ShieldCheck className="text-blue-600 h-6 w-6" />
              InsuranceHelper <span className="text-xs font-normal text-zinc-400">Analysis Session</span>
            </h1>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isAnalysing && <Loader2 className="h-5 w-5 animate-spin text-blue-600" />}
          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm font-medium text-zinc-500">Live API</span>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-6 max-w-lg mx-auto">
            <div className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-3xl">
              <Upload className="h-12 w-12 text-blue-600 mx-auto" />
            </div>
            <div>
              <h2 className="text-2xl font-bold dark:text-white">Start New Analysis</h2>
              <p className="text-zinc-500 mt-2">Upload your medical documents, doctor notes, and insurance cards to begin the automated pre-authorization check.</p>
            </div>
            <button 
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalysing}
              className="flex items-center gap-2 px-8 py-4 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 rounded-full font-bold shadow-xl hover:scale-105 transition-transform disabled:opacity-50"
            >
              <Plus className="h-5 w-5" />
              Upload Documents
            </button>
          </div>
        )}

        <AnimatePresence>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[85%] sm:max-w-2xl flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`flex-shrink-0 h-10 w-10 rounded-full flex items-center justify-center ${msg.role === 'user' ? 'bg-zinc-200 dark:bg-zinc-800' : 'bg-blue-600'}`}>
                  {msg.role === 'user' ? <User size={20} /> : <Bot size={20} className="text-white" />}
                </div>
                
                <div className={`space-y-2 ${msg.role === 'user' ? 'text-right' : ''}`}>
                  {msg.agentName && <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">{msg.agentName}</div>}
                  
                  <div className={`p-4 rounded-2xl shadow-sm border ${
                    msg.role === 'user' 
                      ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-900 border-zinc-900/10' 
                      : 'bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 text-zinc-800 dark:text-zinc-100'
                  }`}>
                    {msg.loading ? (
                      <div className="flex items-center gap-3">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>{msg.content}</span>
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
                    )}

                    {/* Agent 1 Primary View */}
                    {msg.agentName?.includes("Agent 1") && msg.data?.documents && !msg.loading && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {msg.data.documents.map((d: any, i: number) => (
                          <div key={i} className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg text-xs font-bold border border-blue-100 dark:border-blue-900/30">
                            <FileText size={12} />
                            {d.document_type}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Agent 2 Primary View */}
                    {msg.agentName?.includes("Agent 2") && msg.data?.policy_search_fields && !msg.loading && (
                      <div className="mt-4 p-4 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-100 dark:border-zinc-800 space-y-3">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-[10px] font-bold text-zinc-400 uppercase">Insurer</div>
                            <div className="text-sm font-bold truncate">{msg.data.policy_search_fields.insurer_name || 'Not Found'}</div>
                          </div>
                          <div>
                            <div className="text-[10px] font-bold text-zinc-400 uppercase">Policy #</div>
                            <div className="text-sm font-bold truncate">{msg.data.policy_search_fields.policy_number || 'Not Found'}</div>
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] font-bold text-zinc-400 uppercase">Procedure</div>
                          <div className="text-sm font-medium text-blue-600 dark:text-blue-400">{msg.data.policy_search_fields.procedure || 'Not Specified'}</div>
                        </div>
                      </div>
                    )}

                    {/* Agent 3 Primary View */}
                    {msg.agentName?.includes("Agent 3") && msg.data && !msg.loading && (
                      <div className="mt-4 space-y-4">
                        {msg.data.document_requirements?.length > 0 && (
                          <div className="space-y-2">
                            <div className="text-[10px] font-bold text-zinc-400 uppercase flex items-center gap-2">
                              <FileCheck size={14} className="text-blue-500" />
                              Required Documents
                            </div>
                            <div className="grid gap-2">
                              {msg.data.document_requirements.map((r: any, i: number) => (
                                <div key={i} className="p-3 bg-white dark:bg-zinc-900 rounded-xl border border-zinc-100 dark:border-zinc-800 text-xs">
                                  <div className="font-bold">{r.document_type}</div>
                                  <div className="text-zinc-500 mt-1 leading-relaxed">{r.info_needed}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {msg.data.medical_requirements?.length > 0 && (
                          <div className="space-y-2">
                            <div className="text-[10px] font-bold text-zinc-400 uppercase flex items-center gap-2">
                              <ShieldCheck size={14} className="text-emerald-500" />
                              Clinical Eligibility Criteria
                            </div>
                            <div className="grid gap-2">
                              {msg.data.medical_requirements.map((r: any, i: number) => (
                                <div key={i} className="p-3 bg-zinc-50 dark:bg-zinc-950 rounded-xl border border-zinc-100 dark:border-zinc-800 text-xs">
                                  <div className="flex justify-between items-start gap-4">
                                    <span className="font-bold">{r.requirement}</span>
                                    <span className={`text-[9px] px-1.5 py-0.5 rounded uppercase font-bold shrink-0 ${r.importance === 'required' ? 'bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-400' : 'bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400'}`}>
                                      {r.importance}
                                    </span>
                                  </div>
                                  <div className="text-zinc-500 mt-1 leading-relaxed">{r.description}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Agent 4 Primary View */}
                    {msg.agentName?.includes("Agent 4") && msg.data && !msg.loading && (
                      <div className="mt-4 space-y-3">
                        <div className="text-[10px] font-bold text-zinc-400 uppercase">Verification Checklist</div>
                        <div className="space-y-2">
                          {msg.data.satisfied?.map((s: any, i: number) => (
                            <div key={i} className="flex items-center gap-3 p-3 bg-emerald-50/50 dark:bg-emerald-900/10 rounded-xl border border-emerald-100 dark:border-emerald-900/20 text-xs">
                              <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
                              <div className="font-bold text-emerald-800 dark:text-emerald-300">{s.document_type} Verified</div>
                            </div>
                          ))}
                          {(msg.data.partial_documents || msg.data.partial)?.map((p: any, i: number) => (
                            <div key={i} className="flex items-center gap-3 p-3 bg-amber-50/50 dark:bg-amber-900/10 rounded-xl border border-amber-100 dark:border-amber-900/20 text-xs">
                              <AlertCircle size={16} className="text-amber-500 shrink-0" />
                              <div>
                                <div className="font-bold text-amber-800 dark:text-amber-300">{p.document_type} Incomplete</div>
                                <div className="text-[10px] text-amber-600/70 mt-0.5">Missing: {p.info_missing}</div>
                              </div>
                            </div>
                          ))}
                          {(msg.data.missing_documents || msg.data.missing)?.map((m: any, i: number) => (
                            <div key={i} className="flex items-center gap-3 p-3 bg-red-50/50 dark:bg-red-900/10 rounded-xl border border-red-100 dark:border-red-900/20 text-xs">
                              <X size={16} className="text-red-500 shrink-0" />
                              <div className="font-bold text-red-800 dark:text-red-300">{m.document_type} Missing</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Document Request Display (Agent 2/4 blockers) */}
                    {msg.type === 'document_request' && msg.data && (
                      <div className="mt-4 space-y-3">
                        <div className="grid gap-2">
                          {msg.data.missing?.map((doc: any, i: number) => (
                            <div key={i} className="flex items-start gap-3 p-3 bg-red-50 dark:bg-red-900/10 rounded-xl border border-red-100 dark:border-red-900/20 text-sm">
                              <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
                              <div className="text-left">
                                <div className="font-bold text-red-700 dark:text-red-400">{doc.document_type}</div>
                                <div className="text-red-600/80 dark:text-red-300/60 mt-0.5">{doc.info_needed}</div>
                              </div>
                            </div>
                          ))}
                          {msg.data.partial?.map((doc: any, i: number) => (
                            <div key={i} className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-900/10 rounded-xl border border-amber-100 dark:border-amber-900/20 text-sm">
                              <AlertCircle className="h-5 w-5 text-amber-500 mt-0.5" />
                              <div className="text-left">
                                <div className="font-bold text-amber-700 dark:text-amber-400">INCOMPLETE: {doc.document_type}</div>
                                <div className="text-amber-600/80 dark:text-amber-300/60 mt-0.5">Missing: {doc.info_missing}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="flex gap-2">
                           <button 
                             onClick={() => fileInputRef.current?.click()}
                             className="flex-1 flex items-center justify-center gap-2 py-3 bg-blue-600 text-white rounded-xl font-bold shadow-lg hover:shadow-blue-500/20 hover:scale-[1.02] transition-all"
                           >
                             <Upload size={18} /> Upload Documents
                           </button>
                           {!msg.data?.isAgent2 && (
                             <button 
                               onClick={handleProceedManual}
                               className="flex-1 py-3 bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 rounded-xl font-bold hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                             >
                               Proceed anyway
                             </button>
                           )}
                        </div>
                      </div>
                    )}

                    {/* Result Card Display - Agent 5 */}
                    {msg.type === 'result' && msg.data?.agent === 5 && (
                      <div className="mt-4 p-5 rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border border-blue-100 dark:border-blue-900/30">
                        <div className="flex items-center justify-between mb-4">
                           <div className="text-sm font-bold text-blue-900 dark:text-blue-300 uppercase tracking-widest">Eligibility Assessment</div>
                           <div className={`text-2xl font-black ${
                               msg.data.approval_probability > 0.7 ? 'text-emerald-500' : 
                               msg.data.approval_probability > 0.4 ? 'text-amber-500' : 'text-red-500'
                           }`}>
                             {(msg.data.approval_probability * 100).toFixed(0)}%
                           </div>
                        </div>
                        <div className="space-y-4">
                           <div className="p-3 bg-white/50 dark:bg-white/5 rounded-xl border border-white dark:border-white/10">
                              <div className="text-xs text-zinc-400 mb-1 uppercase tracking-widest font-bold">Clinical Summary</div>
                              <div className="text-sm text-zinc-700 dark:text-zinc-300 italic">"{msg.data.clinical_summary}"</div>
                           </div>
                           
                           <div className="space-y-2">
                              {msg.data.requirements_checked?.map((r: any, i: number) => (
                                <div key={i} className="flex flex-col gap-1 p-2 rounded-lg bg-white/30 dark:bg-black/20 border border-black/5 dark:border-white/5">
                                  <div className="flex items-center gap-2 text-xs font-bold">
                                    {r.status === 'met' ? <CheckCircle2 size={14} className="text-emerald-500" /> : <AlertCircle size={14} className="text-amber-500" />}
                                    <span className={r.status === 'met' ? 'text-emerald-700 dark:text-emerald-400' : 'text-amber-700 dark:text-amber-400'}>{r.requirement}</span>
                                  </div>
                                  <div className="text-[10px] text-zinc-500 ml-5 leading-tight">{r.evidence}</div>
                                </div>
                              ))}
                           </div>

                           <div className="pt-2">
                              <div className="text-xs font-bold text-zinc-400 mb-2 uppercase tracking-wide">Final Recommendation:</div>
                              <div className={`text-sm font-bold p-3 rounded-xl border ${
                                msg.data.determination === 'APPROVED' || msg.data.determination === 'LIKELY_APPROVED' 
                                  ? 'bg-emerald-50 text-emerald-700 border-emerald-100 dark:bg-emerald-950/20 dark:border-emerald-900/30' :
                                'bg-amber-50 text-amber-700 border-amber-100 dark:bg-amber-950/20 dark:border-amber-900/30'
                              }`}>
                                {msg.data.recommendation}
                              </div>
                           </div>
                        </div>
                      </div>
                    )}

                    {/* Result Card Display - Agent 6 */}
                    {msg.type === 'result' && msg.data?.agent === 6 && (
                      <div className="mt-4 p-5 rounded-2xl bg-zinc-900 text-white shadow-2xl overflow-hidden relative">
                        <div className="absolute top-0 right-0 p-8 opacity-10">
                          <FileText size={100} />
                        </div>
                        <div className="relative z-10">
                          <div className="text-xs font-bold text-blue-400 mb-1 uppercase tracking-widest">Document Generated</div>
                          <h3 className="text-lg font-bold mb-4">Prior-Auth Form Completed</h3>
                          
                          <div className="grid grid-cols-2 gap-4 mb-6">
                            <div className="bg-white/10 p-3 rounded-xl border border-white/5">
                              <div className="text-[10px] text-zinc-400 uppercase">Fields Filled</div>
                              <div className="text-xl font-bold">{msg.data.fields_filled}</div>
                            </div>
                            <div className="bg-white/10 p-3 rounded-xl border border-white/5">
                              <div className="text-[10px] text-zinc-400 uppercase">Status</div>
                              <div className="text-xl font-bold text-emerald-400">Ready</div>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <a 
                              href={`${api.API_BASE}/static/${msg.data.filled_pdf_path?.split('\\').pop()}`}
                              target="_blank"
                              className="w-full flex items-center justify-center gap-2 py-3 bg-white text-zinc-900 rounded-xl font-bold hover:bg-zinc-200 transition-colors"
                            >
                              <FileCheck size={18} /> View Filled PDF
                            </a>
                            <p className="text-[10px] text-center text-zinc-500 mt-2">
                              PDF generated using pypdf FreeText annotations. Standard HIPAA-compliant EWA form.
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Technical Details Expander */}
                    {msg.data && !msg.loading && (
                      <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800">
                        <details className="group">
                          <summary className="flex items-center gap-2 text-[10px] font-bold text-zinc-400 uppercase tracking-widest cursor-pointer hover:text-blue-500 transition-colors list-none">
                            <ChevronRight size={12} className="group-open:rotate-90 transition-transform" />
                            View Technical Details
                          </summary>
                          <div className="mt-3 overflow-hidden rounded-xl border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/50 p-3 text-xs space-y-4">
                            
                            {/* Agent 1 Details */}
                            {msg.agentName?.includes("Agent 1") && msg.data.documents && (
                              <div className="space-y-3">
                                {msg.data.documents.map((d: any, i: number) => (
                                  <div key={i} className="space-y-1">
                                    <div className="font-bold text-blue-600 dark:text-blue-400">{d.document_type}</div>
                                    <div className="text-zinc-500 italic leading-relaxed whitespace-pre-wrap line-clamp-3 hover:line-clamp-none transition-all">{d.content}</div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {/* Agent 2 Details */}
                            {msg.agentName?.includes("Agent 2") && msg.data.policy_search_fields && (
                              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                                {Object.entries(msg.data.policy_search_fields).map(([key, val]: [string, any]) => (
                                  val && (
                                    <div key={key} className="flex flex-col">
                                      <span className="text-[9px] uppercase font-bold text-zinc-400">{key.replace(/_/g, ' ')}</span>
                                      <span className="text-zinc-700 dark:text-zinc-300 truncate" title={String(val)}>{Array.isArray(val) ? val.join(', ') : String(val)}</span>
                                    </div>
                                  )
                                ))}
                              </div>
                            )}

                            {/* Agent 3 Details */}
                            {msg.agentName?.includes("Agent 3") && (
                              <div className="space-y-4">
                                {msg.data.document_requirements?.length > 0 && (
                                  <div className="space-y-2">
                                    <div className="text-[9px] font-bold text-zinc-400 uppercase">Required Documents</div>
                                    {msg.data.document_requirements.map((r: any, i: number) => (
                                      <div key={i} className="p-2 bg-white dark:bg-zinc-900 rounded-lg border border-zinc-100 dark:border-zinc-800">
                                        <div className="font-bold">{r.document_type}</div>
                                        <div className="text-[10px] text-zinc-500 mt-1">{r.info_needed}</div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {msg.data.medical_requirements?.length > 0 && (
                                  <div className="space-y-2">
                                    <div className="text-[9px] font-bold text-zinc-400 uppercase">Clinical Criteria</div>
                                    {msg.data.medical_requirements.map((r: any, i: number) => (
                                      <div key={i} className="p-2 bg-white dark:bg-zinc-900 rounded-lg border border-zinc-100 dark:border-zinc-800">
                                        <div className="flex justify-between items-center">
                                          <span className="font-bold">{r.requirement}</span>
                                          <span className="text-[9px] bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded uppercase">{r.importance}</span>
                                          {r.image_url && (
                                            <img 
                                              src={`${api.API_BASE}/static/${r.image_url.split('static/').pop()}`}
                                              alt="Clinical evidence"
                                              className="h-10 w-10 object-cover rounded-md ml-2 cursor-pointer"
                                              onClick={() => setViewerDoc({ name: 'Clinical Evidence', url: `${api.API_BASE}/static/${r.image_url.split('static/').pop()}`, type: 'image/jpeg' })}
                                            />
                                          )}
                                        </div>
                                        <div className="text-[10px] text-zinc-500 mt-1">{r.description}</div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Agent 4 Details */}
                            {msg.agentName?.includes("Agent 4") && (
                              <div className="space-y-3">
                                {['satisfied', 'partial', 'missing'].map((status) => {
                                  // Simplified lookup to handle both Agent 4 normal output and Agent 4 request data
                                  const items = status === 'satisfied' ? msg.data.satisfied : 
                                                status === 'partial' ? (msg.data.partial_documents || msg.data.partial) : 
                                                (msg.data.missing_documents || msg.data.missing);
                                  
                                  if (!items || items.length === 0) return null;
                                  return (
                                    <div key={status} className="space-y-2">
                                      <div className={`text-[9px] font-bold uppercase ${status === 'satisfied' ? 'text-emerald-500' : status === 'partial' ? 'text-amber-500' : 'text-red-500'}`}>{status}</div>
                                      {items.map((item: any, i: number) => (
                                        <div key={i} className="p-2 bg-white dark:bg-zinc-900 rounded-lg border border-zinc-100 dark:border-zinc-800">
                                          <div className="font-bold">{item.document_type}</div>
                                          {item.satisfied_by && <div className="text-[9px] text-emerald-600 mt-1">Found in: {item.satisfied_by}</div>}
                                          {item.info_missing && <div className="text-[9px] text-red-500 mt-1">Missing: {item.info_missing}</div>}
                                          {item.info_needed && !item.info_missing && <div className="text-[9px] text-zinc-500 mt-1">{item.info_needed}</div>}
                                        </div>
                                      ))}
                                    </div>
                                  );
                                })}
                              </div>
                            )}

                            {/* Raw Data link as backup */}
                            <div className="pt-2 mt-2 border-t border-zinc-100 dark:border-zinc-800 flex justify-end">
                              <button 
                                onClick={() => console.log(msg.data)}
                                className="text-[9px] text-zinc-400 hover:text-blue-500"
                              >
                                Log raw JSON to console
                              </button>
                            </div>

                          </div>
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

      {/* Footer / Unified Upload Button */}
      <footer className="p-6 border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 shadow-2xl z-20">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Uploaded Documents Gallery */}
          {uploadedDocs.length > 0 && (
            <div className="flex flex-col gap-2">
              <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest px-1">Uploaded Documents ({uploadedDocs.length})</div>
              <div className="flex gap-3 overflow-x-auto pb-2 no-scrollbar">
                {uploadedDocs.map((doc, i) => (
                  <motion.div
                    key={i}
                    whileHover={{ scale: 1.05 }}
                    onClick={() => setViewerDoc(doc)}
                    className="flex-shrink-0 w-32 h-20 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800 overflow-hidden cursor-pointer relative group"
                  >
                    {doc.type.includes('image') ? (
                      <img src={doc.url} alt={doc.name} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity" />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center p-2 text-center">
                        <FileText size={20} className="text-zinc-400" />
                        <span className="text-[8px] text-zinc-500 truncate w-full mt-1 font-mono">{doc.name}</span>
                      </div>
                    )}
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                      <Eye size={16} className="text-white" />
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between gap-6">
            <div className="flex-1">
              <div className="flex items-center gap-3 text-zinc-400 dark:text-zinc-500 italic">
                 {isAnalysing ? (
                   <>
                     <Loader2 size={20} className="animate-spin text-blue-600" />
                     <span className="text-sm font-medium">Pipeline running... checking criteria one by one</span>
                   </>
                 ) : (
                   <>
                     <ShieldCheck size={20} className="text-emerald-500" />
                     <span className="text-sm font-medium">System ready for medical document analysis</span>
                   </>
                 )}
              </div>
            </div>

            <div className="flex gap-3">
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                multiple 
                className="hidden" 
              />
              <button 
                 onClick={() => fileInputRef.current?.click()}
                 disabled={isAnalysing}
                 className="flex items-center gap-3 px-8 py-4 bg-blue-600 text-white rounded-2xl font-bold shadow-xl shadow-blue-500/20 hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:scale-100 disabled:shadow-none"
              >
                <Plus size={24} />
                Add More
              </button>
            </div>
          </div>

          <p className="text-center text-[10px] text-zinc-400 uppercase tracking-[0.2em] pt-2 border-t border-zinc-100 dark:border-zinc-800">
            HIPAA Compliant • Amazon Nova • AI Medical Pipeline
          </p>
        </div>
      </footer>

      {/* Document Viewer Modal */}
      <AnimatePresence>
        {viewerDoc && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 backdrop-blur-xl bg-black/60"
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="relative w-full max-w-5xl max-h-[90vh] bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl overflow-hidden flex flex-col"
            >
              <div className="flex items-center justify-between p-4 border-b border-zinc-100 dark:border-zinc-800">
                <div className="flex items-center gap-3 px-2">
                  <div className="p-2 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
                    {viewerDoc.type.includes('image') ? <ImageIcon size={20} className="text-blue-600" /> : <FileText size={20} className="text-blue-600" />}
                  </div>
                  <div>
                    <h3 className="font-bold text-zinc-900 dark:text-white truncate max-w-[200px] sm:max-w-md">{viewerDoc.name}</h3>
                    <p className="text-xs text-zinc-500 uppercase tracking-widest">{viewerDoc.type}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <a 
                    href={viewerDoc.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg text-zinc-500 transition-colors"
                  >
                    <ExternalLink size={20} />
                  </a>
                  <button 
                    onClick={() => setViewerDoc(null)}
                    className="p-2 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg text-zinc-500 hover:text-red-600 transition-colors"
                  >
                    <X size={20} />
                  </button>
                </div>
              </div>

              <div className="flex-1 bg-zinc-50 dark:bg-zinc-950 overflow-auto p-4 flex items-center justify-center">
                {viewerDoc.type.includes('image') ? (
                  <img src={viewerDoc.url} alt={viewerDoc.name} className="max-w-full h-auto rounded-lg shadow-lg" />
                ) : (
                  <iframe 
                    src={`${viewerDoc.url}#toolbar=0`} 
                    className="w-full h-full min-h-[60vh] rounded-lg border-0"
                    title={viewerDoc.name}
                  />
                )}
              </div>
              
              <div className="p-4 bg-zinc-50 dark:bg-zinc-900/50 border-t border-zinc-100 dark:border-zinc-800 text-center text-xs text-zinc-400">
                Document preview is handled locally within your browser.
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
