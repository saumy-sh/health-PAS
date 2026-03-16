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
  X
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
          content: "Documents processed successfully. Identified content: " + a1Result.documents.map((d: any) => d.document_type).join(", "),
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
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Checking policy details...", loading: true, agentName: "Agent 2" });
        const res = await api.runAgent2(currentInput);
        updateMessage(loaderId, { 
          content: `Policy Identified: ${res.insurer_name || 'Generic'}. Procedure: ${res.procedure_identified}`, 
          loading: false 
        });
        currentInput = res;
        setLastAgentData(res);
      }

      // Step 3: Policy Retriever
      if (startStep <= 3) {
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Retrieving policy requirements...", loading: true, agentName: "Agent 3" });
        const res = await api.runAgent3(currentInput);
        updateMessage(loaderId, { 
          content: `Found ${res.document_requirements?.length || 0} document requirements and ${res.medical_requirements?.length || 0} clinical criteria.`,
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
            content: "Some required documents are missing or incomplete.",
            loading: false 
          });
          
          addMessage({
            role: 'agent',
            type: 'document_request',
            content: "Please provide the following missing documents to proceed, or click 'Proceed Anyway' for preliminary results.",
            data: { 
              missing: res.missing_documents,
              partial: res.partial_documents
            },
            agentName: "Agent 4"
          });
          setIsAnalysing(false);
          return; // Stop here and wait for user
        } else {
          updateMessage(loaderId, { 
            content: "All required documents are present.",
            loading: false 
          });
          currentInput = res;
        }
      }

      // Step 5: Eligibility Reasoning
      if (startStep <= 5) {
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Performing final clinical analysis...", loading: true, agentName: "Agent 5" });
        const res = await api.runAgent5(currentInput);
        setLastAgentData(res);
        
        updateMessage(loaderId, { 
          type: 'result',
          content: `Clinical Analysis Complete. Determination: ${res.determination}`,
          data: { ...res, agent: 5 },
          loading: false,
          agentName: "Agent 5"
        });
        currentInput = res;
      }

      // Step 6: Form Filler
      if (startStep <= 6) {
        console.log("[Pipeline] Starting Agent 6 (Form Filler)...");
        const loaderId = addMessage({ role: 'agent', type: 'text', content: "Generating pre-authorization PDF form...", loading: true, agentName: "Agent 6" });
        const res = await api.runAgent6(currentInput);
        console.log("[Pipeline] Agent 6 Success:", res);
        setLastAgentData(res);
        
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
      addMessage({ role: 'user', type: 'text', content: "Proceed with current documents." });
      runPipeline(5, lastAgentData); // Skip Agent 4's block and go to Agent 5
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

                    {/* Document Request Display */}
                    {msg.type === 'document_request' && msg.data && (
                      <div className="mt-4 space-y-3">
                        <div className="grid gap-2">
                          {msg.data.missing?.map((doc: any, i: number) => (
                            <div key={i} className="flex items-start gap-3 p-3 bg-red-50 dark:bg-red-900/10 rounded-xl border border-red-100 dark:border-red-900/20 text-sm">
                              <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
                              <div className="text-left">
                                <div className="font-bold text-red-700 dark:text-red-400">MISSING: {doc.document_type}</div>
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
                             className="flex-1 flex items-center justify-center gap-2 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 transition-colors"
                           >
                             <Plus size={18} /> Upload Now
                           </button>
                           <button 
                             onClick={handleProceedManual}
                             className="flex-1 py-3 bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 rounded-xl font-bold hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                           >
                             Proceed Anyway
                           </button>
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
                              href={`http://localhost:8001/static/${msg.data.filled_pdf_path?.split('\\').pop()}`}
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
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={chatEndRef} />
      </main>

      {/* Footer / Input */}
      <footer className="p-6 border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900">
        <div className="max-w-4xl mx-auto flex gap-4">
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
             className="p-4 bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 rounded-2xl hover:bg-blue-600 hover:text-white transition-all disabled:opacity-50"
             title="Upload more documents"
          >
            <Plus size={24} />
          </button>
          
          <div className="flex-1 relative text-zinc-400">
            <div className="w-full p-4 pr-12 bg-zinc-100 dark:bg-zinc-800 border-transparent rounded-2xl italic text-sm flex items-center gap-2">
               {isAnalysing ? (
                 <>
                   <Loader2 size={16} className="animate-spin text-blue-600" />
                   <span>Agent Pipeline Active...</span>
                 </>
               ) : (
                 <>
                   <Bot size={16} />
                   <span>Agents are waiting for documents. Use the <Plus size={14} className="inline mx-0.5" /> button to upload.</span>
                 </>
               )}
            </div>
            <button className="absolute right-3 top-3 p-2 text-zinc-400 cursor-not-allowed">
              <Send size={20} />
            </button>
          </div>
        </div>
        <p className="text-center text-[10px] text-zinc-400 mt-4 uppercase tracking-[0.2em]">
          Powered by Amazon Nova & Multimodal Intelligence
        </p>
      </footer>
    </div>
  );
}
