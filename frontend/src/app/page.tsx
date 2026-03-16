"use client";

import Link from "next/link";
import { Shield, Activity, FileCheck, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";

export default function Home() {
  return (
    <div className="relative isolate min-h-screen overflow-hidden bg-white dark:bg-zinc-950">
      {/* Background decoration */}
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
        <div
          className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
          style={{
            clipPath:
              "polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)",
          }}
        />
      </div>

      <div className="mx-auto max-w-7xl px-6 pb-24 pt-10 sm:pb-32 lg:flex lg:px-8 lg:pt-40">
        <div className="mx-auto max-w-2xl flex-shrink-0 lg:mx-0 lg:max-w-xl lg:pt-8">
          <div className="mt-24 sm:mt-32 lg:mt-16">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex space-x-6"
            >
              <span className="rounded-full bg-blue-600/10 px-3 py-1 text-sm font-semibold leading-6 text-blue-600 ring-1 ring-inset ring-blue-600/10">
                New v2.0
              </span>
              <span className="inline-flex items-center space-x-2 text-sm font-medium leading-6 text-zinc-600 dark:text-zinc-400">
                <span>Intelligent Health Analysis</span>
              </span>
            </motion.div>
          </div>
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mt-10 text-4xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-6xl"
          >
            <span className="gradient-text">InsuranceHelper</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mt-6 text-lg leading-8 text-zinc-600 dark:text-zinc-400"
          >
            Streamline your health pre-authorization process with our advanced multi-agent pipeline. 
            Upload documents, analyze policies, and get eligibility reasoning in seconds.
          </motion.p>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mt-10 flex items-center gap-x-6"
          >
            <Link
              href="/analyse"
              className="group relative inline-flex items-center gap-x-2 rounded-full bg-zinc-900 px-8 py-4 text-sm font-semibold text-white shadow-sm hover:bg-zinc-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-900 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Analyse Files
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
          </motion.div>
          
          <div className="mt-20 grid grid-cols-1 gap-8 sm:grid-cols-3">
             <div className="flex flex-col gap-2">
                <Shield className="h-8 w-8 text-blue-600" />
                <h3 className="font-semibold dark:text-white">Secure</h3>
                <p className="text-sm text-zinc-500">HIPAA compliant document processing</p>
             </div>
             <div className="flex flex-col gap-2">
                <Activity className="h-8 w-8 text-purple-600" />
                <h3 className="font-semibold dark:text-white">Intelligent</h3>
                <p className="text-sm text-zinc-500">ML-driven eligibility reasoning</p>
             </div>
             <div className="flex flex-col gap-2">
                <FileCheck className="h-8 w-8 text-indigo-600" />
                <h3 className="font-semibold dark:text-white">Automated</h3>
                <p className="text-sm text-zinc-500">Electronic form filling & submission</p>
             </div>
          </div>
        </div>
        
        {/* Right side - visual aid */}
        <div className="mx-auto mt-16 flex max-w-2xl sm:mt-24 lg:ml-10 lg:mr-0 lg:mt-0 lg:max-w-none lg:flex-none xl:ml-32">
          <div className="max-w-3xl flex-none sm:max-w-5xl lg:max-w-none">
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.4, type: "spring", stiffness: 100 }}
              className="glass -m-2 rounded-xl p-2 lg:-m-4 lg:rounded-2xl"
            >
              <div className="rounded-md bg-white/5 shadow-2xl ring-1 ring-black/10 dark:ring-white/10">
                 <div className="p-8 space-y-4">
                    <div className="flex items-center gap-4 border-b border-zinc-100 dark:border-zinc-800 pb-4">
                       <div className="h-3 w-3 rounded-full bg-red-400" />
                       <div className="h-3 w-3 rounded-full bg-yellow-400" />
                       <div className="h-3 w-3 rounded-full bg-green-400" />
                       <div className="ml-auto text-xs font-mono text-zinc-400">analysis_v2.log</div>
                    </div>
                    <div className="space-y-2 font-mono text-sm">
                       <div className="text-blue-500">[Agent 1] Document Intelligence Started...</div>
                       <div className="text-zinc-400">  &gt; Processing MedicalNotes.pdf</div>
                       <div className="text-emerald-500">[Agent 2] Policy Identified: BlueCross Gold</div>
                       <div className="text-zinc-400">  &gt; Requirement: Physical Therapy (4 weeks)</div>
                       <div className="text-amber-500">[Agent 4] Missing Document: Insurance Card</div>
                       <div className="animate-pulse text-zinc-600 dark:text-zinc-500">_</div>
                    </div>
                 </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
