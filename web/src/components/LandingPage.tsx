'use client';

import React, { useState } from 'react';

interface LandingPageProps {
  onLaunchStudio: () => void;
  onOpenAuth: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onLaunchStudio, onOpenAuth }) => {
  const [activeTab, setActiveTab] = useState<'chat' | 'agents' | 'vision' | 'code' | 'rag'>('chat');
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  const features = [
    {
      icon: '⚡',
      title: 'High-Throughput GPT Core',
      description: 'Custom PyTorch GPT architecture operating with scaled causal self-attention, learned position embeddings, and real-time autoregressive streaming.',
    },
    {
      icon: '🤖',
      title: 'Autonomous AI Agents',
      description: 'ReAct agent execution engine supporting specialized Coding, Research, Email, Calendar, Browser, and Data Analysis agents.',
    },
    {
      icon: '👁️',
      title: 'Multimodal Vision Understanding',
      description: 'Patch convolutional feature projector feeding vision token embeddings directly into sequence attention layers for instant visual reasoning.',
    },
    {
      icon: '📄',
      title: 'Document PDF & DOCX RAG',
      description: 'Parse, extract, and ground responses using uploaded PDF, Word documents, and text files with zero contextual hallucination.',
    },
    {
      icon: '🔧',
      title: 'Extensible Tool Calling',
      description: 'Integrated Python REPL, Web Search, Calculator, Database queries, PDF parsing, and Image generation tools.',
    },
    {
      icon: '🔐',
      title: 'Enterprise Security & RBAC',
      description: 'AES-256 Fernet payload encryption, threat sanitizer, sliding-window rate limiting, and SHA-256 hashed API keys.',
    },
  ];

  const pricingPlans = [
    {
      name: 'Free Starter',
      price: '$0',
      period: 'forever',
      description: 'Essential AI capabilities for personal exploration & lightweight tasks.',
      features: ['Novexa-Micro (0.83M)', '1,000 Chat Tokens/day', 'Standard Personas', 'Basic Code REPL & Math Tools', 'Community Support'],
      cta: 'Get Started Free',
      highlighted: false,
    },
    {
      name: 'Pro Workspace',
      price: '$20',
      period: 'per month',
      description: 'Unrestricted access for power users, developers, and researchers.',
      features: ['Novexa-Micro & Trained Models', 'Unlimited Chat Tokens', 'Autonomous AI Agents Suite', 'Multimodal Vision & PDF RAG', 'AES-256 Encryption & API Keys', 'Priority 24/7 Support'],
      cta: 'Start 14-Day Free Trial',
      highlighted: true,
    },
    {
      name: 'Team & Enterprise',
      price: '$50',
      period: 'per seat/month',
      description: 'Advanced governance, custom deployment, and multi-user collaboration.',
      features: ['Dedicated Kubernetes Auto-Scaling', 'Custom Fine-Tuning Pipeline', 'Multiple Workspaces & RBAC', 'Long-Term Memory Engine', 'Custom Tool Integrations', 'Dedicated Account Manager'],
      cta: 'Contact Sales',
      highlighted: false,
    },
  ];

  const faqs = [
    {
      q: 'What makes Novexa AI unique?',
      a: 'Novexa AI is built completely from scratch using Python, PyTorch, and Next.js without external Hugging Face model dependencies. It offers complete sovereignty over training pipelines, tokenization, inference engines, and security.',
    },
    {
      q: 'Can I run Novexa AI locally or on custom GPU hardware?',
      a: 'Yes! Novexa AI automatically detects CUDA GPU hardware allocations and falls back to optimized CPU execution. It includes Docker Compose and Kubernetes HPA deployment scripts.',
    },
    {
      q: 'How does Document RAG context ingestion work?',
      a: 'Upload PDF, DOCX, or TXT documents via the API or Web Studio. The file parser extracts raw text context and grounds the model generation to answer queries based on your uploaded document.',
    },
    {
      q: 'Is my data encrypted and secure?',
      a: 'All sensitive payload entries can be symmetrically encrypted using AES-256 Fernet encryption at rest, protected with threat sanitization filters and structured audit logs.',
    },
  ];

  return (
    <div className="min-h-screen bg-[#090D16] text-zinc-100 flex flex-col font-sans">
      {/* Background Animated Glow Gradients */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-[#4F46E5]/20 rounded-full blur-3xl animate-pulse-glow" />
        <div className="absolute top-1/3 -right-40 w-[30rem] h-[30rem] bg-[#7C3AED]/15 rounded-full blur-3xl animate-pulse-glow" />
        <div className="absolute bottom-10 left-1/4 w-[28rem] h-[28rem] bg-[#06B6D4]/15 rounded-full blur-3xl animate-pulse-glow" />
      </div>

      {/* Hero Section */}
      <section className="relative z-10 pt-20 pb-16 px-4 md:px-8 max-w-6xl mx-auto text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass-card border border-[#06B6D4]/30 text-xs text-[#06B6D4] font-mono mb-6">
          <span className="w-2 h-2 rounded-full bg-[#06B6D4] animate-ping" />
          Novexa AI v2.0 Enterprise Platform Released
        </div>

        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white mb-6 leading-tight">
          Build. Think. Create. <br />
          <span className="gradient-text">The Next Generation AI Workspace</span>
        </h1>

        <p className="max-w-2xl mx-auto text-sm sm:text-base text-zinc-300 mb-10 leading-relaxed font-normal">
          Empowering engineers, researchers, and enterprises with custom PyTorch GPT language models, Autonomous AI Agents, Multimodal Vision, and Extensible Tool Calling.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 mb-16">
          <button
            onClick={onLaunchStudio}
            className="px-8 py-3.5 rounded-xl glow-button text-white font-semibold text-sm tracking-wide shadow-xl flex items-center gap-2"
          >
            <span>Launch AI Studio</span>
            <span>→</span>
          </button>
          <button
            onClick={onOpenAuth}
            className="px-8 py-3.5 rounded-xl bg-zinc-900/90 hover:bg-zinc-800 text-zinc-200 border border-zinc-700/80 font-semibold text-sm transition-all"
          >
            Sign In / Register
          </button>
        </div>

        {/* Product Showcase Demo Card */}
        <div className="glass-panel rounded-2xl p-4 sm:p-6 border border-zinc-700/60 shadow-2xl text-left max-w-4xl mx-auto">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-3 mb-4">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-rose-500/80" />
              <span className="w-3 h-3 rounded-full bg-amber-500/80" />
              <span className="w-3 h-3 rounded-full bg-emerald-500/80" />
              <span className="text-xs text-zinc-400 font-mono ml-2">Novexa Studio Preview</span>
            </div>
            <div className="flex gap-2">
              {(['chat', 'agents', 'vision', 'code', 'rag'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`text-xs px-3 py-1 rounded-lg font-mono capitalize transition-all ${
                    activeTab === tab ? 'bg-[#4F46E5] text-white' : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-zinc-950/90 rounded-xl p-4 font-mono text-xs sm:text-sm text-zinc-300 space-y-3 min-h-[220px]">
            {activeTab === 'chat' && (
              <div>
                <p className="text-[#06B6D4] font-semibold">User: Explain scaled causal dot-product self-attention.</p>
                <p className="mt-2 text-zinc-200 leading-relaxed">
                  Assistant: Scaled dot-product self-attention calculates similarity scores between Query (Q) and Key (K) matrices scaled by 1/√d_k, applying upper-triangular boolean causal masking (-∞) to prevent future token leaks:
                </p>
                <pre className="mt-2 bg-zinc-900/90 p-3 rounded-lg text-emerald-400 text-xs overflow-x-auto">
                  {"Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k) + M) @ V"}
                </pre>
              </div>
            )}
            {activeTab === 'agents' && (
              <div>
                <p className="text-[#06B6D4] font-semibold">Agent Manager: Autonomous ReAct Coding Agent Goal: Write fibonacci function & test syntax.</p>
                <p className="mt-2 text-amber-400">Step 1 [Thought]: Generating Python code snippet for Fibonacci recursion.</p>
                <p className="text-purple-400">Step 2 [Action]: Invoking python_repl tool...</p>
                <p className="text-emerald-400">Step 3 [Observation]: Execution output validated successfully: [0, 1, 1, 2, 3, 5, 8].</p>
              </div>
            )}
            {activeTab === 'vision' && (
              <div>
                <p className="text-[#06B6D4] font-semibold">Vision Encoder: Image Ingestion (64x64 Patch Projection)</p>
                <p className="mt-2 text-zinc-200">
                  Multimodal feature projector mapped image tensor into 64 sequence embeddings. Classified input graphic: "Architecture Diagram of Micro Transformer Decoder".
                </p>
              </div>
            )}
            {activeTab === 'code' && (
              <div>
                <p className="text-[#06B6D4] font-semibold">Python REPL Tool: math.sqrt(144) + 50</p>
                <p className="mt-2 text-emerald-400">Result: 62.0</p>
              </div>
            )}
            {activeTab === 'rag' && (
              <div>
                <p className="text-[#06B6D4] font-semibold">Document RAG Parser: Research_Paper.pdf (12.4 KB)</p>
                <p className="mt-2 text-zinc-200">
                  Parsed 1,420 words context string. Grounded answer: "The model uses AdamW weight decay splitting with cosine annealing learning rate scheduler."
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative z-10 py-16 px-4 md:px-8 max-w-6xl mx-auto text-center">
        <h2 className="text-2xl sm:text-4xl font-bold text-white mb-4">
          Architected for Speed & Enterprise Intelligence
        </h2>
        <p className="text-sm text-zinc-400 mb-12 max-w-xl mx-auto">
          Every layer of Novexa AI is designed with modular PyTorch components and state-of-the-art web interface capabilities.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 text-left">
          {features.map((f, i) => (
            <div
              key={i}
              className="glass-card rounded-2xl p-6 hover:border-[#06B6D4]/50 transition-all hover:-translate-y-1"
            >
              <div className="text-3xl mb-3">{f.icon}</div>
              <h3 className="font-bold text-base text-zinc-100 mb-2">{f.title}</h3>
              <p className="text-xs text-zinc-400 leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pricing Matrix */}
      <section className="relative z-10 py-16 px-4 md:px-8 max-w-6xl mx-auto text-center">
        <h2 className="text-2xl sm:text-4xl font-bold text-white mb-4">
          Simple, Transparent Pricing
        </h2>
        <p className="text-sm text-zinc-400 mb-12">
          Scale effortlessly from personal sandbox projects to enterprise production environments.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {pricingPlans.map((plan, i) => (
            <div
              key={i}
              className={`glass-panel rounded-2xl p-6 text-left flex flex-col justify-between border ${
                plan.highlighted
                  ? 'border-[#4F46E5] ring-2 ring-[#4F46E5]/40 bg-zinc-900/90'
                  : 'border-zinc-800'
              }`}
            >
              <div>
                {plan.highlighted && (
                  <span className="text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full bg-[#4F46E5] text-white float-right">
                    Most Popular
                  </span>
                )}
                <h3 className="font-bold text-lg text-zinc-100">{plan.name}</h3>
                <div className="mt-3 mb-2 flex items-baseline gap-1">
                  <span className="text-4xl font-extrabold text-white">{plan.price}</span>
                  <span className="text-xs text-zinc-400">{plan.period}</span>
                </div>
                <p className="text-xs text-zinc-400 mb-6">{plan.description}</p>

                <ul className="space-y-2.5 mb-8">
                  {plan.features.map((feat, fi) => (
                    <li key={fi} className="text-xs text-zinc-300 flex items-center gap-2">
                      <span className="text-[#06B6D4]">✓</span> {feat}
                    </li>
                  ))}
                </ul>
              </div>

              <button
                onClick={onLaunchStudio}
                className={`w-full py-2.5 rounded-xl font-semibold text-xs transition-all ${
                  plan.highlighted
                    ? 'glow-button text-white'
                    : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-200'
                }`}
              >
                {plan.cta}
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* FAQ Section */}
      <section className="relative z-10 py-16 px-4 md:px-8 max-w-4xl mx-auto w-full">
        <h2 className="text-2xl sm:text-3xl font-bold text-white text-center mb-8">
          Frequently Asked Questions
        </h2>

        <div className="space-y-4">
          {faqs.map((faq, idx) => {
            const isOpen = openFaqIndex === idx;
            return (
              <div
                key={idx}
                className="glass-card rounded-xl border border-zinc-800/80 overflow-hidden text-left"
              >
                <button
                  onClick={() => setOpenFaqIndex(isOpen ? null : idx)}
                  className="w-full p-4 flex items-center justify-between text-sm font-semibold text-zinc-100 hover:text-[#06B6D4] transition-colors"
                >
                  <span>{faq.q}</span>
                  <span>{isOpen ? '−' : '+'}</span>
                </button>
                {isOpen && (
                  <div className="px-4 pb-4 text-xs text-zinc-400 leading-relaxed border-t border-zinc-800/50 pt-3">
                    {faq.a}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-zinc-800/80 bg-zinc-950 py-10 px-4 md:px-8 text-center text-xs text-zinc-400">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-[#4F46E5] flex items-center justify-center font-bold text-white text-[10px]">
              NA
            </div>
            <span className="font-bold text-zinc-200">Novexa AI</span>
            <span>— The Next Generation AI Workspace</span>
          </div>
          <p>© 2026 Novexa AI Platform. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};
