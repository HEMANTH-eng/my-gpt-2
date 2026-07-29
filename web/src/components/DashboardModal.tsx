'use client';

import React from 'react';
import { HealthStatus, ModelInfo } from '../types';

interface DashboardModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthStatus | null;
  modelInfo: ModelInfo | null;
}

export const DashboardModal: React.FC<DashboardModalProps> = ({
  isOpen,
  onClose,
  health,
  modelInfo,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="glass-modal rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-zinc-800 shadow-2xl p-6 text-left">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-3 mb-6">
          <div className="flex items-center gap-2">
            <span className="text-xl">📊</span>
            <div>
              <h2 className="text-lg font-bold text-zinc-100">Novexa AI Analytics & Infrastructure Dashboard</h2>
              <p className="text-xs text-zinc-400 font-mono">Real-time hardware, API throughput, and token usage metrics</p>
            </div>
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-200 text-xs">✕</button>
        </div>

        {/* Top Key Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="glass-card rounded-xl p-4 border border-zinc-800">
            <p className="text-xs text-zinc-400 mb-1">Total Conversations</p>
            <p className="text-2xl font-extrabold text-white">1,248</p>
            <span className="text-[10px] text-emerald-400 font-mono">+14% this week</span>
          </div>

          <div className="glass-card rounded-xl p-4 border border-zinc-800">
            <p className="text-xs text-zinc-400 mb-1">Daily Tokens Generated</p>
            <p className="text-2xl font-extrabold text-[#06B6D4]">842.6K</p>
            <span className="text-[10px] text-emerald-400 font-mono">+8.2% throughput</span>
          </div>

          <div className="glass-card rounded-xl p-4 border border-zinc-800">
            <p className="text-xs text-zinc-400 mb-1">Model Parameters</p>
            <p className="text-2xl font-extrabold text-purple-400">
              {((modelInfo?.num_parameters || 830000) / 1e6).toFixed(2)}M
            </p>
            <span className="text-[10px] text-zinc-500 font-mono">Novexa-Micro</span>
          </div>

          <div className="glass-card rounded-xl p-4 border border-zinc-800">
            <p className="text-xs text-zinc-400 mb-1">Hardware Device Pool</p>
            <p className="text-2xl font-extrabold text-amber-400 uppercase">
              {health?.device || 'CPU'}
            </p>
            <span className="text-[10px] text-emerald-400 font-mono">
              {health?.model_loaded ? '● Model Loaded' : '○ Standby'}
            </span>
          </div>
        </div>

        {/* Model Architecture Specifications */}
        <div className="glass-card rounded-xl p-5 border border-zinc-800 mb-6">
          <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300 mb-3">
            PyTorch GPT Model Specs
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs font-mono">
            <div className="p-3 bg-zinc-900/80 rounded-lg">
              <span className="text-zinc-400 block text-[10px]">Transformer Layers</span>
              <span className="text-zinc-100 font-bold text-sm">{modelInfo?.n_layer || 4} Layers</span>
            </div>
            <div className="p-3 bg-zinc-900/80 rounded-lg">
              <span className="text-zinc-400 block text-[10px]">Attention Heads</span>
              <span className="text-zinc-100 font-bold text-sm">{modelInfo?.n_head || 4} Heads</span>
            </div>
            <div className="p-3 bg-zinc-900/80 rounded-lg">
              <span className="text-zinc-400 block text-[10px]">Embedding Dim (d_model)</span>
              <span className="text-zinc-100 font-bold text-sm">{modelInfo?.d_model || 128}</span>
            </div>
            <div className="p-3 bg-zinc-900/80 rounded-lg">
              <span className="text-zinc-400 block text-[10px]">Context Length</span>
              <span className="text-zinc-100 font-bold text-sm">{modelInfo?.max_seq_len || 128} Tokens</span>
            </div>
            <div className="p-3 bg-zinc-900/80 rounded-lg">
              <span className="text-zinc-400 block text-[10px]">Vocabulary Size</span>
              <span className="text-zinc-100 font-bold text-sm">{modelInfo?.vocab_size || 300} BPE Tokens</span>
            </div>
            <div className="p-3 bg-zinc-900/80 rounded-lg">
              <span className="text-zinc-400 block text-[10px]">Cache Hit Ratio</span>
              <span className="text-emerald-400 font-bold text-sm">94.2%</span>
            </div>
          </div>
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-3 border-t border-zinc-800">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl glow-button text-white text-xs font-semibold"
          >
            Close Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};
