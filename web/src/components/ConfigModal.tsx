'use client';

import React from 'react';
import { ModelConfig } from '../types';

interface ConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: ModelConfig;
  onChangeConfig: (newConfig: ModelConfig) => void;
}

export const ConfigModal: React.FC<ConfigModalProps> = ({
  isOpen,
  onClose,
  config,
  onChangeConfig,
}) => {
  if (!isOpen) return null;

  const handleReset = () => {
    onChangeConfig({
      temperature: 0.7,
      top_k: 40,
      top_p: 0.95,
      max_new_tokens: 100,
      greedy: false,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="bg-zinc-900 border border-zinc-800 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-6">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan-400" />
            <h3 className="font-semibold text-zinc-100 text-sm">Hyperparameter Controls</h3>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-400 hover:text-zinc-200 text-xs p-1"
          >
            ✕
          </button>
        </div>

        {/* Form Controls */}
        <div className="space-y-4 text-xs">
          {/* Temperature Slider */}
          <div>
            <div className="flex justify-between mb-1 text-zinc-300 font-mono">
              <span>Temperature</span>
              <span className="text-cyan-400">{config.temperature}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="2.0"
              step="0.05"
              value={config.temperature}
              onChange={(e) =>
                onChangeConfig({ ...config, temperature: parseFloat(e.target.value) })
              }
              className="w-full accent-cyan-500 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
            />
            <p className="text-[10px] text-zinc-400 mt-1">
              Higher values increase randomness; lower values make output more deterministic.
            </p>
          </div>

          {/* Top-p Nucleus Slider */}
          <div>
            <div className="flex justify-between mb-1 text-zinc-300 font-mono">
              <span>Top-p (Nucleus)</span>
              <span className="text-cyan-400">{config.top_p}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={config.top_p}
              onChange={(e) =>
                onChangeConfig({ ...config, top_p: parseFloat(e.target.value) })
              }
              className="w-full accent-cyan-500 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
            />
            <p className="text-[10px] text-zinc-400 mt-1">
              Limits sampling to the smallest set of tokens with cumulative probability ≥ p.
            </p>
          </div>

          {/* Top-k Slider */}
          <div>
            <div className="flex justify-between mb-1 text-zinc-300 font-mono">
              <span>Top-k Truncation</span>
              <span className="text-cyan-400">{config.top_k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={config.top_k}
              onChange={(e) =>
                onChangeConfig({ ...config, top_k: parseInt(e.target.value) })
              }
              className="w-full accent-cyan-500 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
            />
          </div>

          {/* Max Tokens Slider */}
          <div>
            <div className="flex justify-between mb-1 text-zinc-300 font-mono">
              <span>Max Tokens to Generate</span>
              <span className="text-cyan-400">{config.max_new_tokens}</span>
            </div>
            <input
              type="range"
              min="10"
              max="512"
              step="10"
              value={config.max_new_tokens}
              onChange={(e) =>
                onChangeConfig({ ...config, max_new_tokens: parseInt(e.target.value) })
              }
              className="w-full accent-cyan-500 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
            />
          </div>

          {/* Greedy Decoding Checkbox */}
          <div className="flex items-center justify-between pt-2 border-t border-zinc-800">
            <div>
              <span className="font-medium text-zinc-200">Greedy Decoding</span>
              <p className="text-[10px] text-zinc-400">Always pick single highest-probability token</p>
            </div>
            <input
              type="checkbox"
              checked={config.greedy}
              onChange={(e) =>
                onChangeConfig({ ...config, greedy: e.target.checked })
              }
              className="w-4 h-4 accent-cyan-500 rounded cursor-pointer"
            />
          </div>
        </div>

        {/* Modal Actions */}
        <div className="flex items-center justify-between pt-4 border-t border-zinc-800">
          <button
            onClick={handleReset}
            className="text-xs text-zinc-400 hover:text-zinc-200"
          >
            Reset Defaults
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs shadow-md shadow-cyan-950"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
