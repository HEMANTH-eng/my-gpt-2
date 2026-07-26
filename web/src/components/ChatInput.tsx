'use client';

import React, { useState, KeyboardEvent } from 'react';
import { ModelConfig } from '../types';

interface ChatInputProps {
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  config: ModelConfig;
}

const PROMPT_SUGGESTIONS = [
  'Explain Self-Attention simply',
  'Write a PyTorch Transformer block',
  'How does Byte Pair Encoding work?',
];

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  config,
}) => {
  const [input, setInput] = useState('');

  const handleSubmit = () => {
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="max-w-3xl mx-auto w-full p-4">
      {/* Quick Suggestion Chips */}
      <div className="flex flex-wrap gap-2 mb-3">
        {PROMPT_SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => onSendMessage(suggestion)}
            disabled={isLoading}
            className="text-[11px] px-3 py-1.5 rounded-full bg-zinc-900/80 hover:bg-zinc-800 text-zinc-300 hover:text-cyan-400 border border-zinc-800 transition-all cursor-pointer disabled:opacity-50"
          >
            💡 {suggestion}
          </button>
        ))}
      </div>

      {/* Input Box Container */}
      <div className="relative bg-zinc-900/90 border border-zinc-800 focus-within:border-cyan-500/60 rounded-2xl p-3 shadow-xl backdrop-blur-md transition-all">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask MyGPT anything... (Press Enter to send, Shift+Enter for new line)"
          disabled={isLoading}
          rows={2}
          className="w-full bg-transparent text-xs sm:text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none resize-none pr-12"
        />

        {/* Bottom Status & Submit Button */}
        <div className="flex items-center justify-between pt-2 border-t border-zinc-800/60">
          <div className="text-[10px] font-mono text-zinc-500 flex items-center gap-2">
            <span>Temp: {config.temperature}</span>
            <span>•</span>
            <span>Top-p: {config.top_p}</span>
            <span>•</span>
            <span>Max Tokens: {config.max_new_tokens}</span>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className="p-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md shadow-cyan-950"
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
