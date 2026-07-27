'use client';

import React, { useState } from 'react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeak = () => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(message.content);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
  };

  return (
    <div
      className={`flex gap-3 my-4 max-w-3xl ${
        isUser ? 'ml-auto justify-end' : 'mr-auto justify-start'
      }`}
    >
      {/* Assistant Avatar */}
      {!isUser && (
        <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 p-[1px] shrink-0 mt-0.5 shadow-md shadow-cyan-950">
          <div className="w-full h-full bg-zinc-950 rounded-[11px] flex items-center justify-center text-cyan-400 font-bold text-xs">
            AI
          </div>
        </div>
      )}

      {/* Message Content Bubble */}
      <div className="group relative max-w-2xl">
        <div
          className={`px-4 py-3 rounded-2xl text-xs sm:text-sm leading-relaxed shadow-sm transition-all ${
            isUser
              ? 'bg-gradient-to-r from-indigo-600 to-cyan-600 text-white rounded-br-none shadow-indigo-950/40'
              : 'bg-zinc-900/80 border border-zinc-800/80 text-zinc-100 rounded-bl-none backdrop-blur-md'
          }`}
        >
          {/* Main Text Output */}
          <div className="whitespace-pre-wrap font-sans">
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-cyan-400 animate-pulse align-middle" />
            )}
          </div>

          {/* Tool Calls Execution Cards */}
          {message.tool_calls && message.tool_calls.length > 0 && (
            <div className="mt-3 space-y-2 border-t border-zinc-800/60 pt-2 font-mono text-[11px]">
              {message.tool_calls.map((tool, idx) => (
                <div key={idx} className="p-2 rounded-lg bg-zinc-950/80 border border-zinc-800 text-cyan-300">
                  <div className="flex items-center gap-1.5 font-semibold text-[10px] text-cyan-400 uppercase tracking-wider">
                    <span>🛠️ Tool Call: {tool.tool}</span>
                  </div>
                  <div className="text-zinc-400 mt-0.5">Args: {tool.argument}</div>
                  <div className="text-emerald-400 mt-1 whitespace-pre-wrap">{tool.output}</div>
                </div>
              ))}
            </div>
          )}

          {/* Action Buttons for Assistant Messages */}
          {!isUser && !message.isStreaming && (
            <div className="opacity-0 group-hover:opacity-100 absolute top-2 right-2 flex items-center gap-1 bg-zinc-900/90 border border-zinc-800 rounded-lg p-1">
              {/* TTS Speaker Button */}
              <button
                onClick={handleSpeak}
                className="p-1 text-zinc-400 hover:text-cyan-400 transition-colors"
                title={isSpeaking ? 'Stop Audio' : 'Listen Text-to-Speech'}
              >
                {isSpeaking ? '🔊' : '🔈'}
              </button>

              {/* Copy Button */}
              <button
                onClick={handleCopy}
                className="p-1 text-zinc-400 hover:text-zinc-200 transition-colors text-[10px]"
                title="Copy text"
              >
                {copied ? '✓ Copied' : '📋'}
              </button>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <div
          className={`text-[10px] text-zinc-400 mt-1 px-1 font-mono ${
            isUser ? 'text-right' : 'text-left'
          }`}
        >
          {message.timestamp}
        </div>
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-300 font-medium text-xs shrink-0 mt-0.5">
          U
        </div>
      )}
    </div>
  );
};
