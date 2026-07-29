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
  const [liked, setLiked] = useState(message.isLiked || false);
  const [disliked, setDisliked] = useState(message.isDisliked || false);

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
        <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#4F46E5] via-[#7C3AED] to-[#06B6D4] p-[1px] shrink-0 mt-0.5 shadow-lg shadow-[#4F46E5]/30">
          <div className="w-full h-full bg-zinc-950 rounded-[11px] flex items-center justify-center text-[#06B6D4] font-extrabold text-[11px]">
            NA
          </div>
        </div>
      )}

      {/* Message Content Bubble */}
      <div className="group relative max-w-2xl">
        <div
          className={`px-4 py-3 rounded-2xl text-xs sm:text-sm leading-relaxed shadow-sm transition-all ${
            isUser
              ? 'bg-gradient-to-r from-[#4F46E5] to-[#7C3AED] text-white rounded-br-none shadow-indigo-950/40'
              : 'glass-card border border-zinc-800/80 text-zinc-100 rounded-bl-none'
          }`}
        >
          {/* Main Text Output */}
          <div className="whitespace-pre-wrap font-sans">
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-[#06B6D4] animate-pulse align-middle" />
            )}
          </div>

          {/* Tool Calls Execution Cards */}
          {message.tool_calls && message.tool_calls.length > 0 && (
            <div className="mt-3 space-y-2 border-t border-zinc-800/60 pt-2 font-mono text-[11px]">
              {message.tool_calls.map((tool, idx) => (
                <div key={idx} className="p-2.5 rounded-xl bg-zinc-950/90 border border-zinc-800 text-[#06B6D4]">
                  <div className="flex items-center gap-1.5 font-semibold text-[10px] text-[#06B6D4] uppercase tracking-wider">
                    <span>🛠️ Tool Call: {tool.tool}</span>
                  </div>
                  <div className="text-zinc-400 mt-0.5">Args: {tool.argument}</div>
                  <div className="text-emerald-400 mt-1 whitespace-pre-wrap">{tool.output}</div>
                </div>
              ))}
            </div>
          )}

          {/* Action Toolbar for Assistant Messages */}
          {!isUser && !message.isStreaming && (
            <div className="mt-2.5 pt-2 border-t border-zinc-800/40 flex items-center justify-between text-[11px] text-zinc-400">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    setLiked(!liked);
                    if (disliked) setDisliked(false);
                  }}
                  className={`hover:text-emerald-400 transition-colors ${liked ? 'text-emerald-400 font-bold' : ''}`}
                  title="Helpful response"
                >
                  👍
                </button>
                <button
                  onClick={() => {
                    setDisliked(!disliked);
                    if (liked) setLiked(false);
                  }}
                  className={`hover:text-rose-400 transition-colors ${disliked ? 'text-rose-400 font-bold' : ''}`}
                  title="Unhelpful response"
                >
                  👎
                </button>
                <button
                  onClick={handleSpeak}
                  className="hover:text-[#06B6D4] transition-colors ml-1"
                  title="Text-to-Speech"
                >
                  {isSpeaking ? '🔊 Stop' : '🔈 Listen'}
                </button>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-[9px] font-mono text-zinc-500">
                  {message.tokensCount || Math.ceil(message.content.length / 4)} tokens
                </span>
                <button
                  onClick={handleCopy}
                  className="hover:text-zinc-200 transition-colors text-[10px]"
                  title="Copy text"
                >
                  {copied ? '✓ Copied' : '📋 Copy'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <div
          className={`text-[10px] text-zinc-500 mt-1 px-1 font-mono ${
            isUser ? 'text-right' : 'text-left'
          }`}
        >
          {message.timestamp}
        </div>
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-300 font-bold text-xs shrink-0 mt-0.5">
          U
        </div>
      )}
    </div>
  );
};

