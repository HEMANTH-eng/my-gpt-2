'use client';

import React, { useState } from 'react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
      <div className="group relative">
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

          {/* Copy Button for Assistant Messages */}
          {!isUser && !message.isStreaming && (
            <button
              onClick={handleCopy}
              className="opacity-0 group-hover:opacity-100 absolute top-2 right-2 p-1.5 rounded-lg bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-all text-[10px] flex items-center gap-1"
            >
              {copied ? (
                <>
                  <svg className="w-3 h-3 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Copied!
                </>
              ) : (
                <>
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  Copy
                </>
              )}
            </button>
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
