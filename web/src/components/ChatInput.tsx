'use client';

import React, { useState, KeyboardEvent, useRef } from 'react';
import { ModelConfig } from '../types';

interface ChatInputProps {
  onSendMessage: (text: string, fileId?: string) => void;
  onFileUpload: (file: File) => Promise<string | null>;
  isLoading: boolean;
  config: ModelConfig;
}

const PROMPT_SUGGESTIONS = [
  'Explain Self-Attention simply',
  '[TOOL: search("PyTorch 2.0 release features")]',
  '[TOOL: python("math.sqrt(144) + 50")]',
];

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  onFileUpload,
  isLoading,
  config,
}) => {
  const [input, setInput] = useState('');
  const [attachedFileId, setAttachedFileId] = useState<string | null>(null);
  const [attachedFileName, setAttachedFileName] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    if ((!input.trim() && !attachedFileId) || isLoading) return;
    onSendMessage(input.trim(), attachedFileId || undefined);
    setInput('');
    setAttachedFileId(null);
    setAttachedFileName(null);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Voice Input Speech-to-Text (STT)
  const handleVoiceInput = () => {
    if (typeof window === 'undefined') return;

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please try Chrome/Edge.');
      return;
    }

    if (isListening) {
      setIsListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput((prev) => (prev ? `${prev} ${transcript}` : transcript));
    };

    recognition.start();
  };

  // File Upload Handler
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const fileId = await onFileUpload(file);
    setIsUploading(false);

    if (fileId) {
      setAttachedFileId(fileId);
      setAttachedFileName(file.name);
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

      {/* File Attachment Pill */}
      {attachedFileName && (
        <div className="mb-2 inline-flex items-center gap-2 bg-cyan-950/80 border border-cyan-800/60 text-cyan-300 text-xs px-3 py-1 rounded-lg">
          <span>📄 Attached: {attachedFileName}</span>
          <button
            onClick={() => {
              setAttachedFileId(null);
              setAttachedFileName(null);
            }}
            className="hover:text-rose-400 font-bold ml-1"
          >
            ✕
          </button>
        </div>
      )}

      {/* Input Box Container */}
      <div className="relative bg-zinc-900/90 border border-zinc-800 focus-within:border-cyan-500/60 rounded-2xl p-3 shadow-xl backdrop-blur-md transition-all">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask MyGPT anything... (Press Enter to send, Shift+Enter for new line)"
          disabled={isLoading}
          rows={2}
          className="w-full bg-transparent text-xs sm:text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none resize-none pr-24"
        />

        {/* Bottom Controls Bar */}
        <div className="flex items-center justify-between pt-2 border-t border-zinc-800/60">
          <div className="flex items-center gap-2">
            {/* File Upload Button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || isUploading}
              className="p-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-cyan-400 transition-colors text-xs flex items-center gap-1"
              title="Attach PDF, DOCX, or TXT file"
            >
              {isUploading ? '⏳' : '📎 Upload File'}
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Voice Input Mic Button (STT) */}
            <button
              onClick={handleVoiceInput}
              disabled={isLoading}
              className={`p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1 ${
                isListening
                  ? 'bg-rose-950 text-rose-400 border border-rose-800 animate-pulse'
                  : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-cyan-400'
              }`}
              title="Voice Input (Speech-to-Text)"
            >
              {isListening ? '🎙️ Listening...' : '🎤 Voice'}
            </button>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={(!input.trim() && !attachedFileId) || isLoading}
            className="p-2 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md shadow-cyan-950 font-medium text-xs flex items-center gap-1.5"
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <span>Send</span>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
