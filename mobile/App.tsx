import React, { useState } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'assistant', content: 'Hello! I am MyGPT Mobile. How can I assist you today?' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'chat' | 'agents' | 'settings'>('chat');

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('http://10.0.2.2:8000/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setMessages([
          ...newMessages,
          { id: (Date.now() + 1).toString(), role: 'assistant', content: data.message?.content || 'No response.' },
        ]);
      } else {
        setMessages([
          ...newMessages,
          { id: (Date.now() + 1).toString(), role: 'assistant', content: `Received: ${userMsg.content}` },
        ]);
      }
    } catch {
      setMessages([
        ...newMessages,
        { id: (Date.now() + 1).toString(), role: 'assistant', content: `Mobile Offline Mode: Echo '${userMsg.content}'` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#09090b" />

      {/* Header Bar */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>MyGPT Mobile</Text>
        <Text style={styles.headerBadge}>v2.0 iOS/Android</Text>
      </View>

      {/* Main Content Area */}
      <View style={styles.content}>
        {activeTab === 'chat' && (
          <ScrollView style={styles.chatArea} contentContainerStyle={styles.chatContainer}>
            {messages.map((m) => (
              <View
                key={m.id}
                style={[
                  styles.messageBubble,
                  m.role === 'user' ? styles.userBubble : styles.assistantBubble,
                ]}
              >
                <Text style={styles.messageText}>{m.content}</Text>
              </View>
            ))}
          </ScrollView>
        )}

        {activeTab === 'agents' && (
          <View style={styles.placeholderBox}>
            <Text style={styles.placeholderTitle}>🤖 Mobile AI Agents</Text>
            <Text style={styles.placeholderSub}>Coding, Research, Email, Calendar, Browser, Data Analysis</Text>
          </View>
        )}

        {activeTab === 'settings' && (
          <View style={styles.placeholderBox}>
            <Text style={styles.placeholderTitle}>⚙️ Settings</Text>
            <Text style={styles.placeholderSub}>Server URL: http://10.0.2.2:8000</Text>
          </View>
        )}
      </View>

      {/* Bottom Chat Input Bar */}
      {activeTab === 'chat' && (
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder="Type a message..."
            placeholderTextColor="#71717a"
          />
          <TouchableOpacity style={styles.sendButton} onPress={handleSend} disabled={loading}>
            <Text style={styles.sendButtonText}>{loading ? '...' : 'Send'}</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Navigation Tab Bar */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === 'chat' && styles.activeTabItem]}
          onPress={() => setActiveTab('chat')}
        >
          <Text style={styles.tabText}>💬 Chat</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === 'agents' && styles.activeTabItem]}
          onPress={() => setActiveTab('agents')}
        >
          <Text style={styles.tabText}>🤖 Agents</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === 'settings' && styles.activeTabItem]}
          onPress={() => setActiveTab('settings')}
        >
          <Text style={styles.tabText}>⚙️ Settings</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#09090b' },
  header: { height: 56, paddingHorizontal: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', borderBottomWidth: 1, borderBottomColor: '#27272a' },
  headerTitle: { fontSize: 16, fontWeight: '600', color: '#f4f4f5' },
  headerBadge: { fontSize: 10, color: '#38bdf8', backgroundColor: '#082f49', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 10 },
  content: { flex: 1 },
  chatArea: { flex: 1 },
  chatContainer: { padding: 16 },
  messageBubble: { padding: 12, borderRadius: 16, marginBottom: 12, maxWidth: '80%' },
  userBubble: { alignSelf: 'flex-end', backgroundColor: '#2563eb' },
  assistantBubble: { alignSelf: 'flex-start', backgroundColor: '#18181b', borderWidth: 1, borderColor: '#27272a' },
  messageText: { fontSize: 14, color: '#f4f4f5', lineHeight: 20 },
  inputContainer: { flexDirection: 'row', padding: 12, borderTopWidth: 1, borderTopColor: '#27272a', backgroundColor: '#09090b' },
  input: { flex: 1, backgroundColor: '#18181b', borderWidth: 1, borderColor: '#27272a', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 8, color: '#f4f4f5', fontSize: 14 },
  sendButton: { marginLeft: 8, backgroundColor: '#38bdf8', paddingHorizontal: 16, justifyContent: 'center', borderRadius: 20 },
  sendButtonText: { color: '#09090b', fontWeight: '600', fontSize: 14 },
  tabBar: { height: 56, flexDirection: 'row', borderTopWidth: 1, borderTopColor: '#27272a', backgroundColor: '#18181b' },
  tabItem: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  activeTabItem: { borderTopWidth: 2, borderTopColor: '#38bdf8' },
  tabText: { fontSize: 12, color: '#a1a1aa', fontWeight: '500' },
  placeholderBox: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  placeholderTitle: { fontSize: 18, fontWeight: '600', color: '#f4f4f5', marginBottom: 8 },
  placeholderSub: { fontSize: 12, color: '#71717a', textAlign: 'center' },
});
