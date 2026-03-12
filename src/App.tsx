import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, MessageSquare, Mic, Loader2, CheckCircle2, ArrowRight, Sparkles, User } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import ReactMarkdown from 'react-markdown';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

type View = 'upload' | 'analysis' | 'chat' | 'voice';

interface Message {
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
}

export default function App() {
  const [currentView, setCurrentView] = useState<View>('upload');
  const [image, setImage] = useState<{ url: string; base64: string; mimeType: string } | null>(null);
  const [analysis, setAnalysis] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  
  // Voice state
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [isVoiceConnecting, setIsVoiceConnecting] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Float32Array[]>([]);
  const nextPlayTimeRef = useRef(0);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        const base64Data = base64String.split(',')[1];
        setImage({
          url: URL.createObjectURL(file),
          base64: base64Data,
          mimeType: file.type,
        });
        setCurrentView('analysis');
        analyzeImage(base64Data, file.type);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async (base64Data: string, mimeType: string) => {
    setIsAnalyzing(true);
    try {
      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: {
          parts: [
            {
              inlineData: {
                data: base64Data,
                mimeType: mimeType,
              },
            },
            {
              text: "Analyze this room photo for decluttering and organization. Provide a structured response with: 1. A brief overall assessment. 2. A list of identified clutter or problem areas. 3. A step-by-step action plan to organize the space. 4. Suggested storage solutions or organizational systems. Use markdown formatting.",
            },
          ],
        },
      });
      setAnalysis(response.text || 'No analysis provided.');
    } catch (error) {
      console.error('Analysis error:', error);
      setAnalysis('Failed to analyze the image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;

    const newMessage: Message = { role: 'user', text: chatInput, timestamp: new Date() };
    setChatHistory(prev => [...prev, newMessage]);
    setChatInput('');
    setIsChatting(true);

    try {
      const contents = chatHistory.map(msg => ({
        role: msg.role,
        parts: [{ text: msg.text }]
      }));
      
      contents.push({
        role: 'user',
        parts: [{ text: newMessage.text }]
      });

      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: contents,
        config: {
          systemInstruction: `You are a helpful, encouraging, and expert home organization and decluttering assistant. ${analysis ? `Here is the context of the user's room analysis: ${analysis}` : ''}`,
        }
      });

      setChatHistory(prev => [...prev, { role: 'model', text: response.text || '', timestamp: new Date() }]);
    } catch (error) {
      console.error('Chat error:', error);
      setChatHistory(prev => [...prev, { role: 'model', text: 'Sorry, I encountered an error. Please try again.', timestamp: new Date() }]);
    } finally {
      setIsChatting(false);
    }
  };

  // --- Voice Logic ---
  const startVoiceSession = async () => {
    setIsVoiceConnecting(true);
    try {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const source = audioContextRef.current.createMediaStreamSource(mediaStreamRef.current);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      const sessionPromise = ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-09-2025",
        callbacks: {
          onopen: () => {
            setIsVoiceActive(true);
            setIsVoiceConnecting(false);
          },
          onmessage: (message: LiveServerMessage) => {
            if (message.serverContent?.modelTurn?.parts) {
              for (const part of message.serverContent.modelTurn.parts) {
                if (part.inlineData && part.inlineData.data) {
                  const base64Audio = part.inlineData.data;
                  playAudioChunk(base64Audio);
                }
              }
            }
            if (message.serverContent?.interrupted) {
              audioQueueRef.current = [];
              nextPlayTimeRef.current = audioContextRef.current?.currentTime || 0;
            }
          },
          onerror: (error) => {
            console.error('Voice session error:', error);
            stopVoiceSession();
          },
          onclose: () => {
            stopVoiceSession();
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Aoede" } },
          },
          systemInstruction: `You are a friendly, encouraging home organization and decluttering assistant. Guide the user through cleaning their room, step-by-step. Keep your responses concise and conversational, as if you are right there with them. ${analysis ? `Context of their room: ${analysis}` : ''}`,
        },
      });

      sessionRef.current = sessionPromise;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          let s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Convert to base64
        const uint8Array = new Uint8Array(pcm16.buffer);
        let binary = '';
        for (let i = 0; i < uint8Array.byteLength; i++) {
          binary += String.fromCharCode(uint8Array[i]);
        }
        const base64Data = btoa(binary);

        sessionPromise.then((session) => {
          session.sendRealtimeInput({
            media: {
              mimeType: 'audio/pcm;rate=16000',
              data: base64Data
            }
          });
        });
      };

    } catch (error) {
      console.error('Voice session error:', error);
      setIsVoiceConnecting(false);
      stopVoiceSession();
    }
  };

  const playAudioChunk = (base64Audio: string) => {
    if (!audioContextRef.current) return;
    
    const binaryString = atob(base64Audio);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const pcm16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) {
      float32[i] = pcm16[i] / 32768.0;
    }

    audioQueueRef.current.push(float32);
    scheduleNextAudio();
  };

  const scheduleNextAudio = () => {
    if (!audioContextRef.current || audioQueueRef.current.length === 0) return;
    
    const ctx = audioContextRef.current;
    if (nextPlayTimeRef.current < ctx.currentTime) {
      nextPlayTimeRef.current = ctx.currentTime;
    }

    const chunk = audioQueueRef.current.shift()!;
    const buffer = ctx.createBuffer(1, chunk.length, 24000); // Output sample rate is 24kHz
    buffer.getChannelData(0).set(chunk);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start(nextPlayTimeRef.current);
    
    nextPlayTimeRef.current += buffer.duration;
    
    source.onended = () => {
      if (audioQueueRef.current.length > 0) {
        scheduleNextAudio();
      }
    };
  };

  const stopVoiceSession = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (sessionRef.current) {
      sessionRef.current.then((session: any) => session.close());
      sessionRef.current = null;
    }
    setIsVoiceActive(false);
    audioQueueRef.current = [];
  };

  useEffect(() => {
    return () => {
      stopVoiceSession();
    };
  }, []);

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900 font-sans flex">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-stone-200 flex flex-col p-6">
        <div className="flex items-center gap-3 mb-12">
          <div className="w-10 h-10 bg-emerald-100 text-emerald-600 rounded-xl flex items-center justify-center">
            <Sparkles className="w-6 h-6" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight text-stone-800">Declutter AI</h1>
        </div>

        <nav className="flex flex-col gap-2 flex-1">
          <button 
            onClick={() => setCurrentView('upload')}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${currentView === 'upload' ? 'bg-stone-100 text-stone-900 font-medium' : 'text-stone-500 hover:bg-stone-50 hover:text-stone-700'}`}
          >
            <UploadCloud className="w-5 h-5" />
            Upload Room
          </button>
          <button 
            onClick={() => setCurrentView('analysis')}
            disabled={!image}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${!image ? 'opacity-50 cursor-not-allowed' : currentView === 'analysis' ? 'bg-stone-100 text-stone-900 font-medium' : 'text-stone-500 hover:bg-stone-50 hover:text-stone-700'}`}
          >
            <ImageIcon className="w-5 h-5" />
            Analysis
          </button>
          <button 
            onClick={() => setCurrentView('chat')}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${currentView === 'chat' ? 'bg-stone-100 text-stone-900 font-medium' : 'text-stone-500 hover:bg-stone-50 hover:text-stone-700'}`}
          >
            <MessageSquare className="w-5 h-5" />
            Chat Assistant
          </button>
          <button 
            onClick={() => setCurrentView('voice')}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${currentView === 'voice' ? 'bg-stone-100 text-stone-900 font-medium' : 'text-stone-500 hover:bg-stone-50 hover:text-stone-700'}`}
          >
            <Mic className="w-5 h-5" />
            Live Voice
          </button>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <AnimatePresence mode="wait">
          {currentView === 'upload' && (
            <motion.div 
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex-1 flex items-center justify-center p-8"
            >
              <div className="max-w-xl w-full">
                <div className="text-center mb-8">
                  <h2 className="text-3xl font-semibold tracking-tight mb-3">Let's organize your space</h2>
                  <p className="text-stone-500">Upload a photo of a room that needs decluttering, and our AI will create a personalized action plan.</p>
                </div>
                
                <label className="relative flex flex-col items-center justify-center w-full h-80 border-2 border-stone-300 border-dashed rounded-3xl cursor-pointer bg-white hover:bg-stone-50 transition-colors group">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <div className="w-16 h-16 bg-emerald-50 text-emerald-500 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                      <UploadCloud className="w-8 h-8" />
                    </div>
                    <p className="mb-2 text-sm text-stone-600 font-medium"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                    <p className="text-xs text-stone-500">PNG, JPG or WEBP (Max 10MB)</p>
                  </div>
                  <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                </label>
              </div>
            </motion.div>
          )}

          {currentView === 'analysis' && (
            <motion.div 
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex-1 overflow-y-auto p-8"
            >
              <div className="max-w-4xl mx-auto space-y-8">
                <div className="flex items-center justify-between">
                  <h2 className="text-3xl font-semibold tracking-tight">Room Analysis</h2>
                  <button onClick={() => setCurrentView('upload')} className="text-sm text-stone-500 hover:text-stone-800 font-medium">
                    Upload new photo
                  </button>
                </div>

                {image && (
                  <div className="rounded-3xl overflow-hidden border border-stone-200 shadow-sm bg-white">
                    <img src={image.url} alt="Room" className="w-full h-[400px] object-cover" referrerPolicy="no-referrer" />
                  </div>
                )}

                <div className="bg-white rounded-3xl border border-stone-200 shadow-sm p-8">
                  {isAnalyzing ? (
                    <div className="flex flex-col items-center justify-center py-12 text-stone-500">
                      <Loader2 className="w-10 h-10 animate-spin mb-4 text-emerald-500" />
                      <p className="font-medium text-lg">Analyzing your space...</p>
                      <p className="text-sm mt-2">Identifying clutter and creating an action plan.</p>
                    </div>
                  ) : analysis ? (
                    <div className="prose prose-stone max-w-none prose-headings:font-semibold prose-h3:text-xl prose-h3:mt-8 prose-h3:mb-4 prose-p:text-stone-600 prose-li:text-stone-600">
                      <ReactMarkdown>{analysis}</ReactMarkdown>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-stone-500">
                      <p>Upload a photo to see the analysis.</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {currentView === 'chat' && (
            <motion.div 
              key="chat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex-1 flex flex-col h-full bg-white"
            >
              <div className="p-6 border-b border-stone-200">
                <h2 className="text-2xl font-semibold tracking-tight">Chat Assistant</h2>
                <p className="text-stone-500 text-sm mt-1">Ask follow-up questions about your room or general organization tips.</p>
              </div>
              
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {chatHistory.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-stone-400">
                    <MessageSquare className="w-12 h-12 mb-4 opacity-20" />
                    <p>Start a conversation with your decluttering assistant.</p>
                  </div>
                ) : (
                  chatHistory.map((msg, i) => (
                    <div key={i} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                      <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${msg.role === 'user' ? 'bg-emerald-100 text-emerald-600' : 'bg-stone-200 text-stone-600'}`}>
                        {msg.role === 'user' ? <User className="w-5 h-5" /> : <Sparkles className="w-5 h-5" />}
                      </div>
                      <div className={`flex flex-col max-w-[75%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                        <div className={`flex items-center gap-2 mb-1 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                          <span className="text-sm font-medium text-stone-700">{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                          <span className="text-xs text-stone-400">{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        </div>
                        <div className={`rounded-2xl px-5 py-3.5 ${msg.role === 'user' ? 'bg-emerald-600 text-white rounded-tr-sm' : 'bg-white border border-stone-200 text-stone-800 rounded-tl-sm shadow-sm'}`}>
                          <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'prose-invert' : ''}`}>
                            <ReactMarkdown>{msg.text}</ReactMarkdown>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
                {isChatting && (
                  <div className="flex gap-4 flex-row">
                    <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-stone-200 text-stone-600">
                      <Sparkles className="w-5 h-5" />
                    </div>
                    <div className="flex flex-col items-start">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-stone-700">Assistant</span>
                      </div>
                      <div className="bg-white border border-stone-200 text-stone-800 rounded-2xl rounded-tl-sm shadow-sm px-5 py-4 flex items-center gap-2">
                        <span className="w-2 h-2 bg-stone-400 rounded-full animate-bounce"></span>
                        <span className="w-2 h-2 bg-stone-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                        <span className="w-2 h-2 bg-stone-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="p-6 border-t border-stone-200 bg-stone-50">
                <div className="flex gap-3 max-w-4xl mx-auto">
                  <input 
                    type="text" 
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Ask about storage, specific items, or cleaning methods..."
                    className="flex-1 bg-white border border-stone-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all"
                  />
                  <button 
                    onClick={handleSendMessage}
                    disabled={isChatting || !chatInput.trim()}
                    className="bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-3 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    Send
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {currentView === 'voice' && (
            <motion.div 
              key="voice"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex-1 flex flex-col items-center justify-center p-8 bg-stone-900 text-white"
            >
              <div className="max-w-md w-full text-center space-y-12">
                <div>
                  <h2 className="text-4xl font-semibold tracking-tight mb-4">Live Assistant</h2>
                  <p className="text-stone-400 text-lg">Talk to your AI assistant hands-free while you declutter your space.</p>
                </div>

                <div className="relative flex items-center justify-center h-48">
                  {isVoiceActive && (
                    <>
                      <div className="absolute inset-0 bg-emerald-500/20 rounded-full animate-ping" style={{ animationDuration: '3s' }}></div>
                      <div className="absolute inset-4 bg-emerald-500/20 rounded-full animate-ping" style={{ animationDuration: '2s', animationDelay: '0.5s' }}></div>
                    </>
                  )}
                  <button
                    onClick={isVoiceActive ? stopVoiceSession : startVoiceSession}
                    disabled={isVoiceConnecting}
                    className={`relative z-10 w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${
                      isVoiceActive 
                        ? 'bg-emerald-500 hover:bg-emerald-600 shadow-[0_0_40px_rgba(16,185,129,0.4)]' 
                        : 'bg-stone-800 hover:bg-stone-700 border border-stone-700'
                    }`}
                  >
                    {isVoiceConnecting ? (
                      <Loader2 className="w-12 h-12 animate-spin" />
                    ) : (
                      <Mic className={`w-12 h-12 ${isVoiceActive ? 'text-white' : 'text-stone-400'}`} />
                    )}
                  </button>
                </div>

                <div className="h-12">
                  {isVoiceActive ? (
                    <p className="text-emerald-400 font-medium animate-pulse">Listening and speaking...</p>
                  ) : isVoiceConnecting ? (
                    <p className="text-stone-400">Connecting to assistant...</p>
                  ) : (
                    <p className="text-stone-500">Tap the microphone to start</p>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
