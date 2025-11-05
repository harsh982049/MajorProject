import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import Picker from "@emoji-mart/react"
import { Smile, Trash2, Plus } from "lucide-react"
import { animate, stagger } from "animejs" // ✨ anime.js v4

import { api, API_BASE, getToken } from "@/lib/api"
import ChatMessage from "@/components/ChatMessage"
import ChatSidebar from "@/components/ChatSidebar"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import Navbar from "@/components/Navbar"
import Footer from "@/components/Footer"

const ACTIVE_KEY = "active_chat_id"

// ---------- Helpers ----------
function getOrCreateSessionId() {
  let id = localStorage.getItem("chatbot_session_id")
  if (!id) {
    const bytes = new Uint8Array(16)
    crypto.getRandomValues(bytes)
    bytes[6] = (bytes[6] & 0x0f) | 0x40
    bytes[8] = (bytes[8] & 0x3f) | 0x80
    id = [...bytes].map((b, i) => {
      const s = b.toString(16).padStart(2, "0")
      return [4, 6, 8, 10].includes(i) ? "-" + s : s
    }).join("")
    localStorage.setItem("chatbot_session_id", id)
  }
  return id
}

// Stream SSE via fetch so we can send Authorization header (EventSource cannot)
async function streamSSE({ url, headers = {}, onToken, onError, onDone, signal }) {
  try {
    const res = await fetch(url, { headers, signal })
    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status}`)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder("utf-8")
    let buf = ""

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx).trim()
        buf = buf.slice(idx + 2)
        if (!chunk) continue
        const line = chunk.split("\n").find(l => l.startsWith("data: "))
        if (!line) continue
        const payloadStr = line.slice(6)
        if (payloadStr === "[DONE]") {
          onDone?.()
          return
        }
        try {
          const payload = JSON.parse(payloadStr)
          if (payload.error) onError?.(new Error(payload.error))
          const token = payload.token || ""
          if (token) onToken?.(token)
        } catch {
          /* ignore malformed lines */
        }
      }
    }
    onDone?.()
  } catch (e) {
    if (e?.name === "AbortError") return
    onError?.(e)
  }
}

export default function PanicSOSChatbot() {
  const [messages, setMessages] = useState([
    { id: 1, role: "ai", content: "Hello! I'm here to help you through any panic or anxiety you might be experiencing. How are you feeling right now?" },
  ])

  const [inputMessage, setInputMessage] = useState("")
  const [showEmojiPicker, setShowEmojiPicker] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [working, setWorking] = useState(false) // guards create/delete spam
  const [error, setError] = useState("")
  const [activeChatId, setActiveChatId] = useState(() => {
    const cached = localStorage.getItem(ACTIVE_KEY)
    return cached ? Number(cached) : null
  })
  const [isAuthed, setIsAuthed] = useState(!!getToken())

  const scrollAreaRef = useRef(null)
  const emojiPickerRef = useRef(null)
  const sessionIdRef = useRef(getOrCreateSessionId())
  const typingMsgIdRef = useRef(null)
  const sseAbortRef = useRef(null) // abort controller for SSE
  const createdOnceGuardRef = useRef(false) // prevent double-create loops

  // ---------- Auto-scroll ----------
  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]')
    if (!viewport) return
    const toBottom = () => { viewport.scrollTop = viewport.scrollHeight }
    const id = setTimeout(toBottom, 30)
    return () => clearTimeout(id)
  }, [messages])

  // ---------- Close emoji on outside / Esc ----------
  useEffect(() => {
    if (!showEmojiPicker) return
    const onClick = (e) => {
      if (emojiPickerRef.current && !emojiPickerRef.current.contains(e.target)) {
        setShowEmojiPicker(false)
      }
    }
    const onKey = (e) => e.key === "Escape" && setShowEmojiPicker(false)
    document.addEventListener("mousedown", onClick)
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("mousedown", onClick)
      document.removeEventListener("keydown", onKey)
    }
  }, [showEmojiPicker])

  // ---------- SSE cleanup on unmount ----------
  useEffect(() => {
    return () => {
      if (sseAbortRef.current) sseAbortRef.current.abort()
    }
  }, [])

  // ---------- Anime.js entrance micro-animations ----------
  const animateIn = useCallback(() => {
    // Pop new rows in
    animate(".chat-row", {
     opacity: [0, 1],
     translateY: [6, 0],
     duration: 280,
     easing: "easeOutQuad",
     delay: stagger(25) // ✅ v4 API
   })
  }, [])

  useEffect(() => {
    animateIn()
  }, [messages, animateIn])

  // ---------- Load/ensure chat (JWT) or stick to session mode ----------
  useEffect(() => {
    async function ensureChat() {
      if (!isAuthed) return
      try {
        const { data: list } = await api.get("/api/chats")
        const chats = list?.chats || []

        const loadMessages = async (chatId) => {
          const { data: hist } = await api.get(`/api/chats/${chatId}/messages?limit=100`)
          const loaded = (hist?.messages || []).map((m) => ({
            id: m.id,
            role: m.role === "human" ? "user" : "ai",
            content: m.content,
          }))
          setMessages(loaded.length ? loaded : [
            { id: 1, role: "ai", content: "How are you feeling right now?" },
          ])
        }

        // 1) If cached id exists and is valid → reuse it
        if (activeChatId && chats.some(c => c.chat_id === activeChatId)) {
          localStorage.setItem(ACTIVE_KEY, String(activeChatId))
          await loadMessages(activeChatId)
          return
        }

        // 2) Otherwise: REUSE latest normal chat (even if not empty) to avoid duplicates on refresh
        const normals = chats.filter(c => !c.is_journal)
        if (normals.length) {
          const candidate = normals[0] // assume newest first by backend
          setActiveChatId(candidate.chat_id)
          localStorage.setItem(ACTIVE_KEY, String(candidate.chat_id))
          await loadMessages(candidate.chat_id)
          return
        }

        // 3) Only if none exist at all, create ONE new chat (guard against double-run)
        if (!createdOnceGuardRef.current) {
          createdOnceGuardRef.current = true
          const { data } = await api.post("/api/chats", { is_journal: false })
          setActiveChatId(data.chat_id)
          localStorage.setItem(ACTIVE_KEY, String(data.chat_id))
          setMessages([
            { id: 1, role: "ai", content: "New conversation started. How are you feeling right now?" },
          ])
        }
      } catch {
        // Fallback to guest mode if anything fails
        setIsAuthed(false)
      }
    }
    ensureChat()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Persist active chat id whenever it changes
  useEffect(() => {
    if (activeChatId) {
      localStorage.setItem(ACTIVE_KEY, String(activeChatId))
    }
  }, [activeChatId])

  // ---------- Compose helpers ----------
  const addTypingBubble = () => {
    const id = Date.now() + 1
    typingMsgIdRef.current = id
    setMessages((prev) => [...prev, { id, role: "ai", content: "", typing: true }])

    // ✨ micro “pulse” while typing
    requestAnimationFrame(() => {
      animate(`#typing-${id}`, {
        opacity: [{ value: 0.5, duration: 200 }, { value: 1, duration: 200 }],
        loop: true,
        easing: "linear"
      })
    })
    return id
  }
  const appendToken = (id, token) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, content: (m.content || "") + token } : m)))
  }
  const finalizeTyping = (id) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, typing: false } : m)))
  }

  // ---------- Send ----------
  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (isSending) return
    setError("")
    const text = inputMessage.trim()
    if (!text) return

    // push user message locally
    const userMsg = { id: Date.now(), role: "user", content: text }
    setMessages((prev) => [...prev, userMsg])
    setInputMessage("")
    setIsSending(true)

    const typingId = addTypingBubble()

    // ripple effect on send button (anime)
    animate(".send-btn", {
      scale: [{ value: 0.98, duration: 60 }, { value: 1, duration: 120 }],
      easing: "easeOutQuad"
    })

    // setup abortable stream
    if (sseAbortRef.current) sseAbortRef.current.abort()
    const controller = new AbortController()
    sseAbortRef.current = controller

    try {
      if (isAuthed && activeChatId) {
        const qs = new URLSearchParams({ chat_id: String(activeChatId), message: text })
        const url = `${API_BASE}/api/chatbot/stream?${qs.toString()}`
        const token = getToken() || ""

        await streamSSE({
          url,
          headers: token ? { Authorization: `Bearer ${token}` } : {},
          onToken: (tok) => appendToken(typingId, tok),
          onError: (err) => setError(err.message || "Stream error"),
          onDone: () => finalizeTyping(typingId),
          signal: controller.signal
        })
      } else {
        const qs = new URLSearchParams({ session_id: sessionIdRef.current, message: text })
        const url = `${API_BASE}/api/chatbot/stream?${qs.toString()}`
        await streamSSE({
          url,
          onToken: (tok) => appendToken(typingId, tok),
          onError: (err) => setError(err.message || "Stream error"),
          onDone: () => finalizeTyping(typingId),
          signal: controller.signal
        })
      }
    } catch (err) {
      if (err?.name !== "AbortError") {
        setError(err?.message || "Sorry, I’m having trouble responding right now.")
        finalizeTyping(typingId)
      }
    } finally {
      setIsSending(false)
    }
  }

  // ---------- Header actions ----------
  const handleNewChat = async () => {
    if (!isAuthed || working) return
    setWorking(true)
    setError("")
    try {
      const { data } = await api.post("/api/chats", { is_journal: false })
      setActiveChatId(data.chat_id)
      localStorage.setItem(ACTIVE_KEY, String(data.chat_id))
      setMessages([{ id: 1, role: "ai", content: "New conversation started. How are you feeling right now?" }])
    } catch {
      setError("Could not start a new chat.")
    } finally {
      setWorking(false)
    }
  }

  const handleDeleteChat = async () => {
    if (!isAuthed || !activeChatId || working) return
    if (!confirm("Delete this chat and its data? This cannot be undone.")) return
    setWorking(true)
    setError("")
    try {
      await api.delete(`/api/chats/${activeChatId}`)

      // find another chat to show (reuse latest) or create exactly one new chat
      const { data: list } = await api.get("/api/chats")
      const normals = (list?.chats || []).filter(c => !c.is_journal && c.chat_id !== activeChatId)

      if (normals.length) {
        const nextId = normals[0].chat_id
        setActiveChatId(nextId)
        localStorage.setItem(ACTIVE_KEY, String(nextId))
        const { data: hist } = await api.get(`/api/chats/${nextId}/messages?limit=100`)
        const loaded = (hist?.messages || []).map((m) => ({
          id: m.id, role: m.role === "human" ? "user" : "ai", content: m.content
        }))
        setMessages(loaded.length ? loaded : [{ id: 1, role: "ai", content: "How are you feeling right now?" }])
      } else {
        const { data } = await api.post("/api/chats", { is_journal: false })
        setActiveChatId(data.chat_id)
        localStorage.setItem(ACTIVE_KEY, String(data.chat_id))
        setMessages([{ id: 1, role: "ai", content: "New conversation started. How are you feeling right now?" }])
      }
    } catch {
      setError("Could not delete chat.")
    } finally {
      setWorking(false)
    }
  }

  // ---------- Reset (guest only) ----------
  const resetConversation = async () => {
    if (isAuthed && activeChatId) {
      await handleNewChat()
      return
    }
    setError("")
    setIsSending(true)
    try {
      await api.post("/api/chatbot/reset", { session_id: sessionIdRef.current })
      setMessages([{ id: 1, role: "ai", content: "Hello! I'm here to help. How are you feeling right now?" }])
    } catch {
      setError("Could not reset the conversation. Please try again.")
    } finally {
      setIsSending(false)
    }
  }

  // ---------- UI ----------
  return (
    <>
      <Navbar />
      <motion.div
        className="container mx-auto px-4 py-8 max-w-5xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="grid grid-cols-1 md:grid-cols-[16rem_1fr] gap-4">
          {/* Sidebar only when logged in */}
          {isAuthed ? (
            <ChatSidebar
              activeChatId={activeChatId}
              onSelect={async (id) => {
                setError("")
                if (!id) {
                  setActiveChatId(null)
                  localStorage.removeItem(ACTIVE_KEY)
                  return
                }
                if (id === activeChatId) return
                setActiveChatId(id)
                localStorage.setItem(ACTIVE_KEY, String(id))
                try {
                  const { data } = await api.get(`/api/chats/${id}/messages?limit=100`)
                  const loaded = (data?.messages || []).map((m) => ({
                    id: m.id, role: m.role === "human" ? "user" : "ai", content: m.content
                  }))
                  setMessages(loaded.length ? loaded : [
                    { id: 1, role: "ai", content: "How are you feeling right now?" },
                  ])
                } catch {
                  setError("Could not load messages.")
                }
              }}
              className="h-full"
            />
          ) : (
            <div className="hidden md:block" />
          )}

          {/* Chat card */}
          <Card className="w-full">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-2xl font-bold text-primary">
                Panic SOS Chatbot
                {!isAuthed && <span className="ml-2 text-xs text-muted-foreground">(guest)</span>}
              </CardTitle>

              <div className="flex gap-2">
                {isAuthed && (
                  <>
                    <Button
                      variant="outline" size="sm"
                      onClick={handleNewChat} title="New chat"
                      disabled={working}
                      className="relative overflow-hidden"
                      onMouseDown={() => animate(".new-btn", { scale: 0.98, duration: 80 })}
                      onMouseUp={() => animate(".new-btn", { scale: 1, duration: 120 })}
                    >
                      <span className="new-btn flex items-center">
                        <Plus className="h-4 w-4 mr-1" /> New
                      </span>
                    </Button>

                    {activeChatId && (
                      <Button
                        variant="destructive" size="sm"
                        onClick={handleDeleteChat} title="Delete this chat"
                        disabled={working}
                        className="relative overflow-hidden"
                        onMouseDown={() => animate(".delete-btn", { scale: 0.98, duration: 80 })}
                        onMouseUp={() => animate(".delete-btn", { scale: 1, duration: 120 })}
                      >
                        <span className="delete-btn flex items-center">
                          <Trash2 className="h-4 w-4 mr-1" /> Delete
                        </span>
                      </Button>
                    )}
                  </>
                )}
                <Button variant="secondary" size="sm" onClick={resetConversation} disabled={isSending || working}>
                  {isAuthed ? "Reset (new)" : "Reset"}
                </Button>
              </div>
            </CardHeader>

            <CardContent>
              <ScrollArea className="h-[420px] pr-4" ref={scrollAreaRef}>
                <AnimatePresence initial={false}>
                  {messages.map((m) => (
                    <motion.div
                      key={m.id}
                      className="chat-row" // for anime.js selector
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.25 }}
                      id={m.typing ? `typing-${m.id}` : undefined}
                    >
                      <ChatMessage role={m.role} content={m.content} typing={m.typing} />
                    </motion.div>
                  ))}
                </AnimatePresence>
              </ScrollArea>
              {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
            </CardContent>

            <CardFooter>
              <form onSubmit={handleSendMessage} className="flex w-full gap-2">
                <div className="relative">
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() => setShowEmojiPicker((v) => !v)}
                    disabled={isSending}
                    aria-label="Insert emoji"
                    title="Insert emoji"
                  >
                    <Smile className="h-5 w-5" />
                  </Button>
                  {showEmojiPicker && (
                    <div
                      ref={emojiPickerRef}
                      className="absolute bottom-full mb-2 left-0 bg-white shadow-lg rounded-lg z-50"
                    >
                      <Picker onEmojiSelect={(emoji) => {
                        setInputMessage((prev) => prev + emoji.native)
                        setShowEmojiPicker(false)
                      }} />
                    </div>
                  )}
                </div>

                <Input
                  type="text"
                  placeholder="Type your message here…"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  className="flex-grow"
                  disabled={isSending}
                />

                <Button type="submit" className="send-btn" disabled={isSending || !inputMessage.trim()}>
                  {isSending ? "Sending…" : "Send"}
                </Button>
              </form>
            </CardFooter>
          </Card>
        </div>
      </motion.div>
      <Footer />
    </>
  )
}
