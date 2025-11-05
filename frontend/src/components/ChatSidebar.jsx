import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Trash2 } from "lucide-react"

export default function ChatSidebar({ activeChatId, onSelect, className }) {
  const [chats, setChats] = useState([])
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState("")

  const load = async () => {
    setErr("")
    setLoading(true)
    try {
      const { data } = await api.get("/api/chats")
      setChats(data?.chats || [])
    } catch (e) {
      setErr("Could not load chats.")
    } finally {
      setLoading(false)
    }
  }

  const createChat = async (isJournal = false) => {
    setErr("")
    try {
      const { data } = await api.post("/api/chats", { is_journal: isJournal })
      await load()
      onSelect?.(data.chat_id)
    } catch {
      setErr("Could not create chat.")
    }
  }

  const deleteChat = async (id) => {
    setErr("")
    try {
      await api.delete(`/api/chats/${id}`)
      await load()
      // If the deleted chat was active, parent can choose another after onSelect(null)
      if (activeChatId === id) onSelect?.(null)
    } catch {
      setErr("Could not delete chat.")
    }
  }

  useEffect(() => {
    // If user is logged in, load chats
    if (localStorage.getItem("token") || localStorage.getItem("jwt_token")) load()
  }, [])

  return (
    <div className={`w-full md:w-64 border rounded-lg p-3 flex flex-col gap-3 ${className || ""}`}>
      <div className="flex items-center justify-between">
        <div className="font-semibold">Your Chats</div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => createChat(true)}>Journal</Button>
          <Button size="sm" onClick={() => createChat(false)}>New</Button>
        </div>
      </div>

      {loading && <div className="text-sm text-muted-foreground">Loading…</div>}
      {err && <div className="text-sm text-red-600">{err}</div>}

      <div className="flex flex-col gap-2 overflow-auto">
        {chats.map((c) => (
          <div
            key={c.chat_id}
            className={`group flex items-center justify-between text-left text-sm px-3 py-2 rounded-md border hover:bg-accent ${
              activeChatId === c.chat_id ? "bg-accent" : ""
            }`}
          >
            <button
              onClick={() => onSelect?.(c.chat_id)}
              className="flex-1 text-left"
              title={c.title || (c.is_journal ? "Journal" : "New conversation")}
            >
              <div className="font-medium truncate">
                {c.title || (c.is_journal ? "Journal" : "New conversation")}
              </div>
              <div className="text-xs text-muted-foreground">
                {c.is_journal ? "Journal" : "Chat"} • {new Date(c.created_at).toLocaleDateString()}
              </div>
            </button>

            {!c.is_journal && (
              <button
                aria-label="Delete chat"
                title="Delete chat"
                onClick={() => {
                  if (confirm("Delete this chat and its data? This cannot be undone.")) {
                    deleteChat(c.chat_id)
                  }
                }}
                className="opacity-0 group-hover:opacity-100 transition-opacity ml-2 p-1 rounded hover:bg-destructive/10"
              >
                <Trash2 className="h-4 w-4 text-destructive" />
              </button>
            )}
          </div>
        ))}
        {!chats.length && <div className="text-sm text-muted-foreground">No chats yet.</div>}
      </div>
    </div>
  )
}
