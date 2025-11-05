import { Avatar, AvatarFallback } from "@/components/ui/avatar"

export default function ChatMessage({ role, content, typing }) {
  const isUser = role === "user"
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`flex items-end ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        <Avatar className={`w-8 h-8 ${isUser ? "ml-2" : "mr-2"}`}>
          <AvatarFallback>{isUser ? "U" : "B"}</AvatarFallback>
        </Avatar>
        <div
          className={`px-4 py-2 rounded-lg whitespace-pre-wrap ${
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          }`}
        >
          {content || (typing ? "…" : "")}
        </div>
      </div>
    </div>
  )
}
