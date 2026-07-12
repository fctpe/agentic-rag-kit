export type Regulation = "ai_act" | "gdpr";

export interface Citation {
  index: number;
  regulation: Regulation;
  document: string;
  article: string;
  heading: string;
  url: string;
  snippet: string;
  score: number;
}

export interface Grounding {
  grounded: boolean;
  issues: string[];
}

export interface ApprovalRequest {
  draft: string;
  citations: Citation[];
}

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  citations?: Citation[];
  grounding?: Grounding;
}

export interface Session {
  token: string;
  role: string;
  email: string;
}

export interface PendingApproval {
  id: string;
  thread_id: string;
  payload: { draft?: string; citations?: Citation[] };
  requested_at: string;
}
