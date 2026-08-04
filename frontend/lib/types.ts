export type Regulation = "ai_act" | "gdpr";

export interface Citation {
  index: number;
  regulation: Regulation;
  document: string;
  /**
   * The ref of the unit the chunk came from, rendered verbatim in the citation
   * badge. Two shapes, not one: `"Art. 6"` and `"Annex III"` — the AI Act's
   * high-risk list lives in Annex III, and Art. 6(2) only points at it. Do not
   * strip a `"Art. "` prefix or parse a number out of this; `url` already
   * carries the matching EUR-Lex anchor (`#art_6` / `#anx_III`), built
   * server-side in app/retrieval/citations.py. The field name predates
   * annexes and matches the wire payload.
   */
  article: string;
  heading: string;
  url: string;
  snippet: string;
  /**
   * Raw RRF fusion value — a sum of 1/(60 + rank) terms, so a top hit scores
   * around 0.016. Not rendered: it reads as "1.6% confident" and is not even a
   * ranking a reader could use, because one panel mixes results from several
   * searches whose fusion scores are not comparable. Kept on the type because
   * the API sends it and it is worth having in the network tab when debugging
   * retrieval.
   */
  score: number;
}

export interface Grounding {
  grounded: boolean;
  issues: string[];
}

export interface ApprovalRequest {
  draft: string;
  citations: Citation[];
  /**
   * Brackets the backend refused to turn into links, with the reason. The
   * reviewer decides on the text the user will receive, so the defects in that
   * text travel with it.
   */
  citationIssues: string[];
}

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  citations?: Citation[];
  grounding?: Grounding;
  /** Report run: tokens are suppressed server-side until the approval decision. */
  drafting?: boolean;
  /**
   * From `done.citation_issues`. Not folded into `grounding.issues`: that is the
   * factual-faithfulness verdict, and a bracket naming the wrong article is a
   * citation-resolution defect. Rendering them together would make an answer
   * with a formatting problem look factually unsound, and vice versa.
   */
  citationIssues?: string[];
}

export interface Session {
  token: string;
  role: string;
  email: string;
}

export interface PendingApproval {
  id: string;
  thread_id: string;
  payload: {
    draft?: string;
    citations?: Citation[];
    /** Wire shape (snake_case): this payload is the interrupt value, stored verbatim. */
    citation_issues?: string[];
  };
  requested_at: string;
}
