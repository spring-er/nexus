// ── User ──────────────────────────────────────────────

export interface User {
  id: string;
  email: string;
  full_name: string;
  avatar_url?: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export type UserRole = "admin" | "attorney" | "paralegal" | "viewer";

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface AuthResponse {
  user: User;
  tokens: AuthTokens;
}

// ── Document ──────────────────────────────────────────

export interface Document {
  id: string;
  title: string;
  file_name: string;
  file_type: DocumentFileType;
  file_size: number;
  file_url: string;
  status: DocumentStatus;
  uploaded_by: string;
  tags: string[];
  metadata: DocumentMetadata;
  created_at: string;
  updated_at: string;
}

export type DocumentFileType = "pdf" | "docx" | "txt" | "doc";

export type DocumentStatus =
  | "uploading"
  | "processing"
  | "ready"
  | "error"
  | "archived";

export interface DocumentMetadata {
  page_count?: number;
  word_count?: number;
  language?: string;
  author?: string;
  created_date?: string;
}

export interface DocumentUploadRequest {
  file: File;
  title?: string;
  tags?: string[];
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface DocumentFilters {
  search?: string;
  status?: DocumentStatus;
  file_type?: DocumentFileType;
  tags?: string[];
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: "asc" | "desc";
}

// ── Analysis ──────────────────────────────────────────

export interface Analysis {
  id: string;
  document_id: string;
  type: AnalysisType;
  status: AnalysisStatus;
  result?: AnalysisResult;
  error_message?: string;
  requested_by: string;
  created_at: string;
  completed_at?: string;
}

export type AnalysisType =
  | "summary"
  | "key_clauses"
  | "risk_assessment"
  | "compliance_check"
  | "entity_extraction"
  | "comparison";

export type AnalysisStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed";

export interface AnalysisResult {
  summary?: string;
  key_clauses?: KeyClause[];
  risks?: RiskItem[];
  compliance_issues?: ComplianceIssue[];
  entities?: ExtractedEntity[];
  confidence_score?: number;
}

export interface KeyClause {
  id: string;
  title: string;
  content: string;
  page_number?: number;
  category: string;
  importance: "high" | "medium" | "low";
}

export interface RiskItem {
  id: string;
  title: string;
  description: string;
  severity: "critical" | "high" | "medium" | "low";
  recommendation: string;
  page_reference?: number;
}

export interface ComplianceIssue {
  id: string;
  regulation: string;
  description: string;
  status: "compliant" | "non_compliant" | "needs_review";
  details: string;
}

export interface ExtractedEntity {
  id: string;
  type: "person" | "organization" | "date" | "monetary_value" | "location";
  value: string;
  context: string;
  page_reference?: number;
}

export interface AnalysisRequest {
  document_id: string;
  type: AnalysisType;
}

// ── API ───────────────────────────────────────────────

export interface ApiError {
  detail: string;
  status_code: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}
