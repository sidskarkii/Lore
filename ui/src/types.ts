export interface SearchResult {
  text: string
  collection: string
  collection_display: string
  episode_num: number
  episode_title: string
  timestamp: string
  start_sec: number
  end_sec: number
  url: string
  topic: string
  subtopic: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SearchResult[]
  streaming?: boolean
}

export interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
  provider: string | null
  model: string | null
  message_count: number
}

export interface Collection {
  collection: string
  collection_display: string
  topic: string
  subtopic: string
  episode_count: number
}

export interface ProviderInfo {
  name: string
  display_name: string
  installed: boolean
  authenticated: boolean
  version: string | null
  is_active: boolean
  free_model_count: number
}

export interface IngestJob {
  type: 'folder' | 'youtube'
  label: string
  status: 'pending' | 'running' | 'done' | 'error'
  message: string
}
