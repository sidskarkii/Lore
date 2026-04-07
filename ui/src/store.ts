import { create } from 'zustand'
import type { Session, Collection, ProviderInfo } from './types'

interface AppState {
  sessions: Session[]
  activeSessionId: string | null
  collections: Collection[]
  providers: ProviderInfo[]
  activeProvider: string | null
  sidebarView: 'chat' | 'ingest' | 'library'

  setSessions: (s: Session[]) => void
  setActiveSessionId: (id: string | null) => void
  setCollections: (c: Collection[]) => void
  setProviders: (p: ProviderInfo[]) => void
  setActiveProvider: (p: string | null) => void
  setSidebarView: (v: 'chat' | 'ingest' | 'library') => void
}

export const useStore = create<AppState>((set) => ({
  sessions: [],
  activeSessionId: null,
  collections: [],
  providers: [],
  activeProvider: null,
  sidebarView: 'chat',

  setSessions: (sessions) => set({ sessions }),
  setActiveSessionId: (activeSessionId) => set({ activeSessionId }),
  setCollections: (collections) => set({ collections }),
  setProviders: (providers) => set({ providers }),
  setActiveProvider: (activeProvider) => set({ activeProvider }),
  setSidebarView: (sidebarView) => set({ sidebarView }),
}))
