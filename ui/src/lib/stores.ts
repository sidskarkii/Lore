import { writable } from 'svelte/store'
import type { Session, Collection, ProviderInfo } from './types'

export const sessions = writable<Session[]>([])
export const activeSessionId = writable<string | null>(null)
export const collections = writable<Collection[]>([])
export const activeProvider = writable<string | null>(null)
export const providers = writable<ProviderInfo[]>([])
export const sidebarView = writable<'chat' | 'ingest' | 'library'>('chat')
