import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import { LibraryView } from './components/LibraryView'
import { IngestView } from './components/IngestView'
import { useStore } from './store'
import './App.css'

export default function App() {
  const sidebarView = useStore(s => s.sidebarView)

  return (
    <div className="app">
      <Sidebar />
      <main className="main">
        {sidebarView === 'chat' && <ChatView />}
        {sidebarView === 'ingest' && <IngestView />}
        {sidebarView === 'library' && <LibraryView />}
      </main>
    </div>
  )
}
