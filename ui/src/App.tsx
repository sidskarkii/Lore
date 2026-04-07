import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import './App.css'

export default function App() {
  return (
    <div className="app">
      <Sidebar />
      <main className="main">
        <ChatView />
      </main>
    </div>
  )
}
