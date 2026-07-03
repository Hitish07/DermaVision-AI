import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import History from './pages/History'
import Documentation from './pages/Documentation'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/dashboard/:id" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/docs" element={<Documentation />} />
      </Routes>
    </Router>
  )
}

export default App
