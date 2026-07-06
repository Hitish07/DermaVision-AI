import { API_BASE_URL } from '../config'
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Trash2, ExternalLink, Calendar, Search, ArrowLeft, Download } from 'lucide-react'

interface HistoryItem {
  id: string
  disease: string
  confidence: number
  risk_level: string
  date: string
  image_type: string
  quality_score: number
}

export default function History() {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [sortField, setSortField] = useState<keyof HistoryItem>('date')
  const [sortAsc, setSortAsc] = useState(false)
  const navigate = useNavigate()


  const getHeaders = () => {
    const token = localStorage.getItem('token')
    return {
      Authorization: `Bearer ${token}`
    }
  }

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/history`, {
        headers: getHeaders()
      })
      const data = await res.json()
      setHistory(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const handleDelete = async (id: string) => {
    try {
      await fetch(`${API_BASE_URL}/api/history/${id}`, {
        method: 'DELETE',
        headers: getHeaders()
      })
      setHistory(prev => prev.filter(item => item.id !== id))
    } catch (err) {
      console.error(err)
    }
  }

  const handleSort = (field: keyof HistoryItem) => {
    if (sortField === field) setSortAsc(!sortAsc)
    else {
      setSortField(field)
      setSortAsc(true)
    }
  }

  const sortedAndFiltered = history
    .filter(item =>
      item.disease.toLowerCase().includes(search.toLowerCase()) ||
      item.id.includes(search)
    )
    .sort((a, b) => {
      let valA = a[sortField]
      let valB = b[sortField]
      if (valA < valB) return sortAsc ? -1 : 1
      if (valA > valB) return sortAsc ? 1 : -1
      return 0
    })

  const getRiskColor = (risk: string) => {
    if (risk === 'High') return 'bg-red-100 text-red-700 border-red-200'
    if (risk === 'Medium') return 'bg-amber-100 text-amber-700 border-amber-200'
    return 'bg-emerald-100 text-emerald-700 border-emerald-200'
  }

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center space-x-4 mb-8 bg-white p-4 rounded-2xl shadow-sm border border-slate-100">
          <button onClick={() => navigate('/')} className="p-2 bg-slate-100 rounded-full hover:bg-slate-200 transition-colors">
            <ArrowLeft className="w-5 h-5 text-slate-600" />
          </button>
          <h1 className="text-2xl font-bold tracking-tight">Clinical Scan History</h1>
        </div>

        <div className="bg-white rounded-3xl shadow-sm border border-slate-100 overflow-hidden">

          <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
            <div className="relative">
              <Search className="w-5 h-5 text-slate-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search disease or ID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 w-72 text-sm shadow-sm"
              />
            </div>
            <div className="text-sm text-slate-500 font-bold uppercase tracking-wider">
              {sortedAndFiltered.length} Scans Found
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-white border-b border-slate-100 text-xs uppercase text-slate-400 font-bold tracking-wider">
                  <th className="p-5 pl-6 cursor-pointer hover:text-slate-600" onClick={() => handleSort('date')}>Date</th>
                  <th className="p-5">Scan ID</th>
                  <th className="p-5 cursor-pointer hover:text-slate-600" onClick={() => handleSort('disease')}>Diagnosis</th>
                  <th className="p-5 cursor-pointer hover:text-slate-600" onClick={() => handleSort('risk_level')}>Risk Level</th>
                  <th className="p-5 cursor-pointer hover:text-slate-600" onClick={() => handleSort('quality_score')}>Quality</th>
                  <th className="p-5 text-right pr-6">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {loading ? (
                  <tr><td colSpan={6} className="text-center p-12 text-slate-500 font-medium">Loading history...</td></tr>
                ) : sortedAndFiltered.length === 0 ? (
                  <tr><td colSpan={6} className="text-center p-12 text-slate-500 font-medium">No scans found.</td></tr>
                ) : (
                  sortedAndFiltered.map(item => (
                    <tr key={item.id} className="hover:bg-slate-50 transition-colors group">
                      <td className="p-5 pl-6">
                        <div className="flex items-center text-sm font-semibold text-slate-700">
                          <Calendar className="w-4 h-4 mr-2 text-slate-400" />
                          {new Date(item.date).toLocaleDateString()}
                        </div>
                      </td>
                      <td className="p-5 font-mono text-xs font-medium text-slate-500">{item.id.substring(0, 8)}</td>
                      <td className="p-5">
                        <div>
                          <p className="font-bold text-slate-800">{item.disease}</p>
                          <p className="text-xs text-slate-500 font-medium">{(item.confidence * 100).toFixed(1)}% Confidence</p>
                        </div>
                      </td>
                      <td className="p-5">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold border ${getRiskColor(item.risk_level)}`}>
                          {item.risk_level}
                        </span>
                      </td>
                      <td className="p-5 font-bold text-slate-700">{item.quality_score.toFixed(1)}%</td>
                      <td className="p-5 pr-6">
                        <div className="flex items-center justify-end space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={() => window.open(`${API_BASE_URL}/api/scan/${item.id}/report`, '_blank')}
                            className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors tooltip"
                            title="Download PDF"
                          >
                            <Download className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => navigate(`/dashboard/${item.id}`)}
                            className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors tooltip"
                            title="View Dashboard"
                          >
                            <ExternalLink className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(item.id)}
                            className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors tooltip"
                            title="Delete Scan"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
