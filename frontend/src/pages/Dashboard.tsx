import { API_BASE_URL } from '../config'
import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ShieldAlert, AlertTriangle, ArrowLeft, Download, Share2, Activity, Loader2 } from 'lucide-react'

export default function Dashboard() {
  const { id } = useParams()
  const navigate = useNavigate()

  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('original')

  const getHeaders = () => {
    const token = localStorage.getItem('token')
    return {
      Authorization: `Bearer ${token}`
    }
  }

  useEffect(() => {
    const fetchScan = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/scan/${id}`, {
          headers: getHeaders()
        })
        if (!res.ok) throw new Error("Failed to fetch scan results")
        const result = await res.json()
        setData(result)
      } catch (err: any) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchScan()

    // Poll for async XAI updates every 3 seconds if status is completed (XAI might be pending)
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/scan/${id}/status`, {
          headers: getHeaders()
        })
        const statusData = await res.json()
        if (statusData.status === 'xai_completed' || statusData.status === 'xai_failed') {
          // Re-fetch full data
          const fullRes = await fetch(`${API_BASE_URL}/api/scan/${id}`, {
            headers: getHeaders()
          })
          const fullResult = await fullRes.json()
          setData(fullResult)
          clearInterval(interval)
        }
      } catch {}
    }, 3000)

    return () => clearInterval(interval)
  }, [id])

  if (loading) {
    return <div className="min-h-screen bg-slate-50 flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>
  }

  if (error || !data) {
    return <div className="min-h-screen bg-slate-50 flex items-center justify-center text-red-500 font-bold">{error || "Data not found"}</div>
  }

  const tabs = [
    { id: 'original', label: 'Original', src: `${API_BASE_URL}${data.image_path}` },
    ...(data.hair_removed_path ? [{ id: 'hair_removed', label: 'Hair Removed', src: `${API_BASE_URL}${data.hair_removed_path}` }] : []),
    ...(data.clahe_path ? [{ id: 'clahe', label: 'CLAHE', src: `${API_BASE_URL}${data.clahe_path}` }] : []),
    ...(data.lesion_mask_path ? [{ id: 'lesion_mask', label: 'Lesion Mask', src: `${API_BASE_URL}${data.lesion_mask_path}` }] : []),
    ...(data.gradcam_path ? [{ id: 'gradcam', label: 'GradCAM', src: `${API_BASE_URL}${data.gradcam_path}` }] : []),
    ...(data.gradcam_plus_plus_path ? [{ id: 'gradcam++', label: 'GradCAM++', src: `${API_BASE_URL}${data.gradcam_plus_plus_path}` }] : []),
    ...(data.shap_path ? [{ id: 'shap', label: 'SHAP', src: `${API_BASE_URL}${data.shap_path}` }] : []),
    ...(data.ig_path ? [{ id: 'ig', label: 'Integrated Gradients', src: `${API_BASE_URL}${data.ig_path}` }] : []),
    ...(data.lime_path ? [{ id: 'lime', label: 'LIME', src: `${API_BASE_URL}${data.lime_path}` }] : []),
  ]

  let riskColor = "bg-emerald-100 text-emerald-700"
  if (data.risk_level === 'High') riskColor = "bg-red-100 text-red-700"
  else if (data.risk_level === 'Medium') riskColor = "bg-amber-100 text-amber-700"

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex justify-between items-center mb-8 bg-white p-4 rounded-2xl shadow-sm border border-slate-100">
          <div className="flex items-center space-x-4">
            <button onClick={() => navigate('/history')} className="p-2 bg-slate-100 rounded-full hover:bg-slate-200 transition">
              <ArrowLeft className="w-5 h-5 text-slate-700" />
            </button>
            <h1 className="text-2xl font-bold tracking-tight">Clinical Analysis Report</h1>
          </div>
          <div className="flex space-x-3">
            <button className="flex items-center space-x-2 px-4 py-2 bg-slate-100 border border-slate-200 rounded-xl shadow-sm text-slate-700 font-medium hover:bg-slate-200 transition">
              <Share2 className="w-4 h-4" /> <span>Share</span>
            </button>
            <button
              onClick={() => window.open(`${API_BASE_URL}/api/scan/${id}/report`, '_blank')}
              className="flex items-center space-x-2 px-5 py-2 bg-blue-600 text-white rounded-xl shadow-md font-semibold hover:bg-blue-700 transition"
            >
              <Download className="w-4 h-4" /> <span>PDF Report</span>
            </button>
          </div>
        </div>

        {/* Smartphone Warning */}
        {data.is_smartphone && (
          <div className="mb-8 p-5 bg-amber-50 border border-amber-200 rounded-2xl flex items-start space-x-4 shadow-sm">
            <AlertTriangle className="w-6 h-6 text-amber-500 shrink-0 mt-1" />
            <div>
              <h4 className="text-sm font-bold text-amber-900 uppercase tracking-wide">Smartphone Image Detected</h4>
              <p className="text-sm text-amber-800 mt-1">
                The AI model was primarily trained using professional dermoscopic images. Predictions on smartphone photographs may have reduced reliability. Results are provided for educational purposes only.
              </p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* LEFT: Image Gallery & Processing Pipeline (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-white p-2 rounded-3xl shadow-sm border border-slate-100 overflow-hidden">
              <div className="aspect-square bg-slate-900 rounded-2xl overflow-hidden relative flex items-center justify-center">
                <img
                  src={tabs.find(t => t.id === activeTab)?.src}
                  alt="Medical Scan"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            {/* Pipeline Tabs */}
            <div className="flex space-x-2 overflow-x-auto pb-2 scrollbar-hide">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-5 py-2.5 rounded-xl whitespace-nowrap text-sm font-bold transition-all ${
                    activeTab === tab.id
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50 hover:border-slate-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Quality Metrics Panel */}
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
              <h3 className="text-sm font-bold text-slate-800 mb-6 flex items-center uppercase tracking-wide">
                <Activity className="w-4 h-4 mr-2 text-blue-500" /> Image Quality Assessment
              </h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold uppercase mb-1">Quality Score</p>
                  <p className="text-xl font-black text-slate-800">{data.quality_score.toFixed(1)}%</p>
                </div>
                <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold uppercase mb-1">Blur</p>
                  <p className="text-sm font-bold text-slate-800">{data.blur_score}</p>
                </div>
                <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold uppercase mb-1">Lighting</p>
                  <p className="text-sm font-bold text-slate-800">{data.brightness}</p>
                </div>
                <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold uppercase mb-1">Noise</p>
                  <p className="text-sm font-bold text-slate-800">{data.noise}</p>
                </div>
                <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold uppercase mb-1">Skin Ratio</p>
                  <p className="text-sm font-bold text-slate-800">{data.skin_ratio.toFixed(1)}%</p>
                </div>
                <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                  <p className="text-xs text-slate-500 font-bold uppercase mb-1">Image Type</p>
                  <p className="text-sm font-bold text-slate-800">{data.image_type}</p>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: Diagnostics (5 cols) */}
          <div className="lg:col-span-5 space-y-6">

            {/* Primary Result */}
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <p className="text-xs text-slate-500 font-bold uppercase tracking-widest mb-2">Primary Diagnosis</p>
                  <h2 className="text-5xl font-black text-slate-900 tracking-tighter">{data.disease}</h2>
                </div>
                <div className={`px-4 py-2 rounded-full font-bold text-sm ${riskColor}`}>
                  {data.risk_level} Risk
                </div>
              </div>

              <div className="mb-8">
                <div className="flex justify-between text-sm font-bold mb-2">
                  <span className="text-slate-600">Model Confidence</span>
                  <span className="text-slate-900">{(data.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                  <div className="bg-blue-600 h-3 rounded-full transition-all duration-1000" style={{ width: `${data.confidence * 100}%` }}></div>
                </div>
              </div>

              {/* Recommendation */}
              <div className="p-5 bg-blue-50/50 rounded-2xl border border-blue-100">
                <h4 className="text-sm font-bold text-blue-900 uppercase tracking-wide mb-2">Clinical Recommendation</h4>
                <p className="text-sm text-blue-800/80 font-medium leading-relaxed">
                  {data.recommendation}
                </p>
              </div>
            </div>

            {/* Differential Diagnosis (Top 3) */}
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
              <h3 className="text-xs font-bold text-slate-500 mb-6 uppercase tracking-widest">Differential Probabilities</h3>
              <div className="space-y-6">
                {data.top_predictions?.map((item: any, idx: number) => (
                  <div key={idx}>
                    <div className="flex justify-between text-sm font-bold mb-2">
                      <span className="text-slate-700">{item.class_name}</span>
                      <span className="text-slate-500">{(item.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-50 rounded-full h-2 overflow-hidden">
                      <div className="bg-slate-300 h-2 rounded-full" style={{ width: `${item.confidence * 100}%` }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Medical Disclaimer */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 border-l-4 border-l-blue-500">
              <div className="flex items-start space-x-3">
                <ShieldAlert className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
                <p className="text-xs text-slate-500 font-medium leading-relaxed">
                  <strong className="text-slate-700">Medical Disclaimer:</strong> DermaVision AI is an experimental clinical decision-support tool.
                  The predictions provided are for educational purposes and do not constitute a medical diagnosis.
                  Always consult a certified healthcare professional.
                </p>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
