import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { UploadCloud, CheckCircle, Loader2, AlertCircle } from 'lucide-react'

export default function Upload() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [status, setStatus] = useState<string>('idle') // idle, uploading, validating, quality, detection, predicting, success, error
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  
  const navigate = useNavigate()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const f = acceptedFiles[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setStatus('idle')
    setErrorMessage(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    maxFiles: 1
  })

  const handleAnalyze = async () => {
    if (!file) return
    setErrorMessage(null)

    try {
      // 1. Upload
      setStatus('uploading')
      const formData = new FormData()
      formData.append('file', file)
      const uploadRes = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      })
      if (!uploadRes.ok) throw new Error("Failed to upload image")
      const { scan_id } = await uploadRes.json()

      // 2. Validate
      setStatus('validating')
      const valRes = await fetch(`http://localhost:8000/api/validate/${scan_id}`, { method: 'POST' })
      const valData = await valRes.json()
      if (!valData.is_valid) {
        setStatus('error')
        setErrorMessage(`Validation Failed: ${valData.reason}`)
        return
      }

      // 3. Quality
      setStatus('quality')
      const qualRes = await fetch(`http://localhost:8000/api/quality/${scan_id}`, { method: 'POST' })
      const qualData = await qualRes.json()
      if (qualData.quality_score < 30) {
        setStatus('error')
        setErrorMessage("Image quality is too poor for analysis (Blurry/Dark). Please retake.")
        return
      }

      // 4. Detection
      setStatus('detection')
      const detRes = await fetch(`http://localhost:8000/api/detect/${scan_id}`, { method: 'POST' })
      const detData = await detRes.json()
      // We no longer halt the pipeline if lesion_detected is false. We allow it to proceed to prediction anyway.

      // 5. Prediction
      setStatus('predicting')
      const predRes = await fetch(`http://localhost:8000/api/predict/${scan_id}`, { method: 'POST' })
      if (!predRes.ok) throw new Error("Failed during prediction")
      
      setStatus('success')
      setTimeout(() => navigate(`/dashboard/${scan_id}`), 800)

    } catch (err: any) {
      setStatus('error')
      setErrorMessage(err.message || "Network error connecting to AI server.")
    }
  }

  const steps = [
    { id: 'uploading', label: 'Uploading Securely' },
    { id: 'validating', label: 'Intelligent Image Validation' },
    { id: 'quality', label: 'Quality Assessment' },
    { id: 'detection', label: 'Morphological Lesion Detection' },
    { id: 'predicting', label: 'AI Inference & Explainability' }
  ]

  const getStepStatus = (stepId: string) => {
    const currentIndex = steps.findIndex(s => s.id === status)
    const stepIndex = steps.findIndex(s => s.id === stepId)
    
    if (status === 'success') return 'complete'
    if (status === 'error' && currentIndex === stepIndex) return 'error'
    if (currentIndex > stepIndex) return 'complete'
    if (currentIndex === stepIndex) return 'active'
    return 'pending'
  }

  return (
    <div className="min-h-screen bg-slate-50 p-8 flex flex-col items-center justify-center font-sans text-slate-900">
      
      <div className="max-w-4xl w-full bg-white/70 backdrop-blur-xl rounded-3xl shadow-xl overflow-hidden flex flex-col md:flex-row border border-white">
        
        {/* Left Side: Dropzone */}
        <div className="md:w-1/2 p-8 md:p-12 bg-gradient-to-br from-blue-50/50 to-indigo-50/50 border-r border-slate-100 flex flex-col items-center justify-center">
          <h2 className="text-3xl font-bold text-slate-800 mb-2 text-center tracking-tight">Upload Lesion</h2>
          <p className="text-slate-500 mb-8 text-center text-sm">Secure, encrypted clinical pipeline</p>
          
          <div {...getRootProps()} className={`w-full p-8 border-2 border-dashed rounded-3xl flex flex-col items-center justify-center cursor-pointer transition-all ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50/50'}`}>
            <input {...getInputProps()} />
            {preview ? (
              <img src={preview} alt="Preview" className="w-48 h-48 object-cover rounded-2xl shadow-md mb-4 ring-4 ring-white" />
            ) : (
              <div className="w-20 h-20 bg-white rounded-full shadow-sm flex items-center justify-center mb-4">
                 <UploadCloud className="w-10 h-10 text-blue-500" />
              </div>
            )}
            <p className="text-sm text-slate-500 text-center font-medium max-w-[200px]">
              {isDragActive ? "Drop image here..." : preview ? "Click or drag to replace image" : "Drag & drop skin lesion image here"}
            </p>
          </div>
          
          <button 
            onClick={handleAnalyze}
            disabled={!file || (status !== 'idle' && status !== 'error')}
            className={`mt-8 w-full py-4 rounded-full font-bold text-lg shadow-lg transition-all flex items-center justify-center space-x-2 ${
              !file || (status !== 'idle' && status !== 'error')
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-blue-600/30'
            }`}
          >
            {status !== 'idle' && status !== 'error' && status !== 'success' ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> <span>Analyzing...</span></>
            ) : (
              <span>Start Clinical Analysis</span>
            )}
          </button>
        </div>

        {/* Right Side: Animated Stepper Pipeline */}
        <div className="md:w-1/2 p-8 md:p-12 bg-white flex flex-col justify-center">
          <h3 className="text-xl font-bold text-slate-800 mb-8 tracking-tight">Multi-Stage Pipeline</h3>
          
          <div className="space-y-6">
            {steps.map((step, idx) => {
              const stepStatus = getStepStatus(step.id)
              return (
                <div key={step.id} className="flex items-center space-x-4">
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 border-4 border-white shadow-sm transition-colors duration-500`}
                       style={{
                         backgroundColor: stepStatus === 'complete' ? '#22c55e' : stepStatus === 'active' ? '#3b82f6' : stepStatus === 'error' ? '#ef4444' : '#f1f5f9',
                         color: stepStatus !== 'pending' ? 'white' : '#94a3b8'
                       }}>
                    {stepStatus === 'complete' ? <CheckCircle className="w-8 h-8" /> : 
                     stepStatus === 'active' ? <Loader2 className="w-8 h-8 animate-spin" /> :
                     stepStatus === 'error' ? <AlertCircle className="w-8 h-8" /> :
                     <span className="font-bold text-lg">{idx + 1}</span>}
                  </div>
                  <div className={`p-3 rounded-xl flex-grow transition-all duration-300 ${stepStatus === 'active' ? 'bg-blue-50 border border-blue-100 shadow-sm' : 'bg-transparent'}`}>
                     <span className={`font-semibold text-sm ${stepStatus === 'active' ? 'text-blue-700' : stepStatus === 'error' ? 'text-red-600' : stepStatus === 'complete' ? 'text-slate-800' : 'text-slate-400'}`}>
                      {step.label}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>

          {errorMessage && (
            <div className="mt-8 p-4 bg-red-50 text-red-600 text-sm font-semibold rounded-2xl border border-red-100 flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}
          
        </div>
      </div>
    </div>
  )
}
