import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldCheck, Activity, Cpu, ArrowRight, Microscope, Scan, ShieldAlert } from 'lucide-react';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 selection:bg-blue-100">
      
      {/* Navigation */}
      <nav className="flex justify-between items-center px-8 py-6 max-w-7xl mx-auto">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">DermaVision AI</span>
        </div>
        <div className="flex items-center space-x-6">
          <a href="#features" className="text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors">Features</a>
          <a href="#clinical-validation" className="text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors">Clinical Validation</a>
          <button onClick={() => navigate('/history')} className="text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors">History</button>
          <button 
            onClick={() => navigate('/upload')}
            className="px-5 py-2 bg-slate-900 text-white text-sm font-semibold rounded-full shadow-md hover:bg-blue-600 transition-colors"
          >
            New Scan
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative px-8 pt-20 pb-32 max-w-7xl mx-auto overflow-hidden">
        <div className="max-w-3xl relative z-10">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-xs font-bold uppercase tracking-widest mb-6"
          >
            <ShieldCheck className="w-4 h-4" />
            <span>Clinical Support Edition</span>
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold tracking-tighter leading-[1.1] mb-6"
          >
            Intelligent Skin Lesion <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">Analysis Platform</span>
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-lg md:text-xl text-slate-500 mb-10 max-w-2xl leading-relaxed"
          >
            A multi-stage AI pipeline providing robust clinical decision support. 
            Featuring real-time image validation, quality assessment, and transparent explainable AI.
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-4"
          >
            <button 
              onClick={() => navigate('/upload')}
              className="px-8 py-4 bg-slate-900 text-white rounded-2xl font-bold text-lg shadow-xl shadow-slate-900/20 hover:bg-blue-600 hover:-translate-y-1 transition-all flex items-center space-x-2"
            >
              <span>Start AI Analysis</span>
              <ArrowRight className="w-5 h-5" />
            </button>
            <button 
              onClick={() => navigate('/docs')}
              className="px-8 py-4 bg-white text-slate-700 border border-slate-200 rounded-2xl font-bold text-lg hover:bg-slate-50 transition-colors"
            >
              Read Documentation
            </button>
          </motion.div>
        </div>

        {/* Decorative Background Elements */}
        <div className="absolute top-20 right-0 w-[500px] h-[500px] bg-gradient-to-br from-blue-100/50 to-indigo-100/50 rounded-full blur-3xl -z-10 opacity-70"></div>
      </section>

      {/* Trust Badges */}
      <section className="border-y border-slate-200 bg-white py-12">
        <div className="max-w-7xl mx-auto px-8 grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100 flex flex-col items-center text-center group hover:shadow-md transition-shadow">
              <div className="w-20 h-20 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Microscope className="w-10 h-10" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">MobileNetV2 Engine</h3>
              <p className="text-slate-600">Lightweight & highly optimized CNN for fast clinical inference.</p>
            </div>
            
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100 flex flex-col items-center text-center group hover:shadow-md transition-shadow">
              <div className="w-20 h-20 bg-emerald-50 text-emerald-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Scan className="w-10 h-10" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">Explainable AI</h3>
              <p className="text-slate-600">Grad-CAM, SHAP & LIME heatmaps provide transparent decision tracking.</p>
            </div>
            
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100 flex flex-col items-center text-center group hover:shadow-md transition-shadow">
              <div className="w-20 h-20 bg-purple-50 text-purple-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Activity className="w-10 h-10" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">Multi-Stage Pipeline</h3>
              <p className="text-slate-600">Validation, Quality Assessment, and Lesion Detection before inference.</p>
            </div>
            
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100 flex flex-col items-center text-center group hover:shadow-md transition-shadow">
              <div className="w-20 h-20 bg-amber-50 text-amber-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <ShieldAlert className="w-10 h-10" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">Clinical Focus</h3>
              <p className="text-slate-600">Mapped Risk Assessment prevents over-reliance on softmax confidence.</p>
            </div>
        </div>
      </section>
      
      {/* Medical Disclaimer */}
      <section className="bg-slate-900 py-16 text-white text-center px-8">
         <div className="max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold mb-4 flex items-center justify-center"><ShieldCheck className="w-6 h-6 mr-2 text-blue-400"/> Medical Disclaimer</h2>
            <p className="text-slate-300 leading-relaxed max-w-2xl mx-auto">
              DermaVision AI is a research and educational platform designed for computer vision exploration in dermatology. 
              It is not FDA approved and should <strong>never</strong> be used as a substitute for professional medical diagnosis, advice, or treatment.
            </p>
         </div>
      </section>
      
    </div>
  );
};

export default LandingPage;
