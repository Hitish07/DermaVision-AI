import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Brain, ScanSearch, Target, Activity } from 'lucide-react';

export default function Documentation() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      
      {/* Header */}
      <div className="bg-slate-900 text-white py-12 px-8 mb-10">
        <div className="max-w-4xl mx-auto">
          <button 
            onClick={() => navigate('/')} 
            className="flex items-center space-x-2 text-slate-400 hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Home</span>
          </button>
          <h1 className="text-4xl font-bold mb-4">How DermaVision AI Works</h1>
          <p className="text-lg text-slate-300">A simplified guide to understanding our clinical AI pipeline.</p>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-8 space-y-12">
        
        {/* Section 1 */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
          <div className="flex items-center space-x-4 mb-6">
            <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center">
              <ScanSearch className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold">1. Image Quality & Preprocessing</h2>
          </div>
          <p className="text-slate-600 leading-relaxed mb-4">
            Before the AI ever sees your image, we need to make sure it's clean and readable. 
            Skin images often have hair, reflections, or poor lighting. Our system automatically:
          </p>
          <ul className="list-disc pl-6 text-slate-600 space-y-2">
            <li><strong>Hair Removal:</strong> Uses an algorithm called "DullRazor" to digitally erase thick body hair that might block the lesion.</li>
            <li><strong>Contrast Enhancement:</strong> Uses "CLAHE" to balance the lighting and make the colors pop correctly.</li>
            <li><strong>Quality Check:</strong> Detects if the image is too blurry or dark, and warns you to take a better photo.</li>
          </ul>
        </section>

        {/* Section 2 */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
          <div className="flex items-center space-x-4 mb-6">
            <div className="w-12 h-12 bg-purple-100 text-purple-600 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold">2. The Deep Learning Engine (MobileNetV2)</h2>
          </div>
          <p className="text-slate-600 leading-relaxed mb-4">
            DermaVision AI uses a Convolutional Neural Network (CNN) called <strong>MobileNetV2</strong>. 
            Think of a CNN as a virtual brain that has looked at tens of thousands of skin lesions and learned to recognize the subtle patterns of different skin diseases.
          </p>
          <p className="text-slate-600 leading-relaxed">
            When you upload an image, MobileNetV2 scans it and outputs a percentage score for 7 different classes of skin lesions, categorizing it as Benign (safe) or Malignant (dangerous).
          </p>
        </section>

        {/* Section 3 */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
          <div className="flex items-center space-x-4 mb-6">
            <div className="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-xl flex items-center justify-center">
              <Target className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold">3. Explainable AI (Heatmaps)</h2>
          </div>
          <p className="text-slate-600 leading-relaxed mb-4">
            In medicine, we cannot blindly trust a black-box AI. That is why we provide Explainable AI (XAI) heatmaps:
          </p>
          <ul className="list-disc pl-6 text-slate-600 space-y-2">
            <li><strong>Grad-CAM:</strong> Creates a colorful heatmap over your image. Red areas show exactly which parts of the skin the AI stared at to make its decision.</li>
            <li>If the AI correctly identifies a melanoma, but the heatmap shows it was looking at a dark spot in the background instead of the actual mole, doctors know not to trust that specific prediction!</li>
          </ul>
        </section>

        {/* Section 4 */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
          <div className="flex items-center space-x-4 mb-6">
            <div className="w-12 h-12 bg-amber-100 text-amber-600 rounded-xl flex items-center justify-center">
              <Activity className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-bold">4. Clinical Disclaimer</h2>
          </div>
          <p className="text-slate-600 leading-relaxed">
            While highly accurate, this platform is strictly for <strong>educational and research purposes</strong>. 
            AI should never replace a biopsy or a professional dermatologist. The model struggles with smartphone images and darker skin tones due to biases in the training data, and a high confidence score does not equal a medical diagnosis.
          </p>
        </section>

        <div className="text-center mt-12">
          <button 
            onClick={() => navigate('/upload')}
            className="px-8 py-4 bg-blue-600 text-white rounded-full font-bold text-lg shadow-lg shadow-blue-600/30 hover:-translate-y-1 transition-transform"
          >
            Try the AI Pipeline Now
          </button>
        </div>

      </div>
    </div>
  );
}
