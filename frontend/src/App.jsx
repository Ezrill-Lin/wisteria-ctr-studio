import { useState } from 'react'
import TextAdPredictor from './components/TextAdPredictor'
import ImageAdPredictor from './components/ImageAdPredictor'
import HistoryPage from './components/HistoryPage'
import './App.css'

function App() {
  const [activeView, setActiveView] = useState('prediction') // 'prediction' or 'history'
  const [activeTab, setActiveTab] = useState('text')
  const [predictionHistory, setPredictionHistory] = useState([])

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Wisteria CTR Studio
              </h1>
              <p className="mt-1 text-sm text-gray-600">
                AI-powered Click-Through Rate prediction using synthetic personas
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full">v3.1.0</span>
              <span className="hidden sm:inline">Powered by Gemini 2.5 Flash Lite</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Main Navigation - Prediction vs History */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6 p-2">
          <div className="flex gap-2">
            <button
              onClick={() => setActiveView('prediction')}
              className={`flex-1 px-6 py-3 rounded-lg text-sm font-medium transition-all ${
                activeView === 'prediction'
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-md'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <span className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Prediction
              </span>
            </button>
            <button
              onClick={() => setActiveView('history')}
              className={`flex-1 px-6 py-3 rounded-lg text-sm font-medium transition-all ${
                activeView === 'history'
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-md'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <span className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                History
                {predictionHistory.length > 0 && (
                  <span className="ml-1 px-2 py-0.5 text-xs bg-white/20 rounded-full">
                    {predictionHistory.length}
                  </span>
                )}
              </span>
            </button>
          </div>
        </div>

        {/* Content based on active view */}
        {activeView === 'prediction' ? (
          <>
            {/* Tab Navigation for Text/Image */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
              <div className="flex border-b border-gray-200">
                <button
                  onClick={() => setActiveTab('text')}
                  className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                    activeTab === 'text'
                      ? 'text-purple-600 border-b-2 border-purple-600 bg-purple-50'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Text Ad Prediction
                  </span>
                </button>
                <button
                  onClick={() => setActiveTab('image')}
                  className={`flex-1 px-6 py-4 text-sm font-medium transition-colors ${
                    activeTab === 'image'
                      ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Image Ad Prediction
                  </span>
                </button>
              </div>
            </div>

            {/* Tab Content */}
            <div className="transition-all duration-300">
              {activeTab === 'text' ? (
                <TextAdPredictor 
                  predictionHistory={predictionHistory}
                  setPredictionHistory={setPredictionHistory}
                />
              ) : (
                <ImageAdPredictor 
                  predictionHistory={predictionHistory}
                  setPredictionHistory={setPredictionHistory}
                />
              )}
            </div>
          </>
        ) : (
          <HistoryPage 
            history={predictionHistory} 
            onClose={() => setActiveView('prediction')} 
          />
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 pb-8 text-center text-sm text-gray-500">
        <p>Wisteria CTR Studio • Using SiliconSampling v2 synthetic personas</p>
        <p className="mt-1">Model: gemini-2.5-flash-lite (unified for decisions & analysis)</p>
      </footer>
    </div>
  )
}

export default App
