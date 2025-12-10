import { useState } from 'react'
import Results from './Results'

// Production API URL (Cloud Run with CORS enabled)
const API_URL = 'https://wisteria-ctr-studio-azlh47c4pq-uc.a.run.app'
// For local development, use: const API_URL = 'http://localhost:8080'

function TextAdPredictor() {
  const [formData, setFormData] = useState({
    ad_text: '',
    population_size: 100,
    ad_platform: 'facebook',
    persona_version: 'v2',
    persona_strategy: 'random',
    concurrent_requests: 20,
    include_persona_details: false
  })
  
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/predict/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : type === 'number' ? parseInt(value) : value
    }))
  }

  return (
    <div className="space-y-6">
      {/* Form */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Ad Text Input */}
          <div>
            <label htmlFor="ad_text" className="block text-sm font-medium text-gray-700 mb-2">
              Advertisement Text *
            </label>
            <div className="relative">
              <div
                contentEditable
                suppressContentEditableWarning
                onInput={(e) => {
                  const text = e.currentTarget.textContent;
                  setFormData(prev => ({ ...prev, ad_text: text }));
                }}
                onFocus={(e) => {
                  // Remove placeholder styling when focused
                  if (e.currentTarget.textContent === 'e.g., Special 0% APR credit card offer for travel rewards') {
                    e.currentTarget.textContent = '';
                    e.currentTarget.classList.remove('text-gray-400');
                    e.currentTarget.classList.add('text-gray-900');
                  }
                }}
                onBlur={(e) => {
                  // Restore placeholder if empty
                  if (!e.currentTarget.textContent.trim()) {
                    e.currentTarget.textContent = 'e.g., Special 0% APR credit card offer for travel rewards';
                    e.currentTarget.classList.add('text-gray-400');
                    e.currentTarget.classList.remove('text-gray-900');
                    setFormData(prev => ({ ...prev, ad_text: '' }));
                  }
                }}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all min-h-[100px] text-gray-400 outline-none"
                style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}
              >
                e.g., Special 0% APR credit card offer for travel rewards
              </div>
              <input
                type="hidden"
                name="ad_text"
                value={formData.ad_text}
                required
              />
            </div>
            <p className="mt-1 text-xs text-gray-500">Enter the advertisement text you want to test</p>
          </div>

          {/* Configuration Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Population Size */}
            <div>
              <label htmlFor="population_size" className="block text-sm font-medium text-gray-700 mb-2">
                Population Size
              </label>
              <input
                type="number"
                id="population_size"
                name="population_size"
                min="1"
                max="10000"
                value={formData.population_size}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
              <p className="mt-1 text-xs text-gray-500">Number of personas to evaluate (1-10,000)</p>
            </div>

            {/* Platform */}
            <div>
              <label htmlFor="ad_platform" className="block text-sm font-medium text-gray-700 mb-2">
                Platform
              </label>
              <select
                id="ad_platform"
                name="ad_platform"
                value={formData.ad_platform}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="facebook">Facebook</option>
                <option value="instagram">Instagram</option>
                <option value="tiktok">TikTok</option>
                <option value="youtube">YouTube</option>
                <option value="amazon">Amazon</option>
              </select>
            </div>

            {/* Persona Version */}
            <div>
              <label htmlFor="persona_version" className="block text-sm font-medium text-gray-700 mb-2">
                <span className="flex items-center gap-1.5">
                  Persona Version
                  <div className="group relative">
                    <svg className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-help" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                    <div className="absolute left-0 bottom-full mb-2 hidden group-hover:block w-64 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg z-10">
                      <div className="font-semibold mb-1">Choose persona depth:</div>
                      <div className="space-y-1">
                        <div><span className="font-medium">v2 (recommended):</span> Includes detailed behavioral tendencies and core beliefs for more realistic predictions</div>
                        <div><span className="font-medium">v1:</span> Uses simpler demographic-only personas</div>
                      </div>
                      <div className="absolute left-4 top-full w-2 h-2 bg-gray-900 transform rotate-45 -mt-1"></div>
                    </div>
                  </div>
                </span>
              </label>
              <select
                id="persona_version"
                name="persona_version"
                value={formData.persona_version}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="v1">v1 - Basic personas</option>
                <option value="v2">v2 - Enhanced (recommended)</option>
              </select>
            </div>

            {/* Persona Strategy */}
            <div>
              <label htmlFor="persona_strategy" className="block text-sm font-medium text-gray-700 mb-2">
                <span className="flex items-center gap-1.5">
                  Persona Strategy
                  <div className="group relative">
                    <svg className="w-4 h-4 text-gray-400 hover:text-gray-600 cursor-help" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                    <div className="absolute left-0 bottom-full mb-2 hidden group-hover:block w-72 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg z-10">
                      <div className="font-semibold mb-1">How personalities are assigned:</div>
                      <div className="space-y-1">
                        <div><span className="font-medium">Random:</span> Baseline with randomly assigned personality traits across demographics</div>
                        <div><span className="font-medium">WPP:</span> Uses research-based personality distributions correlated by country/demographics</div>
                        <div><span className="font-medium">IPIP:</span> Matches personality traits based on demographic patterns</div>
                      </div>
                      <div className="absolute left-4 top-full w-2 h-2 bg-gray-900 transform rotate-45 -mt-1"></div>
                    </div>
                  </div>
                </span>
              </label>
              <select
                id="persona_strategy"
                name="persona_strategy"
                value={formData.persona_strategy}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="random">Random - Baseline</option>
                <option value="wpp">WPP - Survey-based</option>
                <option value="ipip">IPIP - Demographic matching</option>
              </select>
            </div>
          </div>

          {/* Advanced Options */}
          <div className="pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  name="include_persona_details"
                  checked={formData.include_persona_details}
                  onChange={handleChange}
                  className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                />
                <span className="ml-2 text-sm text-gray-700">Include individual persona responses</span>
              </label>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !formData.ad_text}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Analyzing {formData.population_size} personas...
              </span>
            ) : (
              'Predict CTR'
            )}
          </button>
        </form>
      </div>

      {/* Loading Progress */}
      {loading && (
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="relative">
                <svg className="animate-spin h-8 w-8 text-purple-600" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-gray-900">Processing CTR Prediction</h3>
                <p className="text-xs text-gray-600 mt-0.5">Evaluating {formData.population_size} synthetic personas with AI</p>
              </div>
            </div>
          </div>
          
          {/* Progress Steps */}
          <div className="space-y-2 mt-4">
            <div className="flex items-center gap-3 text-sm">
              <div className="flex-shrink-0 w-5 h-5 rounded-full bg-purple-600 flex items-center justify-center">
                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span className="text-gray-700">Loading synthetic personas ({formData.persona_version} - {formData.persona_strategy})</span>
            </div>
            
            <div className="flex items-center gap-3 text-sm">
              <div className="flex-shrink-0 w-5 h-5 rounded-full border-2 border-purple-600 animate-pulse"></div>
              <span className="text-gray-700">Simulating user decisions with AI ({formData.concurrent_requests} concurrent)</span>
            </div>
            
            <div className="flex items-center gap-3 text-sm">
              <div className="flex-shrink-0 w-5 h-5 rounded-full border-2 border-gray-300"></div>
              <span className="text-gray-400">Analyzing results and generating insights</span>
            </div>
          </div>
          
          <div className="mt-4 text-xs text-gray-500 text-center">
            Estimated time: {Math.ceil(formData.population_size / formData.concurrent_requests)}–{Math.ceil(formData.population_size / formData.concurrent_requests) + 5} seconds
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-red-600 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && <Results data={result} adType="text" adContent={formData.ad_text} />}
    </div>
  )
}

export default TextAdPredictor
