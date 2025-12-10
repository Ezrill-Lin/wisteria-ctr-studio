import { useState } from 'react'
import Results from './Results'

// Production API URL (Cloud Run with CORS enabled)
const API_URL = 'https://wisteria-ctr-studio-azlh47c4pq-uc.a.run.app'
// For local development, use: const API_URL = 'http://localhost:8080'

function ImageAdPredictor({ predictionHistory, setPredictionHistory }) {
  const [formData, setFormData] = useState({
    image_url: '',
    population_size: 100,
    ad_platform: 'facebook',
    persona_version: 'v2',
    persona_strategy: 'random',
    concurrent_requests: 20,
    include_persona_details: true
  })
  
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [uploadMethod, setUploadMethod] = useState('url') // 'url' or 'file'
  const [previewUrl, setPreviewUrl] = useState('')

  const downloadHistory = () => {
    if (predictionHistory.length === 0) {
      alert('No prediction history to download')
      return
    }

    const dataStr = JSON.stringify(predictionHistory, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `wisteria-ctr-history-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file')
      return
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError('Image must be smaller than 5MB')
      return
    }

    setError(null)
    
    // Read file and convert to base64
    const reader = new FileReader()
    reader.onloadend = () => {
      const base64String = reader.result
      setFormData(prev => ({ ...prev, image_url: base64String }))
      setPreviewUrl(base64String)
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/predict/image`, {
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
      
      // Add to history with timestamp and full data
      const historyEntry = {
        timestamp: new Date().toISOString(),
        type: 'image',
        adContent: previewUrl,
        parameters: {
          population_size: formData.population_size,
          ad_platform: formData.ad_platform,
          persona_version: formData.persona_version,
          persona_strategy: formData.persona_strategy
        },
        results: {
          ctr: data.ctr ?? data.estimated_ctr ?? 0,
          clicks: data.total_clicks ?? data.clicks ?? 0,
          population: data.total_personas ?? data.population_size ?? 0
        },
        fullData: data // Store complete response for detail view
      }
      setPredictionHistory(prev => [...prev, historyEntry])
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

  const handleUrlChange = (e) => {
    const url = e.target.value
    setFormData(prev => ({ ...prev, image_url: url }))
    setPreviewUrl(url)
  }

  return (
    <div className="space-y-6">
      {/* Form */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Upload Method Toggle */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Image Source *
            </label>
            <div className="flex gap-4">
              <button
                type="button"
                onClick={() => setUploadMethod('url')}
                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                  uploadMethod === 'url'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Image URL
              </button>
              <button
                type="button"
                onClick={() => setUploadMethod('file')}
                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                  uploadMethod === 'file'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Upload File
              </button>
            </div>
          </div>

          {/* Image URL Input */}
          {uploadMethod === 'url' && (
            <div>
              <label htmlFor="image_url" className="block text-sm font-medium text-gray-700 mb-2">
                Image URL
              </label>
              <input
                type="url"
                id="image_url"
                name="image_url"
                required
                value={formData.image_url}
                onChange={handleUrlChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                placeholder="https://example.com/ad-image.jpg"
              />
              <p className="mt-1 text-xs text-gray-500">Enter a publicly accessible image URL</p>
            </div>
          )}

          {/* File Upload Input */}
          {uploadMethod === 'file' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Image
              </label>
              <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg hover:border-purple-400 transition-colors">
                <div className="space-y-2 text-center">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <div className="text-sm text-gray-600">
                    <label
                      htmlFor="file-upload"
                      className="relative cursor-pointer bg-white rounded-md font-medium text-purple-600 hover:text-purple-500"
                    >
                      <span>Upload a file</span>
                      <input
                        id="file-upload"
                        name="file-upload"
                        type="file"
                        accept="image/*"
                        onChange={handleFileUpload}
                        className="sr-only"
                      />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">PNG, JPG, GIF up to 5MB</p>
                </div>
              </div>
            </div>
          )}

          {/* Image Preview */}
          {previewUrl && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Preview
              </label>
              <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
                <img
                  src={previewUrl}
                  alt="Ad preview"
                  className="max-h-64 mx-auto rounded"
                  onError={() => {
                    setError('Failed to load image. Please check the URL or file.')
                    setPreviewUrl('')
                  }}
                />
              </div>
            </div>
          )}

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
            disabled={loading || !formData.image_url}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Predict CTR
          </button>
        </form>
      </div>

      {/* Loading Progress */}
      {loading && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-center gap-3">
            <svg className="animate-spin h-8 w-8 text-purple-600" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <div className="text-center">
              <p className="text-sm font-medium text-gray-900">Analyzing {formData.population_size} personas...</p>
              <p className="text-xs text-gray-500 mt-1">This may take a few moments</p>
            </div>
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
      {result && <Results data={result} adType="image" adContent={previewUrl} />}
    </div>
  )
}

export default ImageAdPredictor
