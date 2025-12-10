import { useState } from 'react'
import Results from './Results'

function HistoryPage({ history, onClose }) {
  const [selectedRecord, setSelectedRecord] = useState(null)

  const downloadRecord = (record) => {
    const dataStr = JSON.stringify(record, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    const timestamp = new Date(record.timestamp).toISOString().replace(/[:.]/g, '-')
    link.download = `prediction-${record.type}-${timestamp}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  if (selectedRecord) {
    // Detail view
    return (
      <div className="space-y-6">
        {/* Back button */}
        <button
          onClick={() => setSelectedRecord(null)}
          className="flex items-center gap-2 text-purple-600 hover:text-purple-700 font-medium"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to History
        </button>

        {/* Record header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                {selectedRecord.type === 'text' ? '📝 Text Ad' : '🖼️ Image Ad'} Prediction
              </h2>
              <p className="text-sm text-gray-500">
                {new Date(selectedRecord.timestamp).toLocaleString()}
              </p>
            </div>
            <button
              onClick={() => downloadRecord(selectedRecord)}
              className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download
            </button>
          </div>

          {/* Parameters */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-gray-200">
            <div>
              <p className="text-xs text-gray-500 uppercase">Population</p>
              <p className="text-lg font-semibold text-gray-900">{selectedRecord.parameters.population_size}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Platform</p>
              <p className="text-lg font-semibold text-gray-900">{selectedRecord.parameters.ad_platform}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Persona Version</p>
              <p className="text-lg font-semibold text-gray-900">{selectedRecord.parameters.persona_version}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase">Strategy</p>
              <p className="text-lg font-semibold text-gray-900">{selectedRecord.parameters.persona_strategy}</p>
            </div>
          </div>
        </div>

        {/* Results */}
        {selectedRecord.fullData && (
          <Results 
            data={selectedRecord.fullData} 
            adType={selectedRecord.type} 
            adContent={selectedRecord.adContent} 
          />
        )}
      </div>
    )
  }

  // List view
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Prediction History</h1>
            <p className="text-sm text-gray-600 mt-1">
              {history.length} prediction{history.length !== 1 ? 's' : ''} in this session
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* History list */}
      <div className="space-y-4">
        {history.map((record, index) => (
          <div
            key={index}
            className="bg-white rounded-lg shadow-sm border border-gray-200 hover:border-purple-300 transition-all cursor-pointer"
            onClick={() => setSelectedRecord(record)}
          >
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-2xl">
                      {record.type === 'text' ? '📝' : '🖼️'}
                    </span>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {record.type === 'text' ? 'Text Ad' : 'Image Ad'} - {new Date(record.timestamp).toLocaleString()}
                    </h3>
                  </div>
                  <div className="text-sm text-gray-600 mb-3">
                    {record.type === 'text' ? (
                      <p className="line-clamp-2">{record.adContent}</p>
                    ) : (
                      <img src={record.adContent} alt="Ad" className="max-h-20 rounded" />
                    )}
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <div>
                      <span className="text-gray-500">CTR:</span>{' '}
                      <span className="font-semibold text-purple-600">
                        {(record.results.ctr * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Clicks:</span>{' '}
                      <span className="font-semibold text-blue-600">{record.results.clicks}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Population:</span>{' '}
                      <span className="font-semibold text-gray-700">{record.results.population}</span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    downloadRecord(record)
                  }}
                  className="flex items-center gap-2 px-3 py-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors text-sm font-medium"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {history.length === 0 && (
        <div className="bg-gray-50 rounded-lg border-2 border-dashed border-gray-300 p-12 text-center">
          <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No predictions yet</h3>
          <p className="text-gray-600">Make a prediction to see it appear here</p>
        </div>
      )}
    </div>
  )
}

export default HistoryPage
