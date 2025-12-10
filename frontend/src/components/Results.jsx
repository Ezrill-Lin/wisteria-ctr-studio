function Results({ data, adType, adContent }) {
  if (!data) return null

  // Map API response fields to component variables
  const ctr = data.ctr ?? data.estimated_ctr ?? 0
  const clicks = data.total_clicks ?? data.clicks ?? 0
  const population = data.total_personas ?? data.population_size ?? 0
  const analysis = data.final_analysis ?? data.analysis ?? ''
  const personaResponses = data.persona_responses ?? []

  return (
    <div className="space-y-6">
      {/* CTR Summary Card */}
      <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg border border-purple-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-800">Prediction Results</h2>
          <span className="px-3 py-1 bg-white rounded-full text-sm font-medium text-purple-700 border border-purple-200">
            {adType === 'text' ? '📝 Text Ad' : '🖼️ Image Ad'}
          </span>
        </div>

        {/* Ad Content Preview */}
        <div className="bg-white rounded-lg p-4 mb-4 border border-gray-200">
          <p className="text-xs font-medium text-gray-500 uppercase mb-2">Ad Content</p>
          {adType === 'text' ? (
            <p className="text-gray-800 text-sm">{adContent}</p>
          ) : (
            <img src={adContent} alt="Advertisement" className="max-h-32 rounded" />
          )}
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg p-4 border border-purple-200">
            <p className="text-xs font-medium text-gray-500 uppercase mb-1">Estimated CTR</p>
            <p className="text-3xl font-bold text-purple-600">
              {(ctr * 100).toFixed(2)}%
            </p>
          </div>
          
          <div className="bg-white rounded-lg p-4 border border-purple-200">
            <p className="text-xs font-medium text-gray-500 uppercase mb-1">Clicks</p>
            <p className="text-3xl font-bold text-blue-600">{clicks}</p>
          </div>

          <div className="bg-white rounded-lg p-4 border border-purple-200">
            <p className="text-xs font-medium text-gray-500 uppercase mb-1">Population</p>
            <p className="text-3xl font-bold text-gray-700">{population}</p>
          </div>
        </div>
      </div>

      {/* Analysis Card */}
      {analysis && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-5 h-5 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
            </svg>
            <h3 className="text-lg font-semibold text-gray-800">AI Analysis</h3>
          </div>
          <div className="prose prose-sm max-w-none">
            <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">{analysis}</p>
          </div>
        </div>
      )}

      {/* Persona Responses (if included) */}
      {personaResponses && personaResponses.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
              </svg>
              <h3 className="text-lg font-semibold text-gray-800">Individual Persona Responses</h3>
            </div>
            <span className="px-2 py-1 bg-gray-100 rounded text-xs font-medium text-gray-600">
              {personaResponses.length} personas
            </span>
          </div>

          <div className="max-h-96 overflow-y-auto space-y-3">
            {personaResponses.map((persona, index) => (
              <div
                key={index}
                className={`border rounded-lg p-4 transition-colors ${
                  persona.clicked
                    ? 'bg-green-50 border-green-200'
                    : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-800">
                      {persona.demographics || 'No demographic information available'}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${
                      persona.clicked
                        ? 'bg-green-100 text-green-700'
                        : 'bg-gray-200 text-gray-600'
                    }`}
                  >
                    {persona.clicked ? '✓ Clicked' : '✗ No Click'}
                  </span>
                </div>

                {persona.reasoning && (
                  <p className="text-xs text-gray-600 mt-2 border-t border-gray-200 pt-2">
                    <span className="font-medium">Reasoning:</span> {persona.reasoning}
                  </p>
                )}

                {persona.ocean_scores && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs font-medium text-gray-600 mb-2">Big Five Traits</p>
                    <div className="grid grid-cols-5 gap-2">
                      {Object.entries(persona.ocean_scores).map(([trait, score]) => (
                        <div key={trait} className="text-center">
                          <p className="text-[10px] text-gray-500 uppercase">{trait[0]}</p>
                          <p className="text-xs font-semibold text-gray-700">{score}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Export Button */}
      <div className="flex justify-end">
        <button
          onClick={() => {
            const dataStr = JSON.stringify(data, null, 2)
            const dataBlob = new Blob([dataStr], { type: 'application/json' })
            const url = URL.createObjectURL(dataBlob)
            const link = document.createElement('a')
            link.href = url
            link.download = `ctr-prediction-${Date.now()}.json`
            link.click()
            URL.revokeObjectURL(url)
          }}
          className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium text-sm flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Export Results
        </button>
      </div>
    </div>
  )
}

export default Results
