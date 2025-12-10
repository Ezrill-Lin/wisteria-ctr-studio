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

      {/* AI Analysis - Card-based layout */}
      {analysis && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-5 h-5 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
            </svg>
            <h3 className="text-lg font-semibold text-gray-800">AI Analysis</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(() => {
              const sections = {};
              let currentSection = null;
              let currentContent = [];
              
              // Parse analysis line by line
              const lines = analysis.split('\n');
              
              for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                
                // Skip intro lines
                if (line.toLowerCase().startsWith("here's an analysis") ||
                    line.toLowerCase() === "analysis:" ||
                    line === "**Analysis**") {
                  continue;
                }
                
                // Check if it's a section header (e.g., **Performance**: content or **Strengths**)
                const headerMatch = line.match(/^\*\*(.+?)\*\*:?\s*(.*)$/);
                if (headerMatch) {
                  // Save previous section
                  if (currentSection && currentContent.length > 0) {
                    sections[currentSection] = currentContent.join('\n');
                  }
                  // Start new section
                  currentSection = headerMatch[1].trim();
                  currentContent = [];
                  // If there's content on the same line as the header, add it
                  if (headerMatch[2].trim()) {
                    currentContent.push(headerMatch[2].trim());
                  }
                } else if (currentSection && line) {
                  // Add content to current section
                  currentContent.push(line);
                }
              }
              
              // Save last section
              if (currentSection && currentContent.length > 0) {
                sections[currentSection] = currentContent.join('\n');
              }
              
              // If no sections were found, display the raw analysis as fallback
              if (Object.keys(sections).length === 0) {
                return (
                  <div className="bg-white rounded-lg shadow-sm border-2 border-gray-200 bg-gray-50 p-5">
                    <div className="text-sm text-gray-700 whitespace-pre-wrap">
                      {analysis}
                    </div>
                  </div>
                );
              }
              
              // Define section icons and colors
              const sectionConfig = {
                'Performance': { icon: '📊', color: 'blue' },
                'Strengths': { icon: '✅', color: 'green' },
                'Weaknesses': { icon: '⚠️', color: 'orange' },
                'Quick Wins': { icon: '🎯', color: 'purple' }
              };
              
              return Object.entries(sections).map(([title, content]) => {
                const config = sectionConfig[title] || { icon: '📝', color: 'gray' };
                const colorClasses = {
                  blue: 'border-blue-200 bg-blue-50',
                  green: 'border-green-200 bg-green-50',
                  orange: 'border-orange-200 bg-orange-50',
                  purple: 'border-purple-200 bg-purple-50',
                  gray: 'border-gray-200 bg-gray-50'
                };
                
                return (
                  <div key={title} className={`bg-white rounded-lg shadow-sm border-2 ${colorClasses[config.color]} p-5`}>
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-2xl">{config.icon}</span>
                      <h4 className="font-semibold text-gray-900 text-base">{title}</h4>
                    </div>
                    <div className="text-sm text-gray-700 space-y-2">
                      {content.split('\n').map((line, idx) => {
                        const trimmedLine = line.trim();
                        
                        // Helper function to render text with bold formatting
                        const renderTextWithBold = (text) => {
                          const parts = [];
                          let lastIndex = 0;
                          const boldRegex = /\*\*(.+?)\*\*/g;
                          let match;
                          
                          while ((match = boldRegex.exec(text)) !== null) {
                            // Add text before the bold part
                            if (match.index > lastIndex) {
                              parts.push(text.substring(lastIndex, match.index));
                            }
                            // Add bold part
                            parts.push(<strong key={match.index}>{match[1]}</strong>);
                            lastIndex = match.index + match[0].length;
                          }
                          // Add remaining text
                          if (lastIndex < text.length) {
                            parts.push(text.substring(lastIndex));
                          }
                          
                          return parts.length > 0 ? parts : text;
                        };
                        
                        // Numbered items - convert to bullet points
                        const numMatch = trimmedLine.match(/^(\d+)\.\s+(.+)$/);
                        if (numMatch) {
                          return (
                            <div key={idx} className="flex gap-2">
                              <span className="text-purple-600 mt-0.5">•</span>
                              <p className="flex-1">{renderTextWithBold(numMatch[2])}</p>
                            </div>
                          );
                        }
                        
                        // Bullet points
                        if (/^[\*\-•]\s/.test(trimmedLine)) {
                          const text = trimmedLine.replace(/^[\*\-•]\s+/, '');
                          return (
                            <div key={idx} className="flex gap-2">
                              <span className="text-purple-600 mt-0.5">•</span>
                              <p className="flex-1">{renderTextWithBold(text)}</p>
                            </div>
                          );
                        }
                        
                        // Regular text
                        if (trimmedLine) {
                          return <p key={idx}>{renderTextWithBold(trimmedLine)}</p>;
                        }
                        return null;
                      })}
                    </div>
                  </div>
                );
              });
            })()}
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
                      {persona.age || 'N/A'} years old, {persona.gender || 'N/A'}, {persona.occupation || 'N/A'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {persona.education || 'N/A'} • {persona.location || 'N/A'}
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
