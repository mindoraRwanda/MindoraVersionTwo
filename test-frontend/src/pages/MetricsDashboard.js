import React, { useState, useEffect } from 'react';

const MetricsDashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/metrics/summary');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  const exportMetrics = async (type) => {
    try {
      const response = await fetch(`http://localhost:8000/metrics/export/${type}`, {
        method: 'POST'
      });
      const data = await response.json();
      alert(`✅ ${data.message}`);
    } catch (err) {
      alert(`❌ Export failed: ${err.message}`);
    }
  };

  if (loading && !metrics) {
    return (
      <div className="p-8 text-center">
        <div className="text-gray-600">Loading metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 bg-red-50 border border-red-200 rounded-lg">
        <h3 className="text-red-800 font-semibold mb-2">Error Loading Metrics</h3>
        <p className="text-red-600">{error}</p>
        <button 
          onClick={fetchMetrics}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!metrics) return null;

  const { emotion_metrics, pipeline_metrics, system_health } = metrics;

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            📊 Mindora Analytics Dashboard
          </h1>
          <p className="text-gray-600">
            Real-time performance metrics for your mental health chatbot
          </p>
          <div className="mt-4 flex gap-4">
            <button 
              onClick={fetchMetrics}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              🔄 Refresh
            </button>
            <button 
              onClick={() => exportMetrics('emotion')}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              📥 Export Emotion Metrics
            </button>
            <button 
              onClick={() => exportMetrics('pipeline')}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              📥 Export Pipeline Metrics
            </button>
          </div>
        </div>

        {/* System Health */}
        <div className="mb-6 p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">🏥 System Health</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-600">ML Classifier</div>
              <div className={`text-lg font-bold ${system_health.ml_classifier_active ? 'text-green-600' : 'text-red-600'}`}>
                {system_health.ml_classifier_active ? '✅ Active' : '❌ Inactive'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">RAG Service</div>
              <div className={`text-lg font-bold ${system_health.rag_service_active ? 'text-green-600' : 'text-yellow-600'}`}>
                {system_health.rag_service_active ? '✅ Active' : '⚠️ Inactive'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Total Metrics</div>
              <div className="text-lg font-bold text-blue-600">
                {system_health.total_metrics_tracked}
              </div>
            </div>
          </div>
        </div>

        {/* Emotion Metrics */}
        <div className="mb-6 p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">😊 Emotion Detection Metrics</h2>
          
          {emotion_metrics.total_classifications === 0 ? (
            <p className="text-gray-600">No emotion classifications yet. Send some messages!</p>
          ) : (
            <div className="space-y-6">
              {/* Key Stats */}
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded">
                  <div className="text-sm text-gray-600">Total Classifications</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {emotion_metrics.total_classifications}
                  </div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded">
                  <div className="text-sm text-gray-600">Avg Confidence</div>
                  <div className="text-2xl font-bold text-green-600">
                    {(emotion_metrics.avg_confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded">
                  <div className="text-sm text-gray-600">Avg Latency</div>
                  <div className="text-2xl font-bold text-purple-600">
                    {emotion_metrics.avg_processing_ms.toFixed(0)}ms
                  </div>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded">
                  <div className="text-sm text-gray-600">Cultural Detection</div>
                  <div className="text-2xl font-bold text-yellow-600">
                    {(emotion_metrics.cultural_detection_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Emotion Distribution */}
              {emotion_metrics.emotion_distribution && (
                <div>
                  <h3 className="font-semibold mb-3">Emotion Distribution</h3>
                  <div className="space-y-2">
                    {Object.entries(emotion_metrics.emotion_distribution).map(([emotion, count]) => {
                      const percentage = (count / emotion_metrics.total_classifications * 100).toFixed(1);
                      return (
                        <div key={emotion} className="flex items-center gap-3">
                          <div className="w-24 text-sm capitalize">{emotion}</div>
                          <div className="flex-1 bg-gray-200 rounded-full h-6">
                            <div 
                              className="bg-blue-600 h-6 rounded-full flex items-center justify-end pr-2 text-white text-xs font-semibold"
                              style={{ width: `${percentage}%` }}
                            >
                              {percentage > 10 && `${count} (${percentage}%)`}
                            </div>
                          </div>
                          {percentage <= 10 && (
                            <div className="text-sm text-gray-600">{count} ({percentage}%)</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Pipeline Metrics */}
        <div className="mb-6 p-6 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">⚡ Pipeline Performance Metrics</h2>
          
          {pipeline_metrics.total_executions === 0 ? (
            <p className="text-gray-600">No pipeline executions yet. Send some messages!</p>
          ) : (
            <div className="space-y-6">
              {/* Key Stats */}
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded">
                  <div className="text-sm text-gray-600">Total Executions</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {pipeline_metrics.total_executions}
                  </div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded">
                  <div className="text-sm text-gray-600">Avg Response Time</div>
                  <div className="text-2xl font-bold text-green-600">
                    {(pipeline_metrics.avg_response_time_ms / 1000).toFixed(2)}s
                  </div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded">
                  <div className="text-sm text-gray-600">Avg LLM Calls</div>
                  <div className="text-2xl font-bold text-purple-600">
                    {pipeline_metrics.avg_llm_calls.toFixed(1)}
                  </div>
                </div>
                <div className="text-center p-4 bg-red-50 rounded">
                  <div className="text-sm text-gray-600">Error Rate</div>
                  <div className="text-2xl font-bold text-red-600">
                    {(pipeline_metrics.error_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Strategy Distribution */}
              {pipeline_metrics.strategy_distribution && (
                <div>
                  <h3 className="font-semibold mb-3">Response Strategy Distribution</h3>
                  <div className="space-y-2">
                    {Object.entries(pipeline_metrics.strategy_distribution).map(([strategy, count]) => {
                      const percentage = (count / pipeline_metrics.total_executions * 100).toFixed(1);
                      return (
                        <div key={strategy} className="flex items-center gap-3">
                          <div className="w-40 text-sm">{strategy}</div>
                          <div className="flex-1 bg-gray-200 rounded-full h-6">
                            <div 
                              className="bg-purple-600 h-6 rounded-full flex items-center justify-end pr-2 text-white text-xs font-semibold"
                              style={{ width: `${percentage}%` }}
                            >
                              {percentage > 10 && `${count} (${percentage}%)`}
                            </div>
                          </div>
                          {percentage <= 10 && (
                            <div className="text-sm text-gray-600">{count} ({percentage}%)</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Crisis Distribution */}
              {pipeline_metrics.crisis_distribution && (
                <div>
                  <h3 className="font-semibold mb-3">Crisis Level Distribution</h3>
                  <div className="flex gap-4">
                    {Object.entries(pipeline_metrics.crisis_distribution).map(([level, count]) => {
                      const colors = {
                        none: 'bg-green-100 text-green-800',
                        low: 'bg-yellow-100 text-yellow-800',
                        medium: 'bg-orange-100 text-orange-800',
                        high: 'bg-red-100 text-red-800',
                        critical: 'bg-red-200 text-red-900'
                      };
                      return (
                        <div key={level} className={`flex-1 p-4 rounded ${colors[level] || 'bg-gray-100'}`}>
                          <div className="text-sm capitalize">{level}</div>
                          <div className="text-2xl font-bold">{count}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Auto-refresh indicator */}
        <div className="text-center text-sm text-gray-500">
          Auto-refreshing every 10 seconds
        </div>
      </div>
    </div>
  );
};

export default MetricsDashboard;
