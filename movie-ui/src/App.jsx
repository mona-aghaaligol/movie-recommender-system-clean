import { useState } from 'react';

function App() {
  const [userId, setUserId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);

  const handleGetRecommendations = async () => {
    setHasSearched(true);
    setLoading(true);
    setError('');
    setResults([]);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/v1/recommendations/${userId}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        padding: '2rem',
        fontFamily: 'Arial, sans-serif',
        maxWidth: '600px',
        margin: '0 auto',
        textAlign: 'center',
      }}
    >
      <h1>Movie Recommender</h1>
      <p>Enter a user ID to get movie recommendations.</p>

      <input
        type="text"
        placeholder="Enter user ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        style={{
          padding: '0.6rem',
          width: '220px',
          borderRadius: '6px',
          border: '1px solid #ccc',
        }}
      />

      <button
        onClick={handleGetRecommendations}
        disabled={loading || !userId.trim()}
        style={{
          padding: '0.6rem 1.2rem',
          cursor: loading || !userId.trim() ? 'not-allowed' : 'pointer',
          backgroundColor: loading || !userId.trim() ? '#a5b4fc' : '#4f46e5',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          marginLeft: '0.5rem',
        }}
      >
        {loading ? 'Loading...' : 'Get Recommendations'}
      </button>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {hasSearched && !loading && !error && results.length === 0 && (
        <p>No recommendations found.</p>
      )}

      {results.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Recommendations:</h3>

          {results.map((movie) => (
            <div
              key={movie.movie_id}
              style={{
                padding: '1.2rem',
                marginBottom: '0.75rem',
                borderRadius: '8px',
                backgroundColor: '#f9fafb',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                transition: 'transform 0.15s ease, box-shadow 0.15s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'scale(1.02)';
                e.currentTarget.style.boxShadow =
                  '0 4px 12px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
                e.currentTarget.style.boxShadow =
                  '0 1px 3px rgba(0,0,0,0.1)';
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '1rem',
                }}
              >
                <div
                  style={{
                    width: '70px',
                    height: '100px',
                    borderRadius: '6px',
                    backgroundColor: '#e5e7eb',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.8rem',
                    color: '#6b7280',
                    flexShrink: 0,
                  }}
                >
                  {movie.poster_url ? (
                    <img
                      src={movie.poster_url}
                      alt={movie.title}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                      }}
                    />
                  ) : (
                    'Poster'
                  )}
                </div>

                <div style={{ textAlign: 'left' }}>
                  <div
                    style={{
                      fontWeight: '600',
                      fontSize: '1.05rem',
                      lineHeight: '1.3',
                      marginBottom: '0.25rem',
                    }}
                  >
                    {movie.title}
                  </div>

                  <div
                    style={{
                      color: 'gray',
                      fontSize: '0.9rem',
                      marginTop: '0.2rem',
                    }}
                  >
                    Score: {movie.score.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;