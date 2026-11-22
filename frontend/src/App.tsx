import { useEffect, useState } from 'react'
import Home from './Home'
import HowPage from './HowPage'
import PredictPage, { ModelStatus } from './PredictPage'
import { Link, usePath } from './router'

const TEAM = [
  { name: 'Sultanbek Zhumagali', handle: 'SULTE4', url: 'https://github.com/SULTE4' },
  { name: 'Tamerlan Yessimov', handle: 'a100miles', url: 'https://github.com/a100miles' },
]

function GithubMark() {
  return (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  )
}

export default function App() {
  const path = usePath()
  const onPredict = path.startsWith('/predict')
  const onHow = path.startsWith('/how')
  const [status, setStatus] = useState<ModelStatus>('checking')

  useEffect(() => {
    let active = true
    fetch('/api/health')
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then((body) => { if (active) setStatus(body.status === 'ready' ? 'ready' : 'unavailable') })
      .catch(() => { if (active) setStatus('unavailable') })
    return () => { active = false }
  }, [])

  useEffect(() => {
    document.title = onPredict ? 'Predict · Furniture Model Studio' : onHow ? 'How it works · Furniture Model Studio' : 'Furniture Model Studio'
  }, [onPredict, onHow])

  return (
    <>
      <header className="nav">
        <nav aria-label="Pages">
          <Link to="/" className={onPredict || onHow ? '' : 'on'}>Home</Link>
          <Link to="/predict" className={onPredict ? 'on' : ''}>Predict</Link>
          <Link to="/how" className={onHow ? 'on' : ''}>How it works</Link>
          <Link to="/#results">Results</Link>
          <Link to="/#data">Data</Link>
        </nav>
      </header>

      {onPredict ? <PredictPage status={status} /> : onHow ? <HowPage /> : <Home />}

      {!onPredict && <footer className="footer">
        <video className="footer-video" autoPlay muted loop playsInline preload="metadata" aria-hidden="true">
          <source src="/data-background.mp4" type="video/mp4" />
        </video>
        <div className="footer-cols">
          <div>
            <h4>Project</h4>
            <Link to="/predict">Predict</Link>
            <Link to="/how">How it works</Link>
            <Link to="/#method">Method</Link>
            <Link to="/#results">Results</Link>
            <Link to="/#data">Data</Link>
          </div>
          <div>
            <h4>Built by</h4>
            {TEAM.map((person) => (
              <a key={person.handle} href={person.url} target="_blank" rel="noreferrer" className="gh">
                <GithubMark />
                <span>{person.name}<small>@{person.handle}</small></span>
              </a>
            ))}
          </div>
        </div>
      </footer>}
    </>
  )
}
