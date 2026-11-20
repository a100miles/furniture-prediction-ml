import { CSSProperties, PointerEvent, useEffect, useRef, useState } from 'react'
import { ArrowRight, ArrowUpRight } from 'lucide-react'
import { CountUp, Magnetic, Tilt } from './fx'
import { MaskedLines, Reveal } from './reveal'
import { Link } from './router'

const STEPS = [
  { n: '01', title: 'Specifications', text: 'Width, depth, height, price, materials and colour from 3,961 mebel.kz listings. Colour variants of one design stay on the same side of the split.' },
  { n: '02', title: 'Photos', text: 'Licensed catalogue photos from Amazon Berkeley Objects. A pretrained ResNet-18 turns each photo into features; a small CNN trained from scratch is the baseline.' },
  { n: '03', title: 'Fair comparison', text: 'Specs only, photo only and both together are scored on the same objects with grouped cross-validation and macro-F1, then the errors are read one by one.' },
]

const RESULTS = [
  { value: 0.949, label: 'Specs-only SVM on mebel.kz', note: 'Grouped CV macro-F1, the chosen model' },
  { value: 0.918, label: 'Three sizes alone', note: 'Type is mostly a size question' },
  { value: 0.606, label: 'Same model on another store', note: 'Sizes do not transfer cleanly to Amazon' },
]

const LOGOS = [
  { src: '/logos/mebel-kz.svg', alt: 'mebel.kz', h: 14 },
  { src: '/logos/amazon.png', alt: 'Amazon', h: 16 },
  { src: '/logos/ikea.png', alt: 'IKEA', h: 19 },
]

// Infinite logo strip: the track holds two identical halves and slides by -50%.
function LogoSlider() {
  const half = [...LOGOS, ...LOGOS, ...LOGOS]
  return (
    <section className="logos" aria-label="Data sources">
      <div className="logos-viewport">
        <ul className="logos-track">
          {[...half, ...half].map((logo, index) => (
            <li key={index} aria-hidden={index >= LOGOS.length ? true : undefined}>
              <img src={logo.src} alt={index < LOGOS.length ? logo.alt : ''} style={{ height: logo.h }} loading="lazy" draggable={false} />
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

export default function Home() {
  const [scrollY, setScrollY] = useState(0)
  const heroRef = useRef<HTMLElement>(null)
  const [videoOk, setVideoOk] = useState(true)
  const [heroVideoReady, setHeroVideoReady] = useState(false)

  useEffect(() => {
    let frame = 0
    const onScroll = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => setScrollY(Math.min(window.scrollY, window.innerHeight)))
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => { window.removeEventListener('scroll', onScroll); cancelAnimationFrame(frame) }
  }, [])

  // Headline and photo drift a little against the cursor for depth.
  function heroMove(event: PointerEvent<HTMLElement>) {
    if (event.pointerType !== 'mouse') return
    const box = heroRef.current!.getBoundingClientRect()
    heroRef.current!.style.setProperty('--hx', String((event.clientX - box.left) / box.width - 0.5))
    heroRef.current!.style.setProperty('--hy', String((event.clientY - box.top) / box.height - 0.5))
  }

  return (
    <main className="page">
      <section className="hero" ref={heroRef} onPointerMove={heroMove} aria-label="Introduction">
        <div className="hero-media" style={{ '--py': `${scrollY * 0.25}px` } as CSSProperties}>
          <img src="/hero-minimal.jpg" alt="" />
          <video className={`hero-video${heroVideoReady ? ' is-ready' : ''}`} autoPlay muted loop playsInline preload="metadata" aria-hidden="true" onCanPlay={() => setHeroVideoReady(true)} onError={() => setHeroVideoReady(false)}>
            <source src="/hero-background.mp4" type="video/mp4" />
          </video>
        </div>
        <div className="hero-shade" aria-hidden="true" />
        <div className="hero-fade" aria-hidden="true" />

        <div className="hero-copy">
          <MaskedLines as="h1" lines={['Furniture,', 'read by a model.']} className="display" start={280} immediate />
          <Reveal delay={620}>
            <p className="lede">
              Give it the sizes, price and materials of a listing or a single product photo. The saved models
              tell you what kind of furniture it is and how sure they are.
            </p>
          </Reveal>
          <Reveal delay={780}>
            <Magnetic>
              <Link to="/predict" className="glass-btn">Predict <ArrowRight size={17} /></Link>
            </Magnetic>
          </Reveal>
        </div>
      </section>

      <LogoSlider />

      <section className="section method" id="method" aria-label="Method">
        {videoOk && (
          <video className="method-video" autoPlay muted loop playsInline preload="metadata" aria-hidden="true" onError={() => setVideoOk(false)}>
            <source src="/method-loop.mp4" type="video/mp4" onError={() => setVideoOk(false)} />
          </video>
        )}
        <div className="section-head">
          <Reveal><p className="eyebrow">Method</p></Reveal>
          <MaskedLines lines={['Two sources,', 'one honest comparison.']} className="title" />
        </div>
        <div className="msteps">
          {STEPS.map((step, index) => (
            <Reveal key={step.n} delay={index * 120}>
              <Tilt className="mstep surface">
                <h3>{step.title}</h3>
                <p>{step.text}</p>
                <span className="ghost-num" aria-hidden="true">{step.n}</span>
              </Tilt>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="story" id="results" aria-label="Results">
        <img className="story-bg" src="/results-bg.gif" alt="" aria-hidden="true" loading="lazy" decoding="async" />
        <div className="story-inner">
          <div className="story-head">
            <Reveal><p className="eyebrow light">Results</p></Reveal>
            <MaskedLines lines={['Photos see what', 'a tape measure misses.']} className="title light" />
          </div>
          <div className="story-copy">
            <Reveal delay={160}>
              <div className="big-stat">
                <span className="big-num"><CountUp value={0.794} duration={1800} /></span>
                <span className="big-cap">macro-F1 from specs + photo on ABO,<br />up from <b>0.662</b> with specs alone</span>
              </div>
            </Reveal>
            <Reveal delay={260}>
              <blockquote>
                TV-stand recall rose from 0.08 to 0.64 once the photo was added: the low cabinets that sizes
                alone keep calling chests of drawers.
              </blockquote>
            </Reveal>
          </div>
        </div>
        <div className="stat-row">
          {RESULTS.map((item, index) => (
            <Reveal key={item.label} delay={index * 120} className="stat glass">
              <span className="stat-num"><CountUp value={item.value} /></span>
              <span className="stat-label">{item.label}</span>
              <span className="stat-note">{item.note}</span>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section data" id="data" aria-label="Data and rights">
        <img className="data-bg" src="/data-bg.gif" alt="" aria-hidden="true" loading="lazy" decoding="async" />
        <div className="section-head">
          <Reveal><p className="eyebrow">Data and rights</p></Reveal>
          <MaskedLines lines={['Only data we may', 'lawfully use.']} className="title" />
        </div>
        <div className="cards">
          <Reveal>
            <Tilt className="card surface">
              <div className="card-fig"><b>3,961</b><span>listings, no photos</span></div>
              <h3>mebel.kz listings</h3>
              <p>Sizes, price, materials and colour of 3,961 listings. The store’s copyright notice forbids reusing its photos without consent, so no mebel.kz photo was downloaded.</p>
            </Tilt>
          </Reveal>
          <Reveal delay={120}>
            <Tilt className="card surface">
              <div className="card-fig"><b>3,371</b><span>listings, CC BY 4.0</span></div>
              <h3>Amazon Berkeley Objects</h3>
              <p>3,371 furniture listings with catalogue photos under CC BY 4.0. Credit: Amazon.com; dataset by Guillaumin et al. and Collins &amp; Malik (UC Berkeley).</p>
              <a href="https://amazon-berkeley-objects.s3.amazonaws.com/index.html" target="_blank" rel="noreferrer">Dataset page <ArrowUpRight size={14} /></a>
            </Tilt>
          </Reveal>
          <Reveal delay={240}>
            <Tilt className="card surface">
              <div className="card-fig"><b>0</b><span>used to train a model</span></div>
              <h3>Background photograph</h3>
              <p>“An empty room with a chair in the middle of it” by Serge Le Strat. Used under the Unsplash License, decoration only, never used by a model.</p>
              <a href="https://unsplash.com/photos/an-empty-room-with-a-chair-in-the-middle-of-it-tnw3krPFXxk" target="_blank" rel="noreferrer">Photo on Unsplash <ArrowUpRight size={14} /></a>
            </Tilt>
          </Reveal>
        </div>
      </section>

    </main>
  )
}
