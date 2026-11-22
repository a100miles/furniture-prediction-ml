import { CSSProperties, PointerEvent, useEffect, useRef, useState } from 'react'
import { ArrowRight, ArrowUpRight } from 'lucide-react'
import { CountUp, Magnetic, Tilt } from './fx'
import { MaskedLines, Reveal } from './reveal'
import { Link } from './router'

const STEPS = [
  { n: '01', title: 'Specifications', text: 'Width, depth, height, materials, colour and type from 3,961 mebel.kz listings. Colour variants of one design stay on the same side of the split.' },
  { n: '02', title: 'Product names', text: 'The second source for price, on the same listings and test set: TF-IDF word pieces we fit ourselves against a pretrained multilingual-e5 encoder, with the design name masked as a leakage check.' },
  { n: '03', title: 'Photos', text: 'Licensed catalogue photos from Amazon Berkeley Objects for the type. Pretrained networks (ResNet-18, CLIP, DINOv2) against a small network trained from scratch; the best, DINOv2, was also tested on IKEA photos it had never seen.' },
]

const RESULTS = [
  { value: 92, decimals: 0, suffix: ' / 100', label: 'Type from width, depth and height alone', note: 'Score on 793 listings the model never saw: the type is mostly a question of size (macro-F1 0.918)' },
  { value: 22.1, decimals: 1, suffix: '%', label: 'Typical price error from the specifications', note: 'Average gap to the real price on the same 793 listings (LightGBM, MAPE)' },
  { value: 85, decimals: 0, suffix: ' / 100', label: 'Type from a photo + sizes, Amazon', note: 'Up from 66 with sizes alone; pretrained DINOv2 (macro-F1 0.851)' },
]

const LOGOS = [
  { src: '/logos/mebel-kz.svg', alt: 'mebel.kz', h: 14 },
  { src: '/logos/amazon.png', alt: 'Amazon', h: 16 },
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
              Is a furniture price in Almaty typical for what is sold? Give the sizes, materials and name of a listing
              and the model estimates what it usually costs; give the sizes or a photo and it tells you what kind of
              furniture it is. Along the way we test which details really decide type and price.
            </p>
          </Reveal>
          <Reveal delay={780}>
            <div className="hero-btns">
              <Magnetic>
                <Link to="/predict" className="glass-btn">Predict <ArrowRight size={17} /></Link>
              </Magnetic>
              <Link to="/how" className="text-link">How it works and why <ArrowRight size={15} /></Link>
            </div>
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
          <MaskedLines lines={['A second source,', 'one honest comparison.']} className="title" />
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
            <MaskedLines lines={['Names price what', 'a tape measure misses.']} className="title light" />
          </div>
          <div className="story-copy">
            <Reveal delay={160}>
              <div className="big-stat">
                <span className="big-num"><CountUp value={20.6} decimals={1} duration={1800} />%</span>
                <span className="big-cap">average price error on 793 unseen listings with the specifications + product name,<br />down from <b>22.1%</b> with the specifications alone</span>
              </div>
            </Reveal>
            <Reveal delay={260}>
              <blockquote>
                Fabric, sliding doors and wall-hung cabinets are in the name, not in the size table. With the
                design name masked, three quarters of the gain stays: the model reads words, not price tags.
              </blockquote>
            </Reveal>
          </div>
        </div>
        <div className="stat-row">
          {RESULTS.map((item, index) => (
            <Reveal key={item.label} delay={index * 120} className="stat glass">
              <span className="stat-num"><CountUp value={item.value} decimals={item.decimals} />{item.suffix ?? ''}</span>
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
        <div className="cards cards-4">
          <Reveal>
            <Tilt className="card surface">
              <div className="card-fig"><b>3,961</b><span>listings, no photos</span></div>
              <h3>mebel.kz listings</h3>
              <p>Sizes, price, materials, colour and product name of 3,961 listings. The store’s copyright notice forbids reusing its photos and page text without consent, so no photo or description was used.</p>
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
          <Reveal delay={180}>
            <Tilt className="card surface">
              <div className="card-fig"><b>620 + 221</b><span>IKEA items · photos</span></div>
              <h3>IKEA, as a check</h3>
              <p>620 IKEA Saudi Arabia listings (price, sizes) test the price finding in a second market; 221 IKEA catalogue photos (non-commercial use) test the photo models on photos they never saw. Neither is used to train the app's models.</p>
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
