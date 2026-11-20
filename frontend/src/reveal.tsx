import { CSSProperties, ElementType, ReactNode, useEffect, useRef, useState } from 'react'

/** Adds `is-in` once the element enters the viewport (one-shot). */
export function useInView<T extends Element>(threshold = 0.18) {
  const ref = useRef<T | null>(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const node = ref.current
    if (!node) return
    if (typeof IntersectionObserver === 'undefined') { setInView(true); return }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setInView(true)
          observer.disconnect()
        }
      },
      { threshold, rootMargin: '0px 0px -8% 0px' },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [threshold])
  return { ref, inView }
}

/** Fade + rise + un-blur when scrolled into view. `delay` is in ms. */
export function Reveal({ as: Tag = 'div', delay = 0, className = '', children, ...rest }: {
  as?: ElementType; delay?: number; className?: string; children: ReactNode; id?: string
}) {
  const { ref, inView } = useInView<HTMLElement>()
  return (
    <Tag
      ref={ref}
      className={`reveal ${inView ? 'is-in' : ''} ${className}`}
      style={{ '--d': `${delay}ms` } as CSSProperties}
      {...rest}
    >
      {children}
    </Tag>
  )
}

/** Headline whose lines slide up from behind a mask, one after another. */
export function MaskedLines({ lines, as: Tag = 'h2', className = '', start = 0, immediate = false }: {
  lines: string[]; as?: ElementType; className?: string; start?: number; immediate?: boolean
}) {
  const { ref, inView } = useInView<HTMLElement>(0.3)
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    if (!immediate) return
    const id = requestAnimationFrame(() => setMounted(true))
    return () => cancelAnimationFrame(id)
  }, [immediate])
  const shown = immediate ? mounted : inView
  return (
    <Tag ref={ref} className={`masked ${shown ? 'is-in' : ''} ${className}`} aria-label={lines.join(' ')}>
      {lines.map((line, index) => (
        <span className="mask-line" key={line} aria-hidden="true">
          <span style={{ '--d': `${start + index * 110}ms` } as CSSProperties}>{line}</span>
        </span>
      ))}
    </Tag>
  )
}
