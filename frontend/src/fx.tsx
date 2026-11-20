import { CSSProperties, PointerEvent, ReactNode, useEffect, useRef, useState } from 'react'
import { useInView } from './reveal'

const reduced = () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Wraps a button/link so it leans toward the cursor and springs back on leave. */
export function Magnetic({ children, strength = 0.3, className = '' }: { children: ReactNode; strength?: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  function move(event: PointerEvent<HTMLSpanElement>) {
    if (reduced() || event.pointerType !== 'mouse') return
    const box = ref.current!.getBoundingClientRect()
    const x = (event.clientX - box.left - box.width / 2) * strength
    const y = (event.clientY - box.top - box.height / 2) * strength * 1.2
    ref.current!.style.setProperty('--mx', `${x}px`)
    ref.current!.style.setProperty('--my', `${y}px`)
  }
  function leave() {
    ref.current?.style.setProperty('--mx', '0px')
    ref.current?.style.setProperty('--my', '0px')
  }
  return (
    <span ref={ref} className={`magnetic ${className}`} onPointerMove={move} onPointerLeave={leave}>
      {children}
    </span>
  )
}

/** Card that tilts in 3D toward the cursor and shows a soft light spot. */
export function Tilt({ children, className = '', style }: { children: ReactNode; className?: string; style?: CSSProperties }) {
  const ref = useRef<HTMLDivElement>(null)
  function move(event: PointerEvent<HTMLDivElement>) {
    if (reduced() || event.pointerType !== 'mouse') return
    const box = ref.current!.getBoundingClientRect()
    const px = (event.clientX - box.left) / box.width
    const py = (event.clientY - box.top) / box.height
    ref.current!.style.setProperty('--rx', `${(0.5 - py) * 8}deg`)
    ref.current!.style.setProperty('--ry', `${(px - 0.5) * 10}deg`)
    ref.current!.style.setProperty('--lx', `${px * 100}%`)
    ref.current!.style.setProperty('--ly', `${py * 100}%`)
  }
  function leave() {
    ref.current?.style.setProperty('--rx', '0deg')
    ref.current?.style.setProperty('--ry', '0deg')
  }
  return (
    <div ref={ref} className={`tilt ${className}`} style={style} onPointerMove={move} onPointerLeave={leave}>
      {children}
    </div>
  )
}

/** Counts from 0 to `value` once it scrolls into view. */
export function CountUp({ value, decimals = 3, duration = 1400 }: { value: number; decimals?: number; duration?: number }) {
  const { ref, inView } = useInView<HTMLSpanElement>(0.4)
  const [shown, setShown] = useState(0)
  useEffect(() => {
    if (!inView) return
    if (reduced()) { setShown(value); return }
    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - t, 4)
      setShown(value * eased)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [inView, value, duration])
  return <span ref={ref}>{shown.toFixed(decimals)}</span>
}

/** Smoothly follows a target number (used to morph the live furniture outline). */
export function useSpring(target: number, speed = 0.14) {
  const [value, setValue] = useState(target)
  const current = useRef(target)
  useEffect(() => {
    if (reduced()) { current.current = target; setValue(target); return }
    let frame = 0
    const tick = () => {
      const diff = target - current.current
      if (Math.abs(diff) < 0.05) { current.current = target; setValue(target); return }
      current.current += diff * speed
      setValue(current.current)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, speed])
  return value
}
