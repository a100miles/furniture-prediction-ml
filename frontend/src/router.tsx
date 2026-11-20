import { AnchorHTMLAttributes, MouseEvent, useEffect, useState } from 'react'
import { flushSync } from 'react-dom'

/* A two-page router with animated page changes (View Transitions API, CSS fallback). */

const listeners = new Set<() => void>()
const notify = () => listeners.forEach((listener) => listener())

type DocWithVT = Document & { startViewTransition?: (update: () => void) => unknown }

function jumpTo(hash: string) {
  const target = hash ? document.getElementById(hash.slice(1)) : null
  const top = target ? target.getBoundingClientRect().top + window.scrollY - 24 : 0
  window.scrollTo({ top, behavior: 'instant' as ScrollBehavior })
}

type Point = { x: number; y: number }

/** The new page grows as a circle from `origin` (the click); CSS reads these variables. */
function setOrigin(origin?: Point) {
  const x = origin?.x ?? window.innerWidth / 2
  const y = origin?.y ?? window.innerHeight / 2
  const r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y))
  const root = document.documentElement.style
  root.setProperty('--vt-x', `${x}px`)
  root.setProperty('--vt-y', `${y}px`)
  root.setProperty('--vt-r', `${Math.ceil(r)}px`)
}

function transition(update: () => void, hash: string, origin?: Point) {
  setOrigin(origin)
  const doc = document as DocWithVT
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  if (doc.startViewTransition && !reduce) {
    doc.startViewTransition(() => {
      flushSync(update)
      jumpTo(hash)
    })
  } else {
    update()
    requestAnimationFrame(() => jumpTo(hash))
  }
}

export function navigate(to: string, origin?: Point) {
  const url = new URL(to, window.location.href)
  if (url.pathname === window.location.pathname) {
    if (url.hash) document.getElementById(url.hash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    else window.scrollTo({ top: 0, behavior: 'smooth' })
    return
  }
  transition(() => {
    window.history.pushState({}, '', url.pathname + url.hash)
    notify()
  }, url.hash, origin)
}

export function usePath() {
  const [path, setPath] = useState(window.location.pathname)
  useEffect(() => {
    const sync = () => setPath(window.location.pathname)
    const onPop = () => transition(() => { flushSync(sync); notify() }, window.location.hash)
    listeners.add(sync)
    window.addEventListener('popstate', onPop)
    return () => { listeners.delete(sync); window.removeEventListener('popstate', onPop) }
  }, [])
  return path
}

export function Link({ to, onClick, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement> & { to: string }) {
  function handle(event: MouseEvent<HTMLAnchorElement>) {
    onClick?.(event)
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    // keyboard activation has no pointer position (detail 0): grow from the link itself
    const box = event.currentTarget.getBoundingClientRect()
    const origin = event.detail > 0 ? { x: event.clientX, y: event.clientY } : { x: box.left + box.width / 2, y: box.top + box.height / 2 }
    navigate(to, origin)
  }
  return <a href={to} onClick={handle} {...rest} />
}
