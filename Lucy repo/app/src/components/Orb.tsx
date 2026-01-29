"use client"

import { useEffect, useRef, useMemo } from "react"

export type AgentState = "listening" | "talking" | null

interface OrbProps {
  colors: [string, string]
  seed: number
  agentState: AgentState
}

// Simple seeded random number generator
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = Math.sin(s * 12.9898 + 78.233) * 43758.5453
    return s - Math.floor(s)
  }
}

export function Orb({ colors, seed, agentState }: OrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>(0)
  const timeRef = useRef(0)
  const randomRef = useRef(seededRandom(seed))

  // Generate blob control points
  const blobPoints = useMemo(() => {
    const rand = randomRef.current
    const points: { angle: number; baseRadius: number; speed: number; offset: number }[] = []
    const numPoints = 6
    
    for (let i = 0; i < numPoints; i++) {
      points.push({
        angle: (i / numPoints) * Math.PI * 2,
        baseRadius: 0.75 + rand() * 0.15,
        speed: 0.3 + rand() * 0.4,
        offset: rand() * Math.PI * 2,
      })
    }
    
    return points
  }, [seed])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size for crisp rendering
    const dpr = window.devicePixelRatio || 1
    const size = canvas.clientWidth * dpr
    canvas.width = size
    canvas.height = size
    ctx.scale(dpr, dpr)

    const centerX = canvas.clientWidth / 2
    const centerY = canvas.clientHeight / 2
    const maxRadius = canvas.clientWidth * 0.4

    // Parse gradient colors
    const [color1, color2] = colors

    const animate = () => {
      timeRef.current += 0.016
      const t = timeRef.current

      // Clear canvas
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight)

      // Animation parameters based on agent state
      let intensity = 0.03
      let pulseSpeed = 0.5
      
      if (agentState === "listening") {
        intensity = 0.06
        pulseSpeed = 1.2
      } else if (agentState === "talking") {
        intensity = 0.08
        pulseSpeed = 2
      }

      // Calculate blob points with organic movement
      const points = blobPoints.map((point) => {
        const wave = Math.sin(t * point.speed * pulseSpeed + point.offset)
        const radius = maxRadius * (point.baseRadius + wave * intensity)
        return {
          x: centerX + Math.cos(point.angle) * radius,
          y: centerY + Math.sin(point.angle) * radius,
        }
      })

      // Create soft gradient
      const gradient = ctx.createRadialGradient(
        centerX - maxRadius * 0.3,
        centerY - maxRadius * 0.3,
        0,
        centerX,
        centerY,
        maxRadius * 1.2
      )
      gradient.addColorStop(0, color1)
      gradient.addColorStop(1, color2)

      // Draw smooth blob using spline curves
      ctx.beginPath()
      
      // Calculate control points for smooth curves
      for (let i = 0; i < points.length; i++) {
        const current = points[i]
        const next = points[(i + 1) % points.length]
        
        // Midpoint between current and next
        const midX = (current.x + next.x) / 2
        const midY = (current.y + next.y) / 2
        
        if (i === 0) {
          ctx.moveTo(midX, midY)
        }
        
        ctx.quadraticCurveTo(current.x, current.y, midX, midY)
      }
      
      ctx.closePath()

      // Fill with gradient
      ctx.fillStyle = gradient
      ctx.fill()

      // Add subtle inner highlight
      const highlight = ctx.createRadialGradient(
        centerX - maxRadius * 0.25,
        centerY - maxRadius * 0.25,
        0,
        centerX,
        centerY,
        maxRadius
      )
      highlight.addColorStop(0, "rgba(255, 255, 255, 0.25)")
      highlight.addColorStop(0.5, "rgba(255, 255, 255, 0.08)")
      highlight.addColorStop(1, "rgba(255, 255, 255, 0)")
      
      ctx.fillStyle = highlight
      ctx.fill()

      // Subtle outer glow for active states
      if (agentState) {
        ctx.save()
        ctx.filter = "blur(12px)"
        ctx.globalAlpha = 0.15
        ctx.fillStyle = color2
        ctx.fill()
        ctx.restore()
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      cancelAnimationFrame(animationRef.current)
    }
  }, [colors, blobPoints, agentState])

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full"
      style={{ display: "block" }}
    />
  )
}

export default Orb
