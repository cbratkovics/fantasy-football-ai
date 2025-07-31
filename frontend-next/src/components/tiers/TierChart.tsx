'use client'

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import tiersData from '@/data/tiers_2024.json'

interface Player {
  id: string
  name: string
  team: string
  tier_confidence: number
  projected_points: number
  consistency_score: number
}

interface Tier {
  tier: number
  label: string
  color: string
  players: Player[]
}

interface TierChartProps {
  position: 'QB' | 'RB' | 'WR' | 'TE'
  width?: number
  height?: number
}

export function TierChart({ position, width = 800, height = 600 }: TierChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoveredPlayer, setHoveredPlayer] = useState<Player | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const margin = { top: 40, right: 20, bottom: 60, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const tiers = tiersData.tiers[position] as Tier[]
    const allPlayers = tiers.flatMap((tier, tierIndex) => 
      tier.players.map((player, playerIndex) => ({
        ...player,
        tier: tier.tier,
        tierLabel: tier.label,
        tierColor: tier.color,
        overallRank: tiers.slice(0, tierIndex).reduce((acc, t) => acc + t.players.length, 0) + playerIndex + 1
      }))
    )

    // Scales
    const xScale = d3.scaleLinear()
      .domain([1, allPlayers.length])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain(d3.extent(allPlayers, d => d.projected_points) as [number, number])
      .range([innerHeight, 0])

    const colorScale = d3.scaleOrdinal()
      .domain(tiers.map(t => t.tier.toString()))
      .range(tiers.map(t => t.color))

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Add background grid
    const xAxis = d3.axisBottom(xScale).tickSize(-innerHeight).tickFormat(() => '')
    const yAxis = d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(() => '')

    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll('line')
      .style('stroke', '#374151')
      .style('stroke-opacity', 0.3)

    g.append('g')
      .attr('class', 'grid')
      .call(yAxis)
      .selectAll('line')
      .style('stroke', '#374151')
      .style('stroke-opacity', 0.3)

    // Add tier background regions
    tiers.forEach((tier, tierIndex) => {
      const startRank = tiers.slice(0, tierIndex).reduce((acc, t) => acc + t.players.length, 0) + 1
      const endRank = startRank + tier.players.length - 1

      g.append('rect')
        .attr('x', xScale(startRank) - 10)
        .attr('y', 0)
        .attr('width', xScale(endRank) - xScale(startRank) + 20)
        .attr('height', innerHeight)
        .attr('fill', tier.color)
        .attr('opacity', 0.1)
        .attr('rx', 4)
    })

    // Add tier break lines
    const tierBreaks = tiersData.tier_breaks[position]
    if (tierBreaks) {
      tierBreaks.forEach(tierBreak => {
        const tierEndIndex = tiers.findIndex(t => t.tier === tierBreak.between_tiers[0])
        if (tierEndIndex >= 0) {
          const breakRank = tiers.slice(0, tierEndIndex + 1).reduce((acc, t) => acc + t.players.length, 0) + 0.5
          
          g.append('line')
            .attr('x1', xScale(breakRank))
            .attr('x2', xScale(breakRank))
            .attr('y1', 0)
            .attr('y2', innerHeight)
            .attr('stroke', '#EF4444')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,5')
            .attr('opacity', 0.7)

          // Add break annotation
          g.append('text')
            .attr('x', xScale(breakRank) + 5)
            .attr('y', 20)
            .attr('fill', '#EF4444')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .text(`-${tierBreak.point_gap} pts`)
        }
      })
    }

    // Add connected line
    const line = d3.line<typeof allPlayers[0]>()
      .x(d => xScale(d.overallRank))
      .y(d => yScale(d.projected_points))
      .curve(d3.curveMonotoneX)

    g.append('path')
      .datum(allPlayers)
      .attr('fill', 'none')
      .attr('stroke', '#60A5FA')
      .attr('stroke-width', 2)
      .attr('opacity', 0.6)
      .attr('d', line)

    // Add player circles
    const circles = g.selectAll('.player-circle')
      .data(allPlayers)
      .enter()
      .append('circle')
      .attr('class', 'player-circle')
      .attr('cx', d => xScale(d.overallRank))
      .attr('cy', d => yScale(d.projected_points))
      .attr('r', 6)
      .attr('fill', d => colorScale(d.tier.toString()) as string)
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 8)
          .attr('stroke-width', 3)

        setHoveredPlayer(d)
        setTooltipPosition({ x: event.pageX, y: event.pageY })
      })
      .on('mouseleave', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 6)
          .attr('stroke-width', 2)

        setHoveredPlayer(null)
      })
      .on('mousemove', function(event) {
        setTooltipPosition({ x: event.pageX, y: event.pageY })
      })

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `#${d}`))
      .selectAll('text')
      .style('fill', '#D1D5DB')

    g.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('fill', '#D1D5DB')

    // Add axis labels
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (innerHeight / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('fill', '#D1D5DB')
      .style('font-size', '14px')
      .text('Projected Points')

    g.append('text')
      .attr('transform', `translate(${innerWidth / 2}, ${innerHeight + margin.bottom - 10})`)
      .style('text-anchor', 'middle')
      .style('fill', '#D1D5DB')
      .style('font-size', '14px')
      .text('Overall Rank')

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .style('fill', '#FFFFFF')
      .style('font-size', '18px')
      .style('font-weight', 'bold')
      .text(`${position} Player Tiers - Value Drop Visualization`)

  }, [position, width, height])

  return (
    <div className="relative">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
      >
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="w-full h-auto"
        />
      </motion.div>

      {/* Tooltip */}
      {hoveredPlayer && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: tooltipPosition.x + 10,
            top: tooltipPosition.y - 10,
          }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gray-900 text-white rounded-lg p-3 shadow-xl border border-gray-700 max-w-xs"
          >
            <div className="font-semibold text-lg">{hoveredPlayer.name}</div>
            <div className="text-gray-400 text-sm mb-2">{hoveredPlayer.team} • Rank #{hoveredPlayer.overallRank}</div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span>Tier:</span>
                <span className="font-medium">Tier {hoveredPlayer.tier} ({hoveredPlayer.tierLabel})</span>
              </div>
              <div className="flex justify-between">
                <span>Projected:</span>
                <span className="font-medium text-blue-400">{hoveredPlayer.projected_points} pts</span>
              </div>
              <div className="flex justify-between">
                <span>Consistency:</span>
                <span className="font-medium">{Math.round(hoveredPlayer.consistency_score * 100)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Tier Confidence:</span>
                <span className="font-medium">{Math.round(hoveredPlayer.tier_confidence * 100)}%</span>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}