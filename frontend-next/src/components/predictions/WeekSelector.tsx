'use client'

interface WeekSelectorProps {
  selectedWeek: number
  onWeekChange: (week: number) => void
}

export function WeekSelector({ selectedWeek, onWeekChange }: WeekSelectorProps) {
  const weeks = Array.from({ length: 18 }, (_, i) => i + 1)

  return (
    <div>
      <label htmlFor="week-select" className="block text-sm font-medium text-gray-700 mb-2">
        Week
      </label>
      <select
        id="week-select"
        value={selectedWeek}
        onChange={(e) => onWeekChange(parseInt(e.target.value))}
        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
      >
        {weeks.map((week) => (
          <option key={week} value={week}>
            Week {week}
          </option>
        ))}
      </select>
    </div>
  )
}