'use client'

import { useState, useEffect } from 'react'
import { 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  ClockIcon,
  ServerIcon,
  GlobeAltIcon,
  CpuChipIcon,
  CircleStackIcon,
  CloudIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

type ServiceStatus = 'operational' | 'degraded' | 'outage' | 'maintenance'

interface Service {
  name: string
  status: ServiceStatus
  uptime: string
  responseTime: string
  description: string
  icon: any
  lastUpdated: string
}

interface Incident {
  id: string
  title: string
  status: 'investigating' | 'identified' | 'monitoring' | 'resolved'
  severity: 'low' | 'medium' | 'high' | 'critical'
  startTime: string
  description: string
  updates: {
    time: string
    message: string
    status: string
  }[]
}

const services: Service[] = [
  {
    name: 'API Services',
    status: 'operational',
    uptime: '99.98%',
    responseTime: '125ms',
    description: 'Core API endpoints for player data and predictions',
    icon: ServerIcon,
    lastUpdated: '2 minutes ago'
  },
  {
    name: 'Web Application',
    status: 'operational',
    uptime: '99.99%',
    responseTime: '89ms',
    description: 'Main web application and user interface',
    icon: GlobeAltIcon,
    lastUpdated: '1 minute ago'
  },
  {
    name: 'ML Prediction Engine',
    status: 'operational',
    uptime: '99.95%',
    responseTime: '2.1s',
    description: 'Machine learning models and prediction generation',
    icon: CpuChipIcon,
    lastUpdated: '3 minutes ago'
  },
  {
    name: 'Database',
    status: 'operational',
    uptime: '99.97%',
    responseTime: '45ms',
    description: 'Player data, statistics, and user information storage',
    icon: CircleStackIcon,
    lastUpdated: '1 minute ago'
  },
  {
    name: 'CDN & Assets',
    status: 'operational',
    uptime: '99.99%',
    responseTime: '32ms',
    description: 'Content delivery network and static assets',
    icon: CloudIcon,
    lastUpdated: '4 minutes ago'
  }
]

const recentIncidents: Incident[] = [
  {
    id: '1',
    title: 'Elevated API Response Times',
    status: 'resolved',
    severity: 'medium',
    startTime: '2024-01-15T14:30:00Z',
    description: 'Users experienced slower than normal API response times during peak usage.',
    updates: [
      {
        time: '2024-01-15T15:45:00Z',
        message: 'Issue has been resolved. API response times have returned to normal.',
        status: 'resolved'
      },
      {
        time: '2024-01-15T15:15:00Z',
        message: 'We have identified the cause and are implementing a fix.',
        status: 'identified'
      },
      {
        time: '2024-01-15T14:35:00Z',
        message: 'We are investigating reports of slow API response times.',
        status: 'investigating'
      }
    ]
  }
]

const upcomingMaintenance = [
  {
    title: 'Database Optimization',
    date: '2024-01-20T02:00:00Z',
    duration: '2 hours',
    description: 'Scheduled maintenance to optimize database performance. No user impact expected.',
    impact: 'No expected impact'
  }
]

export default function StatusPage() {
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [refreshing, setRefreshing] = useState(false)

  const getStatusColor = (status: ServiceStatus) => {
    switch (status) {
      case 'operational':
        return 'text-green-600'
      case 'degraded':
        return 'text-yellow-600'
      case 'outage':
        return 'text-red-600'
      case 'maintenance':
        return 'text-blue-600'
      default:
        return 'text-gray-600'
    }
  }

  const getStatusIcon = (status: ServiceStatus) => {
    switch (status) {
      case 'operational':
        return CheckCircleIcon
      case 'degraded':
        return ExclamationTriangleIcon
      case 'outage':
        return XCircleIcon
      case 'maintenance':
        return ClockIcon
      default:
        return CheckCircleIcon
    }
  }

  const getStatusBg = (status: ServiceStatus) => {
    switch (status) {
      case 'operational':
        return 'bg-green-50 border-green-200'
      case 'degraded':
        return 'bg-yellow-50 border-yellow-200'
      case 'outage':
        return 'bg-red-50 border-red-200'
      case 'maintenance':
        return 'bg-blue-50 border-blue-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low':
        return 'bg-blue-100 text-blue-800'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800'
      case 'high':
        return 'bg-orange-100 text-orange-800'
      case 'critical':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short'
    })
  }

  const refreshStatus = async () => {
    setRefreshing(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    setLastUpdated(new Date())
    setRefreshing(false)
  }

  const overallStatus = services.every(s => s.status === 'operational') ? 'operational' : 'issues'

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-12">
            <Breadcrumb items={[{ name: 'System Status', current: true }]} />
            
            <div className="mt-4">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-4xl font-bold text-gray-900">
                    System Status
                  </h1>
                  <p className="mt-2 text-lg text-gray-600">
                    Current status of WinMyLeague.ai services and infrastructure
                  </p>
                </div>
                <div className="text-right">
                  <button
                    onClick={refreshStatus}
                    disabled={refreshing}
                    className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50"
                  >
                    <ArrowPathIcon className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                    Refresh
                  </button>
                  <p className="mt-1 text-xs text-gray-500">
                    Last updated: {lastUpdated.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Overall Status */}
          <div className={`mb-8 rounded-lg border p-6 ${overallStatus === 'operational' ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
            <div className="flex items-center">
              {overallStatus === 'operational' ? (
                <CheckCircleIcon className="h-8 w-8 text-green-600" />
              ) : (
                <ExclamationTriangleIcon className="h-8 w-8 text-yellow-600" />
              )}
              <div className="ml-4">
                <h2 className={`text-xl font-semibold ${overallStatus === 'operational' ? 'text-green-900' : 'text-yellow-900'}`}>
                  {overallStatus === 'operational' ? 'All Systems Operational' : 'Some Systems Experiencing Issues'}
                </h2>
                <p className={`text-sm ${overallStatus === 'operational' ? 'text-green-700' : 'text-yellow-700'}`}>
                  {overallStatus === 'operational' 
                    ? 'All services are running normally with no known issues.'
                    : 'We are working to resolve any service disruptions as quickly as possible.'
                  }
                </p>
              </div>
            </div>
          </div>

          {/* Services Status */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Service Status</h2>
            <div className="space-y-4">
              {services.map((service) => {
                const StatusIcon = getStatusIcon(service.status)
                return (
                  <div
                    key={service.name}
                    className={`rounded-lg border p-6 ${getStatusBg(service.status)}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center mr-4">
                          <service.icon className="h-5 w-5 text-gray-600" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-gray-900">
                            {service.name}
                          </h3>
                          <p className="text-sm text-gray-600">
                            {service.description}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center gap-2 mb-1">
                          <StatusIcon className={`h-5 w-5 ${getStatusColor(service.status)}`} />
                          <span className={`font-medium capitalize ${getStatusColor(service.status)}`}>
                            {service.status}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 space-y-1">
                          <div>Uptime: {service.uptime}</div>
                          <div>Response: {service.responseTime}</div>
                          <div>Updated: {service.lastUpdated}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </section>

          {/* Recent Incidents */}
          {recentIncidents.length > 0 && (
            <section className="mb-12">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Recent Incidents</h2>
              <div className="space-y-6">
                {recentIncidents.map((incident) => (
                  <div
                    key={incident.id}
                    className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">
                          {incident.title}
                        </h3>
                        <p className="text-sm text-gray-600 mt-1">
                          {incident.description}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSeverityColor(incident.severity)}`}>
                          {incident.severity}
                        </span>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                          incident.status === 'resolved' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {incident.status}
                        </span>
                      </div>
                    </div>
                    
                    <div className="border-l-2 border-gray-200 pl-4 ml-2">
                      {incident.updates.map((update, idx) => (
                        <div key={idx} className="mb-3 last:mb-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs font-medium text-gray-500">
                              {formatDate(update.time)}
                            </span>
                            <span className={`px-1.5 py-0.5 text-xs font-medium rounded ${
                              update.status === 'resolved' ? 'bg-green-100 text-green-800' :
                              update.status === 'identified' ? 'bg-blue-100 text-blue-800' :
                              'bg-yellow-100 text-yellow-800'
                            }`}>
                              {update.status}
                            </span>
                          </div>
                          <p className="text-sm text-gray-700">
                            {update.message}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Upcoming Maintenance */}
          {upcomingMaintenance.length > 0 && (
            <section className="mb-12">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Scheduled Maintenance</h2>
              <div className="space-y-4">
                {upcomingMaintenance.map((maintenance, idx) => (
                  <div
                    key={idx}
                    className="bg-blue-50 border border-blue-200 rounded-lg p-6"
                  >
                    <div className="flex items-start">
                      <ClockIcon className="h-6 w-6 text-blue-600 mt-1 mr-3" />
                      <div>
                        <h3 className="text-lg font-semibold text-blue-900">
                          {maintenance.title}
                        </h3>
                        <p className="text-sm text-blue-700 mt-1">
                          {maintenance.description}
                        </p>
                        <div className="mt-3 flex items-center gap-4 text-sm text-blue-600">
                          <span>
                            <strong>Date:</strong> {formatDate(maintenance.date)}
                          </span>
                          <span>
                            <strong>Duration:</strong> {maintenance.duration}
                          </span>
                          <span>
                            <strong>Expected Impact:</strong> {maintenance.impact}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Subscribe to Updates */}
          <section className="text-center bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Stay Updated
            </h2>
            <p className="text-gray-600 mb-6">
              Subscribe to status updates to get notified of any service disruptions or maintenance windows.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center max-w-md mx-auto">
              <input
                type="email"
                placeholder="Enter your email"
                className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              />
              <button className="px-6 py-2 bg-indigo-600 text-white rounded-md font-semibold hover:bg-indigo-700 transition-colors">
                Subscribe
              </button>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}