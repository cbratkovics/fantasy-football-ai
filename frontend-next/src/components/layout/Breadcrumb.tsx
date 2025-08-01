import Link from 'next/link'
import { ChevronRightIcon, HomeIcon } from '@heroicons/react/20/solid'

interface BreadcrumbItem {
  name: string
  href?: string
  current?: boolean
}

interface BreadcrumbProps {
  items: BreadcrumbItem[]
}

export function Breadcrumb({ items }: BreadcrumbProps) {
  const allItems = [
    { name: 'Home', href: '/' },
    ...items
  ]

  return (
    <nav className="flex" aria-label="Breadcrumb">
      <ol role="list" className="flex items-center space-x-2">
        {allItems.map((item, index) => (
          <li key={item.name}>
            <div className="flex items-center">
              {index === 0 ? (
                <Link
                  href={item.href || '/'}
                  className="text-gray-400 hover:text-gray-500"
                >
                  <HomeIcon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
                  <span className="sr-only">{item.name}</span>
                </Link>
              ) : (
                <>
                  <ChevronRightIcon
                    className="h-5 w-5 flex-shrink-0 text-gray-300"
                    aria-hidden="true"
                  />
                  {item.current || !item.href ? (
                    <span className="ml-2 text-sm font-medium text-gray-500" aria-current={item.current ? 'page' : undefined}>
                      {item.name}
                    </span>
                  ) : (
                    <Link
                      href={item.href}
                      className="ml-2 text-sm font-medium text-gray-500 hover:text-gray-700"
                    >
                      {item.name}
                    </Link>
                  )}
                </>
              )}
            </div>
          </li>
        ))}
      </ol>
    </nav>
  )
}