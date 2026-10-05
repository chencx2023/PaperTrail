import { createRoot } from 'react-dom/client'
import { ResearchApp } from './ResearchApp'
import './style.css'
import 'katex/dist/katex.min.css'

createRoot(document.getElementById('root')!).render(<ResearchApp />)
