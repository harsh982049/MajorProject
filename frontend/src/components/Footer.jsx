// File: src/components/Footer.jsx

export default function Footer() {
    return (
      <footer className="border-t bg-white/10 dark:bg-black/20 mt-8">
        <div className="container flex flex-col md:flex-row items-center justify-between py-6">
          {/* Project Name */}
          <div className="flex items-center gap-2 mb-4 md:mb-0">
            <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
              AW
            </div>
            <span className="text-lg font-bold">AI Emotional Wellbeing (College Project)</span>
          </div>
  
          <div className="text-sm text-center md:text-right text-gray-700 dark:text-gray-300">
            <p>© {new Date().getFullYear()} AI Emotional Wellbeing. All rights reserved.</p>
            <p>Developed as part of an academic curriculum.</p>
          </div>
        </div>
      </footer>
    )
  }
  