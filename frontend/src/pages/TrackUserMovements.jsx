import { useState } from "react"
import { motion } from "framer-motion"
import axios from "axios"
import Navbar from "../components/Navbar"
import Footer from "../components/Footer"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { CheckCircle, AlertCircle, MousePointerClick, Keyboard, Activity } from "lucide-react"

export default function StressTrackerConsentPage() {
  const [trackingStarted, setTrackingStarted] = useState(false);
  const [error, setError] = useState(null);

  const handleStartTracking = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/start_tracking");

      if (response.status === 200) {
        setTrackingStarted(true);
        setError(null);
      } else {
        setError(response.data.message || "Failed to start tracking.");
      }
    } catch (err) {
      setError("Could not connect to tracking server.");
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />

      <main className="flex-1 bg-gradient-to-b from-blue-50 to-white dark:from-slate-900 dark:to-slate-800 py-16 md:py-24">
        <div className="container px-4 md:px-6 text-center">
          <motion.div
            className="max-w-2xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-bold mb-4">Consent to Start Stress Monitoring</h1>
            <p className="text-muted-foreground mb-6">
              This tool tracks your mouse and keyboard behavior in the background to help detect stress using rule-based analysis. All data is processed locally and notifications are shown only when needed. You may stop tracking anytime from your system tray.
            </p>

            <Card className="max-w-lg mx-auto">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Activity className="w-5 h-5 text-primary" />
                  Real-Time Behavior Tracking
                </CardTitle>
                <CardDescription>
                  Tracking starts once you click below and continues even after leaving this site.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="text-left space-y-2 text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <MousePointerClick className="w-4 h-4" /> Tracks mouse movement speed & jitter
                  </li>
                  <li className="flex items-center gap-2">
                    <Keyboard className="w-4 h-4" /> Monitors typing speed and backspace rate
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" /> Shows notifications based on stress level
                  </li>
                  <li className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" /> Data is stored locally in a CSV file
                  </li>
                </ul>
              </CardContent>
              <CardFooter className="justify-center">
                <Button size="lg" onClick={handleStartTracking} disabled={trackingStarted}>
                  {trackingStarted ? "Tracking Started" : "I Consent — Start Monitoring"}
                </Button>
              </CardFooter>
              {error && <p className="text-sm text-red-500 mt-2">{error}</p>}
            </Card>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
