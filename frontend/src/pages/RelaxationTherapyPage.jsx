import { useState, useRef, useEffect } from "react"
/* eslint-disable-next-line no-unused-vars */
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Link } from "react-router-dom"
import { Play, Pause, SkipForward, SkipBack, Volume2, Music, Wind, Brain, Activity } from "lucide-react"

const musicTracks = [
  {
    id: 1,
    title: "Calm Waters",
    artist: "Nature Sounds",
    duration: "5:32",
    category: "nature",
    url: "https://example.com/calm-waters.mp3",
  },
  {
    id: 2,
    title: "Forest Ambience",
    artist: "Nature Sounds",
    duration: "4:18",
    category: "nature",
    url: "https://example.com/forest-ambience.mp3",
  },
  {
    id: 3,
    title: "Gentle Rain",
    artist: "Nature Sounds",
    duration: "6:45",
    category: "nature",
    url: "https://example.com/gentle-rain.mp3",
  },
  {
    id: 4,
    title: "Meditation Melody",
    artist: "Zen Music",
    duration: "8:12",
    category: "meditation",
    url: "https://example.com/meditation-melody.mp3",
  },
  {
    id: 5,
    title: "Peaceful Piano",
    artist: "Classical Calm",
    duration: "4:55",
    category: "instrumental",
    url: "https://example.com/peaceful-piano.mp3",
  },
  {
    id: 6,
    title: "Ambient Dreams",
    artist: "Chill Vibes",
    duration: "7:20",
    category: "instrumental",
    url: "https://example.com/ambient-dreams.mp3",
  },
  {
    id: 7,
    title: "Ocean Waves",
    artist: "Nature Sounds",
    duration: "10:15",
    category: "nature",
    url: "https://example.com/ocean-waves.mp3",
  },
  {
    id: 8,
    title: "Relaxing Guitar",
    artist: "Acoustic Calm",
    duration: "5:48",
    category: "instrumental",
    url: "https://example.com/relaxing-guitar.mp3",
  },
]

const RelaxationTherapyPage = () => {
  const [selectedMood, setSelectedMood] = useState("neutral")
  const [activeTab, setActiveTab] = useState("breathing")
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTrack, setCurrentTrack] = useState(null)
  const [volume, setVolume] = useState([70])
  const [breathingPhase, setBreathingPhase] = useState("inhale")
  const [breathingCount, setBreathingCount] = useState(0)
  const breathingIntervalRef = useRef(null)
  const audioRef = useRef(null)

  const startBreathingExercise = () => {
    if (breathingIntervalRef.current) return

    setBreathingPhase("inhale")
    setBreathingCount(0)

    breathingIntervalRef.current = setInterval(() => {
      setBreathingPhase((prev) => {
        if (prev === "inhale") return "hold"
        if (prev === "hold") return "exhale"
        setBreathingCount((count) => count + 1)
        return "inhale"
      })
    }, 4000)
  }

  const stopBreathingExercise = () => {
    if (breathingIntervalRef.current) {
      clearInterval(breathingIntervalRef.current)
      breathingIntervalRef.current = null
    }
    setBreathingPhase("inhale")
    setBreathingCount(0)
  }

  const playTrack = (track) => {
    setCurrentTrack(track)
    setIsPlaying(true)
    console.log(`Playing track: ${track.title}`)
  }

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleVolumeChange = (newValue) => {
    setVolume(newValue)
    if (audioRef.current) {
      audioRef.current.volume = newValue[0] / 100
    }
  }

  const getRecommendedTracks = () => {
    switch (selectedMood) {
      case "anxious":
        return musicTracks.filter((track) => track.category === "nature")
      case "stressed":
        return musicTracks.filter((track) => track.category === "meditation")
      default:
        return musicTracks
    }
  }

  useEffect(() => {
    return () => {
      stopBreathingExercise()
    }
  }, [])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 100 },
    },
  }

  const circleVariants = {
    inhale: {
      scale: 1.5,
      transition: { duration: 4, ease: "easeInOut" },
    },
    hold: {
      scale: 1.5,
      transition: { duration: 4, ease: "easeInOut" },
    },
    exhale: {
      scale: 1,
      transition: { duration: 4, ease: "easeInOut" },
    },
  }

  return (
    <motion.div className="container mx-auto px-4 py-8" variants={containerVariants} initial="hidden" animate="visible">
      <motion.div variants={itemVariants} className="mb-8 text-center">
        <h1 className="text-3xl font-bold mb-2">Adaptive Relaxation & Music Therapy</h1>
        <p className="text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
          Personalized relaxation techniques and music therapy to help reduce stress and improve your emotional
          wellbeing.
        </p>
      </motion.div>

      <motion.div variants={itemVariants} className="mb-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="mr-2 h-5 w-5" /> How are you feeling today?
            </CardTitle>
            <CardDescription>Select your current mood to get personalized recommendations</CardDescription>
          </CardHeader>
          <CardContent>
            <RadioGroup value={selectedMood} onValueChange={setSelectedMood} className="flex flex-wrap gap-4">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="calm" id="calm" />
                <Label htmlFor="calm">Calm</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="neutral" id="neutral" />
                <Label htmlFor="neutral">Neutral</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="anxious" id="anxious" />
                <Label htmlFor="anxious">Anxious</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="stressed" id="stressed" />
                <Label htmlFor="stressed">Stressed</Label>
              </div>
            </RadioGroup>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div variants={itemVariants}>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-3 mb-6">
            <TabsTrigger value="breathing" className="flex items-center">
              <Wind className="mr-2 h-4 w-4" /> Breathing
            </TabsTrigger>
            <TabsTrigger value="meditation" className="flex items-center">
              <Brain className="mr-2 h-4 w-4" /> Meditation
            </TabsTrigger>
            <TabsTrigger value="music" className="flex items-center">
              <Music className="mr-2 h-4 w-4" /> Music Therapy
            </TabsTrigger>
          </TabsList>

          <TabsContent value="breathing">
            <Card>
              <CardHeader>
                <CardTitle>Guided Breathing Exercise</CardTitle>
                <CardDescription>
                  Follow the animation to regulate your breathing pattern and reduce stress
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center">
                <div className="text-center mb-6">
                  <h3 className="text-xl font-semibold mb-2">Box Breathing Technique</h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Inhale for 4 seconds, hold for 4 seconds, exhale for 4 seconds, repeat
                  </p>
                </div>

                <div className="relative flex items-center justify-center h-64 w-64 mb-6">
                  <motion.div
                    className="absolute h-16 w-16 bg-primary/20 rounded-full flex items-center justify-center"
                    animate={breathingPhase}
                    variants={circleVariants}
                  >
                    <div className="h-8 w-8 bg-primary rounded-full" />
                  </motion.div>
                  <div className="absolute text-lg font-medium">
                    {breathingPhase.charAt(0).toUpperCase() + breathingPhase.slice(1)}
                  </div>
                </div>

                <div className="text-center mb-4">
                  <p className="text-sm text-gray-500 dark:text-gray-400">Completed cycles: {breathingCount}</p>
                </div>

                <div className="flex gap-4">
                  {breathingIntervalRef.current ? (
                    <Button variant="outline" onClick={stopBreathingExercise}>
                      Stop Exercise
                    </Button>
                  ) : (
                    <Button onClick={startBreathingExercise}>Start Exercise</Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="meditation">
            <Card>
              <CardHeader>
                <CardTitle>Mindfulness Meditation</CardTitle>
                <CardDescription>Guided practices to help you stay present and reduce stress</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Body Scan Meditation</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                        A 10-minute guided meditation to release tension throughout your body
                      </p>
                      <Button className="w-full">Start Session</Button>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Loving-Kindness Meditation</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                        A 15-minute practice to cultivate compassion and positive emotions
                      </p>
                      <Button className="w-full">Start Session</Button>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Progressive Muscle Relaxation</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                        A 12-minute guided session to release physical tension
                      </p>
                      <Button className="w-full">Start Session</Button>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Mindful Breathing</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                        A 5-minute quick meditation focusing on breath awareness
                      </p>
                      <Button className="w-full">Start Session</Button>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="music">
            <Card>
              <CardHeader>
                <CardTitle>Music Therapy</CardTitle>
                <CardDescription>Calming music tracks to help reduce stress and anxiety</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-6">
                  <h3 className="text-lg font-medium mb-2">Recommended for your mood</h3>
                  <div className="space-y-2">
                    {getRecommendedTracks().map((track) => (
                      <div
                        key={track.id}
                        className={`p-3 rounded-lg flex justify-between items-center cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 ${currentTrack?.id === track.id ? "bg-primary/10" : ""}`}
                        onClick={() => playTrack(track)}
                      >
                        <div>
                          <div className="font-medium">{track.title}</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">{track.artist}</div>
                        </div>
                        <div className="flex items-center">
                          <span className="text-sm text-gray-500 dark:text-gray-400 mr-3">{track.duration}</span>
                          <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
                            <Play className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {currentTrack && (
                  <div className="border-t pt-4">
                    <div className="flex justify-between items-center mb-4">
                      <div>
                        <div className="font-medium">{currentTrack.title}</div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">{currentTrack.artist}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button size="icon" variant="ghost" className="h-8 w-8">
                          <SkipBack className="h-4 w-4" />
                        </Button>
                        <Button size="icon" onClick={togglePlayPause}>
                          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                        </Button>
                        <Button size="icon" variant="ghost" className="h-8 w-8">
                          <SkipForward className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <Volume2 className="h-4 w-4 text-gray-500" />
                      <Slider value={volume} onValueChange={handleVolumeChange} max={100} step={1} className="w-full" />
                    </div>

                    <audio ref={audioRef} src={currentTrack.url} />
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </motion.div>

      <motion.div variants={itemVariants} className="mt-6">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center mb-4">
              <h3 className="text-lg font-medium">Explore Other Features</h3>
              <p className="text-gray-600 dark:text-gray-300">
                Continue your emotional wellbeing journey with our other tools
              </p>
            </div>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/stress-detection">
                <Button variant="outline" className="w-full sm:w-auto">
                  <Activity className="mr-2 h-4 w-4" /> Stress Detection
                </Button>
              </Link>
              <Link to="/stress-visualization">
                <Button variant="outline" className="w-full sm:w-auto">
                  <Activity className="mr-2 h-4 w-4" /> Stress Visualization
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}

export default RelaxationTherapyPage

