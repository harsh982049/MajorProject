/* eslint-disable-next-line no-unused-vars */
import { motion } from "framer-motion"
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar"

// Icons
import { Brain, Shield, Music, Code, Book, GitBranch, Heart, Database } from "lucide-react"

import Logo from "../assets/EmotionalWellBeing.png";
import emotionDetection from "../assets/Emotion-Detection.webp";
import panicSOS from "../assets/PanicSOS.png"
import musicTherapy from "../assets/MusicTherapy.jpg"

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
    },
  },
}

// Feature data
const features = [
  {
    title: "Real-Time Emotion Detection",
    description:
      "Experimental AI algorithms detect and analyze emotional states in real-time, demonstrating practical applications of machine learning in psychology.",
    icon: Brain,
    color: "bg-blue-100 dark:bg-blue-900",
    url: "/stress-detection"
  },
  {
    title: "Panic SOS with AI Companion",
    description:
      "A conceptual prototype showcasing how AI can provide guided interventions during moments of distress using evidence-based techniques.",
    icon: Shield,
    color: "bg-purple-100 dark:bg-purple-900",
    url: "/panic-chatbot"
  },
  {
    title: "Adaptive Relaxation & Music Therapy",
    description:
      "Explores the intersection of AI and sound therapy through personalized relaxation exercises for emotional regulation.",
    icon: Music,
    color: "bg-green-100 dark:bg-green-900",
    url: "/music-relaxation"
  },
]

export default function Dashboard() {
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />

      <main className="flex-1">
        {/* Hero Section */}
        <motion.section
          className="bg-gradient-to-b from-blue-50 to-white dark:from-slate-900 dark:to-slate-800 py-16 md:py-24"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center text-center gap-8">
              <motion.div 
                className="flex flex-col items-center text-center space-y-4 max-w-3xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.5 }}
              >
                <motion.h1
                  className="text-3xl md:text-5xl font-bold tracking-tighter"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                >
                  AI-Driven Emotional Wellbeing Research
                </motion.h1>
                <motion.p
                  className="text-gray-500 md:text-xl dark:text-gray-400"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4, duration: 0.5 }}
                >
                  A senior capstone project exploring the applications of artificial intelligence in understanding, 
                  assessing, and supporting emotional wellbeing among college students.
                </motion.p>
              </motion.div>

              <motion.div 
                className="w-full max-w-3xl"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5, duration: 0.6 }}
                whileHover={{ scale: 1.03 }}
              >
                <div className="relative rounded-lg overflow-hidden shadow-xl">
                  <img 
                    src={Logo}
                    alt="AI Wellbeing Project" 
                    className="w-full h-auto object-cover rounded-lg"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-60"></div>
                </div>
              </motion.div>
            </div>
          </div>
        </motion.section>

        <section className="py-16 md:py-24">
          <div className="container px-4 md:px-6">
            <motion.div
              className="text-center mb-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold tracking-tighter mb-4">Project Modules</h2>
              <p className="max-w-[700px] mx-auto text-gray-500 dark:text-gray-400">
                Our research explores three interconnected approaches to AI-assisted emotional wellbeing.
              </p>
            </motion.div>

            <motion.div
              className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
            >
              {features.map((feature, index) => (
                <motion.div key={index} variants={itemVariants}>
                  <Card className="h-full flex flex-col">
                    <CardHeader>
                      <div
                        className={`w-12 h-12 rounded-full ${feature.color} flex items-center justify-center mb-4`}
                      >
                        <feature.icon className="h-6 w-6 text-primary" />
                      </div>
                      <CardTitle>{feature.title}</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-grow">
                      <CardDescription className="text-base">{feature.description}</CardDescription>
                    </CardContent>
                    <CardFooter className="mt-auto pt-6">
                      <Button variant="default" className="w-full" onClick={() => window.location.href = feature.url}>
                        Check Out This Feature
                      </Button>
                    </CardFooter>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* Technology Stack Section */}
        <section className="py-16 md:py-24 bg-primary/5">
          <div className="container px-4 md:px-6">
            <motion.div
              className="text-center mb-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold tracking-tighter mb-4">Technology Stack</h2>
              <p className="max-w-[700px] mx-auto text-gray-600 dark:text-gray-300">
                The project leverages multiple technologies to create a functional prototype.
              </p>
            </motion.div>

            <motion.div
              className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
            >
              {/* Front-End */}
              <motion.div variants={itemVariants}>
                <Card className="h-full text-center">
                  <CardHeader>
                    <div className="mx-auto w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center mb-4">
                      <Code className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle>Front-End Development</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Built with React.js, Tailwind CSS, shadcn/ui components, and Framer Motion animations to create an accessible and responsive user interface.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>

              {/* ML/AI */}
              <motion.div variants={itemVariants}>
                <Card className="h-full text-center">
                  <CardHeader>
                    <div className="mx-auto w-12 h-12 rounded-full bg-purple-100 dark:bg-purple-900 flex items-center justify-center mb-4">
                      <Brain className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle>Machine Learning</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Implemented using TensorFlow and Python for emotion classification, reinforcement learning for adaptive responses, and natural language processing.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Backend */}
              <motion.div variants={itemVariants}>
                <Card className="h-full text-center">
                  <CardHeader>
                    <div className="mx-auto w-12 h-12 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center mb-4">
                      <Database className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle>Backend Infrastructure</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Developed with Flask for API endpoints and microservices, with MySQL database for secure and efficient data storage and retrieval.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>
          </div>
        </section>

        <section className="py-16 md:py-24">
          <div className="container px-4 md:px-6">
            <motion.div
              className="text-center mb-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold tracking-tighter mb-4">Project Highlights</h2>
              <p className="max-w-[700px] mx-auto text-gray-600 dark:text-gray-300">
                Key features and innovations that make our project stand out.
              </p>
            </motion.div>

            <motion.div 
              className="grid grid-cols-1 md:grid-cols-2 gap-8"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
            >
              <motion.div variants={itemVariants}>
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <GitBranch className="h-5 w-5 text-primary" />
                      Open Source Development
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Our entire codebase is open source and available on GitHub, encouraging collaboration and further development from the community.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5 text-primary" />
                      Privacy-Focused Design
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      We've implemented strong data protection measures to ensure user privacy is maintained throughout all interactions with our application.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Heart className="h-5 w-5 text-primary" />
                      Accessible Interface
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Our UI is designed following WCAG guidelines to ensure that emotional support tools are accessible to everyone, regardless of ability.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Book className="h-5 w-5 text-primary" />
                      Evidence-Based Approach
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      All intervention techniques implemented in our system are based on established psychological practices and therapeutic methods.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>
          </div>
        </section>

        <section className="py-16 md:py-24 bg-primary/5">
          <div className="container px-4 md:px-6">
            <motion.div
              className="text-center mb-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold tracking-tighter mb-4">Our Features In Detail</h2>
              <p className="max-w-[700px] mx-auto text-gray-600 dark:text-gray-300">
                Explore how each component of our system works to support emotional wellbeing.
              </p>
            </motion.div>

            <Tabs defaultValue="emotion-detection" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="emotion-detection">Emotion Detection</TabsTrigger>
                <TabsTrigger value="panic-sos">Panic SOS</TabsTrigger>
                <TabsTrigger value="music-therapy">Music Therapy</TabsTrigger>
              </TabsList>
              
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
              >
                <TabsContent value="emotion-detection" className="mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Brain className="h-5 w-5 text-primary" />
                        Real-Time Emotion Detection
                      </CardTitle>
                      <CardDescription>
                        Using machine learning to identify and track emotional states
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h3 className="text-lg font-medium mb-2">How It Works</h3>
                          <p className="text-muted-foreground mb-4">
                            Our emotion detection system analyzes facial expressions, voice patterns, and text inputs to determine the user's emotional state in real-time. The system is trained to recognize seven core emotions: joy, sadness, anger, fear, surprise, disgust, and neutral.
                          </p>
                          <p className="text-muted-foreground">
                            When a user interacts with the system, we process multiple data points simultaneously to create a comprehensive emotional profile, which is then used to tailor the support provided.
                          </p>
                        </div>
                        <div className="flex items-center justify-center">
                          <motion.div 
                            className="rounded-lg overflow-hidden shadow-lg"
                            whileHover={{ scale: 1.05 }}
                            transition={{ duration: 0.3 }}
                          >
                            <img 
                              src={emotionDetection}
                              alt="Emotion Detection Interface" 
                              className="w-96 h-auto"
                            />
                          </motion.div>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button onClick={() => window.location.href = "/stress-detection"}>
                        Try Emotion Detection
                      </Button>
                    </CardFooter>
                  </Card>
                </TabsContent>
                
                <TabsContent value="panic-sos" className="mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Shield className="h-5 w-5 text-primary" />
                        Panic SOS AI Companion
                      </CardTitle>
                      <CardDescription>
                        Immediate support during moments of high anxiety or panic
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h3 className="text-lg font-medium mb-2">How It Works</h3>
                          <p className="text-muted-foreground mb-4">
                            The Panic SOS feature provides immediate, evidence-based interventions during moments of acute anxiety. Using natural language processing, our AI companion guides users through breathing exercises, grounding techniques, and cognitive reframing.
                          </p>
                          <p className="text-muted-foreground">
                            The system adapts its response based on user feedback and the specific symptoms being described, creating a personalized support experience that helps users regain a sense of calm and control.
                          </p>
                        </div>
                        <div className="flex items-center justify-center">
                          <motion.div 
                            className="rounded-lg overflow-hidden shadow-lg"
                            whileHover={{ scale: 1.05 }}
                            transition={{ duration: 0.3 }}
                          >
                            <img 
                              src={panicSOS} 
                              alt="Panic SOS Interface" 
                              className="w-96 h-auto"
                            />
                          </motion.div>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button onClick={() => window.location.href = "/panic-chatbot"}>
                        Try Panic SOS Chatbot
                      </Button>
                    </CardFooter>
                  </Card>
                </TabsContent>
                
                <TabsContent value="music-therapy" className="mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Music className="h-5 w-5 text-primary" />
                        Adaptive Music Therapy
                      </CardTitle>
                      <CardDescription>
                        Personalized sound experiences for emotional regulation
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h3 className="text-lg font-medium mb-2">How It Works</h3>
                          <p className="text-muted-foreground mb-4">
                            Our music therapy feature uses adaptive algorithms to generate personalized sound environments based on the user's current emotional state and preferences. The system draws on research in psychoacoustics and music therapy to create compositions that promote relaxation and positive emotional shifts.
                          </p>
                          <p className="text-muted-foreground">
                            Users can select different sound environments, adjust parameters like tempo and intensity, and save favorites for future sessions. The system also tracks emotional responses to refine recommendations over time.
                          </p>
                        </div>
                        <div className="flex items-center justify-center">
                          <motion.div 
                            className="rounded-lg overflow-hidden shadow-lg"
                            whileHover={{ scale: 1.05 }}
                            transition={{ duration: 0.3 }}
                          >
                            <img 
                              src={musicTherapy}
                              alt="Music Therapy Interface" 
                              className="w-96 h-60"
                            />
                          </motion.div>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button onClick={() => window.location.href = "/music-relaxation"}>
                        Try Music Therapy
                      </Button>
                    </CardFooter>
                  </Card>
                </TabsContent>
              </motion.div>
            </Tabs>
          </div>
        </section>

        <motion.section
          className="py-16 md:py-24"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center text-center space-y-4">
              <h2 className="text-3xl font-bold tracking-tighter">Ready to Explore?</h2>
              <p className="max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
                Try our AI-powered emotional wellbeing tools and see how technology can support mental health.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 mt-6">
                <Button size="lg" onClick={() => window.location.href = "/stress-detection"}>
                  Start With Emotion Detection
                </Button>
                <Button size="lg" variant="outline" onClick={() => window.location.href = "/panic-chatbot"}>
                  Try The Panic SOS Chatbot
                </Button>
              </div>
            </div>
          </div>
        </motion.section>
      </main>

      <Footer/>
    </div>
  )
}