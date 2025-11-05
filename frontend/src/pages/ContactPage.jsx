import { useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, CheckCircle, Github, Linkedin, Mail } from "lucide-react"
import Navbar from "@/components/Navbar"

const ContactPage = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  })
  const [formErrors, setFormErrors] = useState({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitSuccess, setSubmitSuccess] = useState(false)
  const [submitError, setSubmitError] = useState(false)

  // Team members data
  const teamMembers = [
    {
      name: "Harsh Shah",
      email: "harsh.shah22@spit.ac.in",
      linkedin: "https://linkedin.com/in/alexjohnson",
      github: "https://github.com/alexjohnson",
    },
    {
      name: "Tej Shah",
      email: "tej.shah22@spit.ac.in",
      linkedin: "https://linkedin.com/in/samanthalee",
      github: "https://github.com/samanthalee",
    },
    {
      name: "Veer Shah",
      email: "veer.shah22@spit.ac.in",
      linkedin: "https://linkedin.com/in/michaelchen",
      github: "https://github.com/michaelchen",
    },
  ]

  // Handle input change
  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: value,
    })

    // Clear error when user types
    if (formErrors[name]) {
      setFormErrors({
        ...formErrors,
        [name]: "",
      })
    }
  }

  // Validate form
  const validateForm = () => {
    const errors = {}

    if (!formData.name.trim()) {
      errors.name = "Name is required"
    }

    if (!formData.email.trim()) {
      errors.email = "Email is required"
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      errors.email = "Email is invalid"
    }

    if (!formData.message.trim()) {
      errors.message = "Message is required"
    }

    setFormErrors(errors)
    return Object.keys(errors).length === 0
  }

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault()

    if (validateForm()) {
      setIsSubmitting(true)
      setSubmitError(false)
      setSubmitSuccess(false)

      // Simulate API call
      setTimeout(() => {
        setIsSubmitting(false)
        setSubmitSuccess(true)

        // Reset form after successful submission
        setFormData({
          name: "",
          email: "",
          message: "",
        })
      }, 1500)
    }
  }

  // Animation variants
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

  return (
    <>
        <Navbar/>
        <motion.div className="container mx-auto px-4 py-8" variants={containerVariants} initial="hidden" animate="visible">
            <motion.div variants={itemVariants} className="mb-8 text-center">
                <h1 className="text-3xl font-bold mb-2">Contact Us</h1>
                <p className="text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                Have questions about our AI-driven emotional wellbeing project? We'd love to hear from you!
                </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Team Information */}
                <motion.div variants={itemVariants}>
                <Card className="h-full">
                    <CardHeader>
                    <CardTitle>Meet Our Team</CardTitle>
                    <CardDescription>The minds behind the AI Emotional Wellbeing project</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                    {teamMembers.map((member, index) => (
                        <div key={index} className="flex flex-col sm:flex-row gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="flex-shrink-0">
                            <div className="h-16 w-16 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold text-xl">
                            {member.name
                                .split(" ")
                                .map((n) => n[0])
                                .join("")}
                            </div>
                        </div>
                        <div className="flex-grow">
                            <h3 className="font-medium text-lg">{member.name}</h3>
                            <div className="flex flex-wrap gap-3">
                            <a
                                href={`mailto:${member.email}`}
                                className="flex items-center text-sm text-gray-600 dark:text-gray-300 hover:text-primary"
                            >
                                <Mail className="h-4 w-4 mr-1" /> {member.email}
                            </a>
                            <a
                                href={member.linkedin}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center text-sm text-gray-600 dark:text-gray-300 hover:text-primary"
                            >
                                <Linkedin className="h-4 w-4 mr-1" /> LinkedIn
                            </a>
                            <a
                                href={member.github}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center text-sm text-gray-600 dark:text-gray-300 hover:text-primary"
                            >
                                <Github className="h-4 w-4 mr-1" /> GitHub
                            </a>
                            </div>
                        </div>
                        </div>
                    ))}
                    </CardContent>
                    <CardFooter>
                    <div className="w-full text-center">
                        <a
                        href="https://github.com/ai-emotional-wellbeing"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center text-primary hover:underline"
                        >
                        <Github className="h-4 w-4 mr-2" /> Visit our GitHub repository
                        </a>
                    </div>
                    </CardFooter>
                </Card>
                </motion.div>

                {/* Contact Form */}
                <motion.div variants={itemVariants}>
                <Card className="h-full">
                    <CardHeader>
                    <CardTitle>Send Us a Message</CardTitle>
                    <CardDescription>Fill out the form below and we'll get back to you soon</CardDescription>
                    </CardHeader>
                    <CardContent>
                    {submitSuccess && (
                        <Alert className="mb-6 bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-300 border-green-200 dark:border-green-800">
                        <CheckCircle className="h-4 w-4" />
                        <AlertTitle>Success!</AlertTitle>
                        <AlertDescription>
                            Your message has been sent successfully. We'll get back to you soon.
                        </AlertDescription>
                        </Alert>
                    )}

                    {submitError && (
                        <Alert className="mb-6 bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 border-red-200 dark:border-red-800">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>There was an error sending your message. Please try again.</AlertDescription>
                        </Alert>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="space-y-2">
                        <Label htmlFor="name">Name</Label>
                        <Input
                            id="name"
                            name="name"
                            value={formData.name}
                            onChange={handleChange}
                            placeholder="Your name"
                            className={formErrors.name ? "border-red-500" : ""}
                        />
                        {formErrors.name && <p className="text-red-500 text-sm">{formErrors.name}</p>}
                        </div>

                        <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input
                            id="email"
                            name="email"
                            type="email"
                            value={formData.email}
                            onChange={handleChange}
                            placeholder="Your email address"
                            className={formErrors.email ? "border-red-500" : ""}
                        />
                        {formErrors.email && <p className="text-red-500 text-sm">{formErrors.email}</p>}
                        </div>

                        <div className="space-y-2">
                        <Label htmlFor="message">Message</Label>
                        <Textarea
                            id="message"
                            name="message"
                            value={formData.message}
                            onChange={handleChange}
                            placeholder="Your message"
                            rows={5}
                            className={formErrors.message ? "border-red-500" : ""}
                        />
                        {formErrors.message && <p className="text-red-500 text-sm">{formErrors.message}</p>}
                        </div>

                        <Button type="submit" className="w-full" disabled={isSubmitting}>
                        {isSubmitting ? "Sending..." : "Send Message"}
                        </Button>
                    </form>
                    </CardContent>
                </Card>
                </motion.div>
            </div>
            </motion.div>
    </>
  )
}

export default ContactPage

