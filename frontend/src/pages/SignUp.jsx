import React, {useState} from "react"
/* eslint-disable-next-line no-unused-vars */
import {motion} from "framer-motion"
import {toast} from "react-toastify"
import {ToastContainer} from "react-toastify"
import "react-toastify/dist/ReactToastify.css"
import axios from "axios"
import {Button} from "@/components/ui/button"
import {Input} from "@/components/ui/input"
import {Label} from "@/components/ui/label"
import {EyeIcon, EyeOffIcon, BrainCircuit} from "lucide-react"
import {Link, useNavigate} from "react-router-dom"

export default function SignUp()
{
	const [username, setUsername] = useState("");
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [confirmPassword, setConfirmPassword] = useState("");

	const [showPassword, setShowPassword] = useState(false);
	const [showConfirmPassword, setShowConfirmPassword] = useState(false);

	const [loading, setLoading] = useState(false);

	const navigate = useNavigate();

	// Framer Motion variants
	const containerVariants = {
		hidden: {opacity: 0},
		visible: {
			opacity: 1,
			transition: {
				staggerChildren: 0.1,
				delayChildren: 0.3,
			},
		},
	}

	const itemVariants = {
		hidden: {y: 20, opacity: 0},
		visible: {
			y: 0,
			opacity: 1,
			transition: {type: "spring", stiffness: 100},
		},
	}

	const validateForm = () => {
		if(!username.trim() || !email.trim() || !password.trim() || !confirmPassword.trim())
		{
			toast.error("All fields are required.")
			return false
		}
		if(password !== confirmPassword)
		{
			toast.error("Passwords do not match.")
			return false
		}
		return true
	}

	const handleSubmit = async (e) => {
		e.preventDefault()

		if(!validateForm()) return

		try
		{
			setLoading(true)
			const response = await axios.post("http://127.0.0.1:5000/api/register", {
				username,
				email,
				password,
				confirm_password: confirmPassword,
			})

			toast.success("Registration successful!")

			// If the backend returns a token on registration in any other case, store it in this way
			// const { token } = response.data
			// if (token) {
			//   localStorage.setItem("jwt_token", token)
			// }

			navigate("/login")
		}
		catch(err)
		{
			if(err.response?.data?.error) toast.error(err.response.data.error)
			else toast.error("Something went wrong. Please try again.")
		}
		finally
		{
			setLoading(false)
		}
	}

	return (
		<>
			<div className="flex flex-col md:flex-row min-h-screen">
			{/* Left side - Project content */}
				<motion.div
					className="w-full md:w-1/2 bg-gradient-to-br from-purple-600 to-indigo-700 text-white p-8 md:p-12 flex flex-col justify-center"
					initial={{ x: -50, opacity: 0 }}
					animate={{ x: 0, opacity: 1 }}
					transition={{ duration: 0.6 }}
				>
					<div className="max-w-md mx-auto">
						<motion.div
							className="flex items-center mb-6"
							initial={{ y: -20, opacity: 0 }}
							animate={{ y: 0, opacity: 1 }}
							transition={{ delay: 0.2, duration: 0.5 }}
						>
							<BrainCircuit className="h-10 w-10 mr-2" />
							<h1 className="text-3xl font-bold">AI Emotional Wellbeing</h1>
						</motion.div>

						<motion.h2
							className="text-2xl md:text-4xl font-bold mb-6"
							initial={{ y: 20, opacity: 0 }}
							animate={{ y: 0, opacity: 1 }}
							transition={{ delay: 0.4, duration: 0.5 }}
						>
							Begin Your Journey
						</motion.h2>

						<motion.p
							className="text-lg mb-8 opacity-90"
							initial={{ y: 20, opacity: 0 }}
							animate={{ y: 0, opacity: 1 }}
							transition={{ delay: 0.6, duration: 0.5 }}
						>
							Join our community and discover how AI can help you understand
							and improve your emotional wellbeing. Create an account to start
							your personalized experience.
						</motion.p>

						<motion.div
							className="space-y-6"
							initial={{ y: 20, opacity: 0 }}
							animate={{ y: 0, opacity: 1 }}
							transition={{ delay: 0.8, duration: 0.5 }}
						>
							<div className="bg-white/10 p-4 rounded-lg backdrop-blur-sm">
								<h3 className="font-medium text-lg mb-2">Why Sign Up?</h3>
								<ul className="space-y-2">
									<li className="flex items-start">
										<div className="h-1.5 w-1.5 rounded-full bg-white mt-2 mr-2"></div>
										<p>Personalized emotional analysis and insights</p>
									</li>
									<li className="flex items-start">
										<div className="h-1.5 w-1.5 rounded-full bg-white mt-2 mr-2"></div>
										<p>24/7 AI companion for emotional support</p>
									</li>
									<li className="flex items-start">
										<div className="h-1.5 w-1.5 rounded-full bg-white mt-2 mr-2"></div>
										<p>Custom relaxation techniques based on your needs</p>
									</li>
								</ul>
							</div>
						</motion.div>
					</div>
				</motion.div>

				{/* Right side - Signup form */}
				<motion.div
					className="w-full md:w-1/2 bg-white dark:bg-gray-950 p-8 md:p-12 flex items-center justify-center"
					initial={{ x: 50, opacity: 0 }}
					animate={{ x: 0, opacity: 1 }}
					transition={{ duration: 0.6 }}
				>
					<div className="w-full max-w-md">
						<motion.div
							className="mb-8 text-center"
							initial={{ y: -20, opacity: 0 }}
							animate={{ y: 0, opacity: 1 }}
							transition={{ delay: 0.3, duration: 0.5 }}
						>
							<h2 className="text-3xl font-bold mb-2">Create Account</h2>
							<p className="text-gray-500 dark:text-gray-400">
							Sign up to start your emotional wellbeing journey
							</p>
						</motion.div>

						<motion.form
							onSubmit={handleSubmit}
							variants={containerVariants}
							initial="hidden"
							animate="visible"
							className="space-y-5"
						>
							{/* Username input */}
							<motion.div variants={itemVariants}>
								<div className="space-y-2">
									<Label htmlFor="username">Username</Label>
									<Input
									id="username"
									type="text"
									placeholder="Choose a username"
									value={username}
									onChange={(e) => setUsername(e.target.value)}
									required
									className="h-12"
									/>
								</div>
							</motion.div>

							{/* Email input */}
							<motion.div variants={itemVariants}>
								<div className="space-y-2">
									<Label htmlFor="email">Email</Label>
									<Input
									id="email"
									type="email"
									placeholder="Enter your email"
									value={email}
									onChange={(e) => setEmail(e.target.value)}
									required
									className="h-12"
									/>
								</div>
							</motion.div>

							{/* Password input */}
							<motion.div variants={itemVariants}>
								<div className="space-y-2">
									<Label htmlFor="password">Password</Label>
									<div className="relative">
									<Input
										id="password"
										type={showPassword ? "text" : "password"}
										placeholder="Create a password"
										value={password}
										onChange={(e) => setPassword(e.target.value)}
										required
										className="h-12 pr-10"
									/>
										<button
											type="button"
											onClick={() => setShowPassword(!showPassword)}
											className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
										>
											{showPassword ? (
											<EyeOffIcon className="h-5 w-5" />
											) : (
											<EyeIcon className="h-5 w-5" />
											)}
										</button>
									</div>
								</div>
							</motion.div>

							{/* Confirm Password input */}
							<motion.div variants={itemVariants}>
								<div className="space-y-2">
									<Label htmlFor="confirmPassword">Confirm Password</Label>
									<div className="relative">
										<Input
											id="confirmPassword"
											type={showConfirmPassword ? "text" : "password"}
											placeholder="Confirm your password"
											value={confirmPassword}
											onChange={(e) => setConfirmPassword(e.target.value)}
											required
											className="h-12 pr-10"
										/>
										<button
											type="button"
											onClick={() => setShowConfirmPassword(!showConfirmPassword)}
											className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
										>
											{showConfirmPassword ? (
											<EyeOffIcon className="h-5 w-5" />
											) : (
											<EyeIcon className="h-5 w-5" />
											)}
										</button>
									</div>
								</div>
							</motion.div>

							{/* Terms and conditions */}
							<motion.div variants={itemVariants} className="text-sm text-gray-600 dark:text-gray-400">
							By creating an account, you agree to our{" "}
							<Link
								to="/terms"
								className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
							>
								Terms of Service
							</Link>{" "}
							and{" "}
							<Link
								to="/privacy"
								className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
							>
								Privacy Policy
							</Link>
							.
							</motion.div>

							{/* Sign up button */}
							<motion.div variants={itemVariants}>
								<Button
									type="submit"
									className="w-full h-12 text-base"
									disabled={loading}
								>
									{loading ? "Creating Account..." : "Create Account"}
								</Button>
							</motion.div>

							{/* Login link */}
							<motion.div variants={itemVariants} className="text-center mt-6">
								<p className="text-gray-600 dark:text-gray-400">
									Already have an account?{" "}
									<Link
									to="/login"
									className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
									>
									Login
									</Link>
								</p>
							</motion.div>
						</motion.form>
					</div>
				</motion.div>
			</div>
			<ToastContainer/>
		</>
	)
}
