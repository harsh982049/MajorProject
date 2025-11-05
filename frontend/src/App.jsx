import React from 'react';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';
// import './App.css'

// Import your page components
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import SignUp from './pages/SignUp';
import StressDetection from './pages/StressDetection';
import PanicSOSChatbot from './pages/PanicSOSChatbot';
import RelaxationTherapyPage from './pages/RelaxationTherapyPage';
import ContactPage from './pages/ContactPage';
import TrackUserMovements from './pages/TrackUserMovements';
import StressBehaviour from './pages/StressBehaviour';

function App() {
	return (
		<Router>
			<Routes>
				<Route path="/" element={<Dashboard/>}/>
				<Route path="/contact" element={<ContactPage/>}/>

				{/* Login and Register routes */}
				<Route path="/login" element={<Login/>}/>
				<Route path="/signup" element={<SignUp/>}/>
				<Route path="/stress-detection" element={<StressDetection/>}/>
				<Route path="/panic-chatbot" element={<PanicSOSChatbot/>}/>
				<Route path="/music-relaxation" element={<RelaxationTherapyPage/>}/>
				<Route path="/track-movements" element={<TrackUserMovements/>}/>
				<Route path="/stress-behavior" element={<StressBehaviour />} />
			</Routes>
		</Router>
	)
}

export default App;
