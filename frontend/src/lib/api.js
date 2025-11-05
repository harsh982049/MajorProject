import axios from "axios"

export const API_BASE = "http://127.0.0.1:5000"

export const api = axios.create({
  baseURL: API_BASE,
})

function getToken() {
  return localStorage.getItem("token") || ""
}


// Attach JWT from localStorage if present
api.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export { getToken }
