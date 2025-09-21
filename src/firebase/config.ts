import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDA4wfcXqt-JAvPF2i2y948NLceP1wyfxk",
  authDomain: "euler-34c56.firebaseapp.com",
  projectId: "euler-34c56",
  storageBucket: "euler-34c56.firebasestorage.app",
  messagingSenderId: "959714192194",
  appId: "1:959714192194:web:a96dd65d664df715c2b6db",
  measurementId: "G-ZHKZPLHYQV"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);

// Initialize Analytics (only in browser environment)
export const analytics = typeof window !== 'undefined' ? getAnalytics(app) : null;

export default app;