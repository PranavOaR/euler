import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyA0lLDLPzTfkwzukbDTNJixw8oQDUfqvsU",
  authDomain: "euler-f5167.firebaseapp.com",
  projectId: "euler-f5167",
  storageBucket: "euler-f5167.firebasestorage.app",
  messagingSenderId: "872554004392",
  appId: "1:872554004392:web:9b896049200b019f9e679d",
  measurementId: "G-MCDV8MJ1HK"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);

// Initialize Analytics (only in browser environment)
export const analytics = typeof window !== 'undefined' ? getAnalytics(app) : null;

export default app;