import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword,
  signOut,
  GoogleAuthProvider,
  signInWithPopup,
  sendEmailVerification
} from 'firebase/auth';
import { auth } from '@/firebase/config';

// Google Sign In
export const signInWithGoogle = async () => {
  try {
    const provider = new GoogleAuthProvider();
    const result = await signInWithPopup(auth, provider);
    return { user: result.user, error: null };
  } catch (error: any) {
    return { user: null, error: error.message };
  }
};

// Email Sign In
export const signInWithEmail = async (email: string, password: string) => {
  try {
    const result = await signInWithEmailAndPassword(auth, email, password);
    return { user: result.user, error: null };
  } catch (error: any) {
    return { user: null, error: error.message };
  }
};

// Email Sign Up
export const signUpWithEmail = async (email: string, password: string) => {
  try {
    const result = await createUserWithEmailAndPassword(auth, email, password);
    // Send email verification
    if (result.user) {
      await sendEmailVerification(result.user);
    }
    return { user: result.user, error: null };
  } catch (error: any) {
    return { user: null, error: error.message };
  }
};

// Sign Out
export const signOutUser = async () => {
  try {
    await signOut(auth);
    return { error: null };
  } catch (error: any) {
    return { error: error.message };
  }
};

// Simulate email code verification (for demo purposes)
export const verifyEmailCode = async (email: string, code: string) => {
  // In a real app, you would verify the code with your backend
  // For demo purposes, any 6-digit code will work
  if (code.length === 6 && /^\d+$/.test(code)) {
    try {
      // Create a temporary account with a default password
      // In production, you'd have a proper email verification flow
      const tempPassword = 'TempPassword123!';
      const result = await signUpWithEmail(email, tempPassword);
      return { success: true, error: null };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  } else {
    return { success: false, error: 'Invalid code format' };
  }
};