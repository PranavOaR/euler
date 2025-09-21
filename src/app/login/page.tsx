'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStatus } from '@/context/AuthContext';
import { SignInPage } from '@/components/ui/sign-in-flow-1';

export default function LoginPage() {
  const { user, loading } = useAuthStatus();
  const router = useRouter();

  useEffect(() => {
    // If auth state is not loading and there is a user, redirect to dashboard
    if (!loading && user) {
      router.push('/dashboard');
    }
  }, [user, loading, router]);

  // Show loading state while checking auth status
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
      </div>
    );
  }

  // If user is authenticated, show nothing (redirect is happening)
  if (user) {
    return null;
  }

  // Show login page for unauthenticated users
  return <SignInPage />;
}