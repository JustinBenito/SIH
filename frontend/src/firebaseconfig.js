// Import the functions you need from the SDKs you need
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import 'firebase/firestore';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAU2aHKqKMFQatNf9-179RryhGHzM_BDxw",
  authDomain: "aitube-ad6e9.firebaseapp.com",
  projectId: "aitube-ad6e9",
  storageBucket: "aitube-ad6e9.appspot.com",
  messagingSenderId: "521878532819",
  appId: "1:521878532819:web:275e917eef70f7cbc11a26"
};

// Initialize Firebase
initializeApp(firebaseConfig);

export const auth = getAuth();
export const db = getFirestore();