import { useState,useEffect } from 'react';
import { useNavigate,useLocation} from 'react-router-dom';
import {
  collection,
  doc,
  getDoc,
  updateDoc,
  setDoc,
  onSnapshot
} from "firebase/firestore";
import { auth, db } from '../firebaseconfig';
const LeaderBoard = () => {
  const [leaderboardData, setLeaderboardData] = useState([]);

  
  useEffect(() => {
    const storesCollection = collection(db, 'scores');
    const unsubscribe = onSnapshot(storesCollection, (snapshot) => {
      const data = snapshot.docs.map((doc) => doc.data());
      setLeaderboardData(data);
    });

    return () => unsubscribe();
  }, []); 

  return (
    <div className="relative [background:linear-gradient(107.93deg,_#ffc989,_#fffcb9_48.96%,_rgba(255,_255,_255,_0))] w-full h-[885px] overflow-hidden flex flex-col py-[90px] px-[140px] box-border items-start justify-start text-left text-[76px] text-black font-roboto-flex">
      <div className="self-stretch flex-1 flex flex-col items-start justify-start gap-[32px]">
        <div className="self-stretch flex flex-row items-start justify-start">
          <h4 className="m-0 flex-1 relative text-[inherit] leading-[16px] font-semibold font-inherit flex items-center h-[69px]">
            Leader Board - Python
          </h4>
        </div>
        {leaderboardData.map((item, index) => (
          <button
            key={index}
            className="cursor-pointer py-[27px] px-9 bg-[transparent] self-stretch rounded-31xl shadow-[0px_10px_50px_rgba(255,_255,_255,_0.25)] flex flex-row items-center justify-between border-[1.5px] border-solid border-darkorange-200"
          >
            <b className="relative text-13xl inline-block font-roboto-flex text-darkorange-200 text-center w-[115px] shrink-0">
              {item.email.split('@')[0]}
            </b>
            <b className="relative text-13xl inline-block font-roboto-flex text-darkorange-200 text-center w-[115px] shrink-0">
              {item.score}
            </b>
          </button>
        ))}
      </div>
    </div>
  );
};

export default LeaderBoard;
