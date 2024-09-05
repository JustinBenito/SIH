import React, { useState,useEffect } from 'react';

import ReactMarkdown from 'react-markdown'
import ReactDom from 'react-dom'
const AIRead = () => {

  const [offline, setOffline] = useState(false);

  useEffect(() => {
    // Check for internet connectivity
    const checkConnectivity = () => {
      if (!navigator.onLine) {
        setOffline(true);
      } else {
        setOffline(false);
      }
    };

    // Check connectivity on page load
    checkConnectivity();

    // Add event listener to check connectivity changes
    window.addEventListener('online', checkConnectivity);
    window.addEventListener('offline', checkConnectivity);

    // Clean up event listeners
    return () => {
      window.removeEventListener('online', checkConnectivity);
      window.removeEventListener('offline', checkConnectivity);
    };
  }, []);
let storedBlog=""
storedBlog=localStorage.getItem('blog');
  if (offline) {
    storedBlog = localStorage.getItem('blog');
  }

  return (
    <div className="relative [background:linear-gradient(110.84deg,_#ffd293,_#fffab7_48.96%,_rgba(255,_255,_255,_0))] box-border w-full h-[1353px] overflow-hidden flex flex-col py-[68px] px-[109px] items-center justify-start gap-[69px] text-center text-41xl text-black font-roboto border-[1px] border-solid border-gray-600">
      <div className="self-stretch flex flex-col py-[17px] px-0 items-center justify-center">
        <div className="self-stretch relative leading-[75.5px] capitalize font-medium">
          AI Read
        </div>
      </div>
      <div className="self-stretch flex-1 rounded-xl flex flex-row items-center justify-center text-left text-13xl text-gray-400 font-inter">
        <div className="self-stretch flex-1 rounded-xl bg-text-white flex flex-col py-[107px] px-0 items-center justify-start">
          <div className="relative leading-[130%] inline-block w-full px-2 h-[1038px] shrink-0">


{<ReactMarkdown >{storedBlog}</ReactMarkdown>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIRead;
