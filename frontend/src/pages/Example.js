

import React, { useState,useEffect } from 'react';

import Chatbot from 'react-chatbot-kit';
import 'react-chatbot-kit/build/main.css';
import ActionProvider from './ActionProvider';
import MessageParser from './MessageParser';
import config from './Config';
import { useNavigate } from 'react-router-dom';


// const videoList = [
//   'https://www.youtube.com/embed/QXeEoD0pB3E?rel=0',
//   'https://www.youtube.com/embed/hEgO047GxaQ?rel=0',
//   'https://www.youtube.com/embed/mbryl4MZJms?rel=0',
//   'https://www.youtube.com/embed/DWgzHbglNIo?rel=0',
//   'https://www.youtube.com/embed/Z3UsF2Hurew?rel=0',
//   'https://www.youtube.com/embed/TqPzwenhMj0?rel=0',
//   'https://www.youtube.com/embed/Eaz5e6M8tL4?rel=0',
//   'https://www.youtube.com/embed/Mf7eFtbVxFM?rel=0',
//   'https://www.youtube.com/embed/2IsF7DEtVjg?rel=0',
//   'https://www.youtube.com/embed/4V14G5_CNGg?rel=0',
//   'https://www.youtube.com/embed/1U8TI16AR4s?rel=0',
//   'https://www.youtube.com/embed/_OZIAHg5i7M?rel=0',
//   'https://www.youtube.com/embed/gCCVsvgR2KU?rel=0',
// ];

// const Example = () => {

//     useEffect(() => {
//       // Blog content to be stored
//       const blogContent = `
// # Python for Beginners 
// ## Programming Tutorial - YouTube


// Are you interested in learning programming in a fun and engaging way? Look no further! Welcome to Telusko Learnings, where Navin Reddy presents an exciting programming tutorial series for beginners focusing on the fastest-growing language, Python.

// ## Introduction

// In this comprehensive Python tutorial series, Navin Reddy will guide you from the very basics to mastering the Python programming language. The best part is, you don't need any prior programming experience! Whether you are a complete beginner or have never written a line of code before, this series is the perfect starting point.

// ## Learn Python from Scratch

// The primary goal of this tutorial series is to help you understand the language fundamentals and syntax of Python. Navin takes you on a step-by-step journey, ensuring that you grasp each concept thoroughly. By the end of the series, you'll have a solid foundation in Python, enabling you to build upon it and explore more advanced topics on your own.

// ## Fun and Engaging Learning Experience

// Learning online can sometimes be dull or distracting, but fear not! Navin Reddy is committed to making this Python learning experience as interesting and enjoyable as possible. He incorporates his engaging presence, animations, and various interactive elements to keep you motivated throughout the series.

// ## Free and Accessible

// The best part is that this Python programming tutorial series is freely available on YouTube! You can access all the episodes without any cost, making it an excellent resource for anyone eager to learn Python without breaking the bank.

// ## Join the Python Community

// Learning is even more rewarding when you share your knowledge with others. Before you embark on this exciting journey, make sure to subscribe to Navin Reddy's channel and share it with your friends. Let's grow the Python community together and encourage everyone to explore the fascinating world of programming.

// So, are you ready to dive into the alien world of Python programming? Grab your popcorn and start the first episode of the series today! Happy coding! 🚀🐍

// **Stay tuned for the upcoming episodes and explore the boundless possibilities of Python!**

// *If you have any questions or need further assistance, feel free to drop a comment on the YouTube videos, and the Telusko Learnings community will be there to help you.*
// `;
      
//       // Store the blog content in local storage
//       localStorage.setItem('blog', blogContent);
//     }, []);

//   const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
//   const navigate = useNavigate();



//   return (
//     <div className="relative bg-black w-full h-[1068px] overflow-hidden flex flex-col py-[68px] px-[109px] box-border items-center justify-start gap-[90px] text-center text-41xl text-black font-roboto">

//       <div className="self-stretch flex-1 flex flex-row items-start justify-start gap-[24px] text-left text-xl text-gray-300 lg:flex-col">
//         <div className="flex-1 flex flex-col items-start justify-start gap-[12px] lg:flex-[unset] lg:self-stretch">
//         <div className="self-stretch flex-1 flex-col items-start justify-start relative pb-[46.25%]">
//           <iframe
//             className="self-stretch absolute h-full w-full"
//             src={videoList[currentVideoIndex]}
//             frameBorder="0"
//             allowFullScreen
//             width={400}
//           />
//         </div>
//           <div className="self-stretch flex flex-col items-start justify-start gap-[12px]">
//             <div className="self-stretch flex flex-col items-start justify-start gap-[9px]">
//               <div className="self-stretch flex flex-col items-start justify-start">
//                 <div className="self-stretch font-manrope relative leading-[28px] font-semibold text-white">
//                  Python 101, Intro to Python
//                 </div>
//               </div>
//               <div className="self-stretch flex flex-row items-start justify-between text-base lg:flex-col">
//                 <div className="self-stretch flex-1 flex flex-row flex-wrap items-center justify-start lg:flex-col lg:flex-[unset] lg:self-stretch">
//                   <div className="flex-1 flex flex-row items-center justify-start gap-[49px] lg:flex-[unset] lg:self-stretch">
        
//                     <div className="flex flex-col items-center justify-start">
//                       <button 
//                       onClick={()=>{navigate('/ai-code')}}
//                       className="cursor-pointer  p-0 bg-white text-black relative rounded-lg w-[94.56px] h-9">
//                         <div className="absolute top-[calc(50%_-_18px)] left-[calc(50%_-_31.28px)] w-[62.56px] h-9 overflow-hidden">
//                           <div className="font-manrope absolute top-[10px] left-[calc(50%_-_31.28px)] text-sm leading-[36px] font-medium font-roboto  text-center flex items-center justify-center w-[62.76px] h-4">
//                             AICode
//                           </div>
//                         </div>
//                       </button>
//                     </div>

//                     <div className="flex flex-col items-center justify-start">
//                       <button 
//                       onClick={()=>{navigate('/ai-read')}}
//                       className="cursor-pointer  p-0 bg-white text-black relative rounded-lg w-[94.56px] h-9">
//                         <div className="absolute top-[calc(50%_-_18px)] left-[calc(50%_-_31.28px)] w-[62.56px] h-9 overflow-hidden">
//                           <div className="font-manrope absolute top-[10px] left-[calc(50%_-_31.28px)] text-sm leading-[36px] font-medium font-roboto text-black text-center flex items-center justify-center w-[62.76px] h-4">
//                             AIRead
//                           </div>
//                         </div>
//                       </button>
//                     </div>
//               <div className="w-[94.56px] flex flex-col items-center justify-start">
//                 <button
//                   className="cursor-pointer  p-0 bg-white text-black relative rounded-lg w-[94.56px] h-9"
//                   onClick={() => navigate('/leader-board')}

//                 >
//                   <div className="absolute top-[calc(50%_-_28px)] left-[calc(50%_-_59.5px)] overflow-hidden flex flex-row py-2.5 px-0 items-center justify-center">
//                     <div className="relative text-sm leading-[36px] font-medium font-manrope text-black text-center flex items-center justify-center w-[119px] shrink-0"
//                    >
//                       LeaderBoard
//                     </div>
//                   </div>
//                 </button>
//               </div>
             
//                   </div>
//                 </div>
//               </div>
//             </div>
//             <div className="self-stretch relative h-px" />
//           </div>
//         </div>
//         <div className="h-[763px] flex flex-col items-center justify-center gap-[8px] text-center text-black lg:hidden">
//           <div className="flex flex-row p-2.5 items-center justify-center ">
//           <Chatbot 
//   headerText="Mimi AI"
//   config={config}
//   messageParser={(params) => MessageParser(params, videoList[currentVideoIndex])}
//   actionProvider={ActionProvider}
// />
//           </div>
//           <div className="self-stretch flex-1 flex flex-row items-center justify-start gap-[8px]">
//             <div className="self-stretch flex-1 flex flex-row items-center justify-start">
//             <button className="self-stretch flex-1 relative leading-[12px] flex items-center justify-center" 
//             onClick={handleUpNextClick}>
//               Up Next
//           </button>

//             </div>
//             <div className="flex-1 flex flex-col items-end justify-center gap-[8px] text-left text-[10px] text-text-white">
//               <div className="bg-text-white overflow-hidden flex flex-col items-start justify-start relative gap-[10px]">
//                 <div className="relative w-[168px] h-[94px] z-[0]">
//                 <iframe
//               className="absolute h-full w-full top-[0px] right-[0px] bottom-[0px] left-[0px] max-w-full overflow-hidden max-h-full object-cover"
//               src={videoList[currentVideoIndex+1]}
//               frameBorder="0"
//               allowFullScreen
//             />
//                 </div>
//                 <div className="my-0 mx-[!important] absolute right-[4px] bottom-[4px] bg-black flex flex-row py-[3px] px-1 items-start justify-start z-[1]">
//                   <div className="relative">23:45</div>
//                 </div>
//               </div>
//               <div className="w-[168px] flex flex-col py-0 pr-2 pl-0 box-border items-start justify-start text-xs text-black">
//                 <div className="flex flex-col py-[7px] px-0 items-start justify-start">
//                   <div className="w-[145px] flex flex-row items-start justify-start">
//                     <div className="flex-1 relative leading-[18px]">
//                       Python Boolean
//                     </div>
//                   </div>
//                   <div className="w-[145px] flex flex-col py-0.5 px-0 box-border items-start justify-start text-gray-color">
//                     <div className="w-[145px] flex flex-row items-start justify-start">
//                       <div className="flex-1 relative leading-[18px]">
//                         Intermediate
//                       </div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Example;




// import React, { useState,useEffect } from 'react';
// import Chatbot from 'react-chatbot-kit';
// import 'react-chatbot-kit/build/main.css';
// import ActionProvider from './ActionProvider';
// import MessageParser from './MessageParser';
// import config from './Config';
// import { useNavigate } from 'react-router-dom';


// const videoList = [
//   'https://www.youtube.com/embed/QXeEoD0pB3E?rel=0',
//   'https://www.youtube.com/embed/hEgO047GxaQ?rel=0',
//   'https://www.youtube.com/embed/mbryl4MZJms?rel=0',
//   'https://www.youtube.com/embed/DWgzHbglNIo?rel=0',
//   'https://www.youtube.com/embed/Z3UsF2Hurew?rel=0',
//   'https://www.youtube.com/embed/TqPzwenhMj0?rel=0',
//   'https://www.youtube.com/embed/Eaz5e6M8tL4?rel=0',
//   'https://www.youtube.com/embed/Mf7eFtbVxFM?rel=0',
//   'https://www.youtube.com/embed/2IsF7DEtVjg?rel=0',
//   'https://www.youtube.com/embed/4V14G5_CNGg?rel=0',
//   'https://www.youtube.com/embed/1U8TI16AR4s?rel=0',
//   'https://www.youtube.com/embed/_OZIAHg5i7M?rel=0',
//   'https://www.youtube.com/embed/gCCVsvgR2KU?rel=0',
// ];

// const Example = () => {

//     useEffect(() => {
//       // Blog content to be stored
//       const blogContent = `
// # Python for Beginners 
// ## Programming Tutorial - YouTube


// Are you interested in learning programming in a fun and engaging way? Look no further! Welcome to Telusko Learnings, where Navin Reddy presents an exciting programming tutorial series for beginners focusing on the fastest-growing language, Python.

// ## Introduction

// In this comprehensive Python tutorial series, Navin Reddy will guide you from the very basics to mastering the Python programming language. The best part is, you don't need any prior programming experience! Whether you are a complete beginner or have never written a line of code before, this series is the perfect starting point.

// ## Learn Python from Scratch

// The primary goal of this tutorial series is to help you understand the language fundamentals and syntax of Python. Navin takes you on a step-by-step journey, ensuring that you grasp each concept thoroughly. By the end of the series, you'll have a solid foundation in Python, enabling you to build upon it and explore more advanced topics on your own.

// ## Fun and Engaging Learning Experience

// Learning online can sometimes be dull or distracting, but fear not! Navin Reddy is committed to making this Python learning experience as interesting and enjoyable as possible. He incorporates his engaging presence, animations, and various interactive elements to keep you motivated throughout the series.

// ## Free and Accessible

// The best part is that this Python programming tutorial series is freely available on YouTube! You can access all the episodes without any cost, making it an excellent resource for anyone eager to learn Python without breaking the bank.

// ## Join the Python Community

// Learning is even more rewarding when you share your knowledge with others. Before you embark on this exciting journey, make sure to subscribe to Navin Reddy's channel and share it with your friends. Let's grow the Python community together and encourage everyone to explore the fascinating world of programming.

// So, are you ready to dive into the alien world of Python programming? Grab your popcorn and start the first episode of the series today! Happy coding! 🚀🐍

// **Stay tuned for the upcoming episodes and explore the boundless possibilities of Python!**

// *If you have any questions or need further assistance, feel free to drop a comment on the YouTube videos, and the Telusko Learnings community will be there to help you.*
// `;
      
//       // Store the blog content in local storage
//       localStorage.setItem('blog', blogContent);
//     }, []);

//   const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
//   const navigate = useNavigate();

//   const handleUpNextClick = () => {
//     if (currentVideoIndex < videoList.length - 1) {
//       setCurrentVideoIndex((prevIndex) => prevIndex + 1);

//     }
//   };

//   return (
//     <div className="relative [background:linear-gradient(111.56deg,_#ffcc8c,_#fffdc9_48.96%,_rgba(255,_255,_255,_0))] w-full h-[1068px] overflow-hidden flex flex-col py-[68px] px-[109px] box-border items-center justify-start gap-[90px] text-center text-41xl text-black font-roboto">
//       <div className="self-stretch flex flex-col items-center justify-center">
//         <div className="self-stretch relative leading-[75.5px] capitalize font-medium">
//           Try it out
//         </div>
//       </div>
//       <div className="self-stretch flex-1 flex flex-row items-start justify-start gap-[24px] text-left text-xl text-gray-300 lg:flex-col">
//         <div className="flex-1 flex flex-col items-start justify-start gap-[12px] lg:flex-[unset] lg:self-stretch">
//           <div className="self-stretch flex flex-col items-start justify-start relative pb-[56.25%]">
//             <iframe
//               className="self-stretch absolute bg-gainsboro  top-[0] left-[0] z-[0]"
//               src={videoList[currentVideoIndex]}
//               frameBorder="0"
//               allowFullScreen
//             />
//           </div>
//           <div className="self-stretch flex flex-col items-start justify-start gap-[12px]">
//             <div className="self-stretch flex flex-col items-start justify-start gap-[9px]">
//               <div className="self-stretch flex flex-col items-start justify-start">
//                 <div className="self-stretch relative leading-[28px] font-semibold">
//                  Python 101, Intro to Python
//                 </div>
//               </div>
//               <div className="self-stretch flex flex-row items-start justify-between text-base lg:flex-col">
//                 <div className="self-stretch flex-1 flex flex-row flex-wrap items-center justify-start lg:flex-col lg:flex-[unset] lg:self-stretch">
//                   <div className="flex-1 flex flex-row items-center justify-start gap-[49px] lg:flex-[unset] lg:self-stretch">
//                     <div className="flex flex-row items-center justify-start gap-[12px]">
//                       <div className="rounded-xl overflow-hidden flex flex-row items-center justify-start">
//                         <img
//                           className="relative w-10 h-10 object-cover"
//                           alt=""
//                           src="/gqvpuag9wayv3emkh5utoqkdvhimgla2vtqebhu2rp0zcc1i4ta6opb90ysan3swuvbevm8s48ckc0x00ffffffnorj@2x.png"
//                         />
//                       </div>
//                       <div className="overflow-hidden flex flex-col py-px px-0 items-start justify-center">
//                         <div className="overflow-hidden flex flex-col items-start justify-center">
//                           <div className="overflow-hidden flex flex-col items-start justify-center">
//                             <div className="overflow-hidden flex flex-row py-px px-0 items-start justify-center">
//                               <div className="relative leading-[22px] font-medium flex items-center w-[55px] h-[19px] shrink-0">
//                                 AITube
//                               </div>
//                             </div>
//                           </div>
//                         </div>
//                         <div className="relative w-[96.19px] h-[18px] overflow-hidden shrink-0 text-xs text-dimgray-100">
//                           <div className="absolute w-[100.21%] top-[calc(50%_-_7px)] left-[0%] leading-[18px] flex items-center h-3.5">
//                             1.22K subscribers
//                           </div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="flex flex-col items-center justify-start">
//                       <button 
//                       onClick={()=>{navigate('/ai-code')}}
//                       className="cursor-pointer [border:none] p-0 bg-darkorange-200 relative rounded-lg w-[94.56px] h-9">
//                         <div className="absolute top-[calc(50%_-_18px)] left-[calc(50%_-_31.28px)] w-[62.56px] h-9 overflow-hidden">
//                           <div className="absolute top-[10px] left-[calc(50%_-_31.28px)] text-sm leading-[36px] font-medium font-roboto text-text-white text-center flex items-center justify-center w-[62.76px] h-4">
//                             AICode
//                           </div>
//                         </div>
//                       </button>
//                     </div>
//                     <div className="flex flex-col items-center justify-start">
//                       <button 
//                       onClick={()=>{navigate('/ai-read')}}
//                       className="cursor-pointer [border:none] p-0 bg-darkorange-200 relative rounded-lg w-[94.56px] h-9">
//                         <div className="absolute top-[calc(50%_-_18px)] left-[calc(50%_-_31.28px)] w-[62.56px] h-9 overflow-hidden">
//                           <div className="absolute top-[10px] left-[calc(50%_-_31.28px)] text-sm leading-[36px] font-medium font-roboto text-text-white text-center flex items-center justify-center w-[62.76px] h-4">
//                             AIRead
//                           </div>
//                         </div>
//                       </button>
//                     </div>
//               <div className="w-[94.56px] flex flex-col items-center justify-start">
//                 <button
//                   className="cursor-pointer [border:none] p-0 bg-darkorange-200 self-stretch relative rounded-lg h-9"
//                   onClick={() => navigate('/leader-board')}

//                 >
//                   <div className="absolute top-[calc(50%_-_28px)] left-[calc(50%_-_59.5px)] overflow-hidden flex flex-row py-2.5 px-0 items-center justify-center">
//                     <div className="relative text-sm leading-[36px] font-medium font-roboto text-text-white text-center flex items-center justify-center w-[119px] shrink-0"
//                    >
//                       LeaderBoard
//                     </div>
//                   </div>
//                 </button>
//               </div>
//               <div className="flex flex-row items-center justify-center">
//                       <div className="flex flex-row items-start justify-start">
//                         <button className="cursor-pointer [border:none] p-0 bg-text-white rounded-81xl flex flex-row items-center justify-center">
//                           <div className="flex flex-col items-start justify-start">
//                             <button className="cursor-pointer [border:none] py-0 px-2.5 bg-[transparent] rounded-tl-lg rounded-tr-none rounded-br-none rounded-bl-lg flex flex-row items-center justify-start gap-[6px]">
//                               <img
//                                 className="relative w-6 h-6 overflow-hidden shrink-0"
//                                 alt=""
//                                 src="/svg2.svg"
//                               />
//                               <div className="relative text-sm leading-[36px] font-medium font-roboto text-darkorange-200 text-center">
//                                 169
//                               </div>
//                             </button>
//                           </div>
//                           <div className="relative bg-gray-600 w-px h-6" />
//                           <img
//                             className="relative w-[52px] h-9"
//                             alt=""
//                             src="/ytdtogglebuttonrenderer.svg"
//                           />
//                         </button>
//                       </div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </div>
//             <div className="self-stretch relative h-px" />
//           </div>
//         </div>
//         <div className="h-[763px] flex flex-col items-center justify-center gap-[8px] text-center text-black lg:hidden">
//           <div className="flex flex-row p-2.5 items-center justify-center h-[780px]">
//           <Chatbot 
//   headerText="Mimi AI"
//   config={config}
//   messageParser={(params) => MessageParser(params, videoList[currentVideoIndex])}
//   actionProvider={ActionProvider}
// />


//           </div>
//           <div className="self-stretch flex-1 flex flex-row items-center justify-start gap-[8px]">
//             <div className="self-stretch flex-1 flex flex-row items-center justify-start">
//             <button className="self-stretch flex-1 relative leading-[12px] flex items-center justify-center" 
//             onClick={handleUpNextClick}>
//               Up Next
//           </button>

//             </div>
//             <div className="flex-1 flex flex-col items-end justify-center gap-[8px] text-left text-[10px] text-text-white">
//               <div className="bg-text-white overflow-hidden flex flex-col items-start justify-start relative gap-[10px]">
//                 <div className="relative w-[168px] h-[94px] z-[0]">
//                 <iframe
//               className="absolute h-full w-full top-[0px] right-[0px] bottom-[0px] left-[0px] max-w-full overflow-hidden max-h-full object-cover"
//               src={videoList[currentVideoIndex+1]}
//               frameBorder="0"
//               allowFullScreen
//             />
//                 </div>
//                 <div className="my-0 mx-[!important] absolute right-[4px] bottom-[4px] bg-black flex flex-row py-[3px] px-1 items-start justify-start z-[1]">
//                   <div className="relative">23:45</div>
//                 </div>
//               </div>
//               <div className="w-[168px] flex flex-col py-0 pr-2 pl-0 box-border items-start justify-start text-xs text-black">
//                 <div className="flex flex-col py-[7px] px-0 items-start justify-start">
//                   <div className="w-[145px] flex flex-row items-start justify-start">
//                     <div className="flex-1 relative leading-[18px]">
//                       Python Boolean
//                     </div>
//                   </div>
//                   <div className="w-[145px] flex flex-col py-0.5 px-0 box-border items-start justify-start text-gray-color">
//                     <div className="w-[145px] flex flex-row items-start justify-start">
//                       <div className="flex-1 relative leading-[18px]">
//                         Intermediate
//                       </div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Example;




// import React, { useState,useEffect } from 'react';
// import Chatbot from 'react-chatbot-kit';
// import 'react-chatbot-kit/build/main.css';
// import ActionProvider from './ActionProvider';
// import MessageParser from './MessageParser';
// import config from './Config';
// import { useNavigate } from 'react-router-dom';


// const videoList = [
//   'https://www.youtube.com/embed/QXeEoD0pB3E?rel=0',
//   'https://www.youtube.com/embed/hEgO047GxaQ?rel=0',
//   'https://www.youtube.com/embed/mbryl4MZJms?rel=0',
//   'https://www.youtube.com/embed/DWgzHbglNIo?rel=0',
//   'https://www.youtube.com/embed/Z3UsF2Hurew?rel=0',
//   'https://www.youtube.com/embed/TqPzwenhMj0?rel=0',
//   'https://www.youtube.com/embed/Eaz5e6M8tL4?rel=0',
//   'https://www.youtube.com/embed/Mf7eFtbVxFM?rel=0',
//   'https://www.youtube.com/embed/2IsF7DEtVjg?rel=0',
//   'https://www.youtube.com/embed/4V14G5_CNGg?rel=0',
//   'https://www.youtube.com/embed/1U8TI16AR4s?rel=0',
//   'https://www.youtube.com/embed/_OZIAHg5i7M?rel=0',
//   'https://www.youtube.com/embed/gCCVsvgR2KU?rel=0',
// ];

// const Example = () => {

//     useEffect(() => {
//       // Blog content to be stored
//       const blogContent = `
// # Python for Beginners 
// ## Programming Tutorial - YouTube


// Are you interested in learning programming in a fun and engaging way? Look no further! Welcome to Telusko Learnings, where Navin Reddy presents an exciting programming tutorial series for beginners focusing on the fastest-growing language, Python.

// ## Introduction

// In this comprehensive Python tutorial series, Navin Reddy will guide you from the very basics to mastering the Python programming language. The best part is, you don't need any prior programming experience! Whether you are a complete beginner or have never written a line of code before, this series is the perfect starting point.

// ## Learn Python from Scratch

// The primary goal of this tutorial series is to help you understand the language fundamentals and syntax of Python. Navin takes you on a step-by-step journey, ensuring that you grasp each concept thoroughly. By the end of the series, you'll have a solid foundation in Python, enabling you to build upon it and explore more advanced topics on your own.

// ## Fun and Engaging Learning Experience

// Learning online can sometimes be dull or distracting, but fear not! Navin Reddy is committed to making this Python learning experience as interesting and enjoyable as possible. He incorporates his engaging presence, animations, and various interactive elements to keep you motivated throughout the series.

// ## Free and Accessible

// The best part is that this Python programming tutorial series is freely available on YouTube! You can access all the episodes without any cost, making it an excellent resource for anyone eager to learn Python without breaking the bank.

// ## Join the Python Community

// Learning is even more rewarding when you share your knowledge with others. Before you embark on this exciting journey, make sure to subscribe to Navin Reddy's channel and share it with your friends. Let's grow the Python community together and encourage everyone to explore the fascinating world of programming.

// So, are you ready to dive into the alien world of Python programming? Grab your popcorn and start the first episode of the series today! Happy coding! 🚀🐍

// **Stay tuned for the upcoming episodes and explore the boundless possibilities of Python!**

// *If you have any questions or need further assistance, feel free to drop a comment on the YouTube videos, and the Telusko Learnings community will be there to help you.*
// `;
      
//       // Store the blog content in local storage
//       localStorage.setItem('blog', blogContent);
//     }, []);

//   const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
//   const navigate = useNavigate();

//   const handleUpNextClick = () => {
//     if (currentVideoIndex < videoList.length - 1) {
//       setCurrentVideoIndex((prevIndex) => prevIndex + 1);

//     }
//   };

//   return (
//     <div className="relative [background:linear-gradient(111.56deg,_#ffcc8c,_#fffdc9_48.96%,_rgba(255,_255,_255,_0))]  h-[1068px] flex flex-col py-[68px] px-[109px] box-border items-center justify-start gap-[90px] text-center text-41xl text-black font-roboto">
//       <div className="self-stretch flex flex-col items-center justify-center">
//         <div className="self-stretch relative  capitalize font-medium">
//           Try it out
//         </div>
//       </div>
//       <div className="self-stretch flex-1 flex flex-row items-start justify-start gap-[24px] text-left text-xl text-gray-300 lg:flex-col">
//         <div className="flex-1 flex flex-col items-start justify-start gap-[12px] lg:flex-[unset] lg:self-stretch">
//           <div className=" flex flex-col items-start justify-start relative pb-[56.25%]">
//             <iframe
//               className=" bg-gainsboro h-full overflow-hidden shrink-0 top-[0] left-[0] w-full z-[0]"
//               src={videoList[currentVideoIndex]}
//               frameBorder="0"
//               allowFullScreen
//             />
//           </div>
//           <div className="self-stretch flex flex-col items-start justify-start gap-[12px]">
//             <div className="self-stretch flex flex-col items-start justify-start gap-[9px]">
//               <div className="self-stretch flex flex-col items-start justify-start">
//                 <div className="self-stretch relative leading-[28px] font-semibold">
//                  Python 101, Intro to Python
//                 </div>
//               </div>
//               <div className="self-stretch flex flex-row items-start justify-between text-base lg:flex-col">
//                 <div className="self-stretch flex-1 flex flex-row flex-wrap items-center justify-start lg:flex-col lg:flex-[unset] lg:self-stretch">
//                   <div className="flex-1 flex flex-row items-center justify-start gap-[49px] lg:flex-[unset] lg:self-stretch">
//                     <div className="flex flex-row items-center justify-start gap-[12px]">
//                       <div className="rounded-xl overflow-hidden flex flex-row items-center justify-start">
//                         <img
//                           className="relative w-10 h-10 object-cover"
//                           alt=""
//                           src="/gqvpuag9wayv3emkh5utoqkdvhimgla2vtqebhu2rp0zcc1i4ta6opb90ysan3swuvbevm8s48ckc0x00ffffffnorj@2x.png"
//                         />
//                       </div>
//                       <div className="overflow-hidden flex flex-col py-px px-0 items-start justify-center">
//                         <div className="overflow-hidden flex flex-col items-start justify-center">
//                           <div className="overflow-hidden flex flex-col items-start justify-center">
//                             <div className="overflow-hidden flex flex-row py-px px-0 items-start justify-center">
//                               <div className="relative leading-[22px] font-medium flex items-center w-[55px] h-[19px] shrink-0">
//                                 AITube
//                               </div>
//                             </div>
//                           </div>
//                         </div>
//                         <div className="relative w-[96.19px] h-[18px] overflow-hidden shrink-0 text-xs text-dimgray-100">
//                           <div className="absolute w-[100.21%] top-[calc(50%_-_7px)] left-[0%] leading-[18px] flex items-center h-3.5">
//                             1.22K subscribers
//                           </div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="flex flex-col items-center justify-start">
//                       <button 
//                       onClick={()=>{navigate('/ai-code')}}
//                       className="cursor-pointer [border:none] p-0 bg-darkorange-200 relative rounded-lg w-[94.56px] h-9">
//                         <div className="absolute top-[calc(50%_-_18px)] left-[calc(50%_-_31.28px)] w-[62.56px] h-9 overflow-hidden">
//                           <div className="absolute top-[10px] left-[calc(50%_-_31.28px)] text-sm leading-[36px] font-medium font-roboto text-text-white text-center flex items-center justify-center w-[62.76px] h-4">
//                             AICode
//                           </div>
//                         </div>
//                       </button>
//                     </div>
//                     <div className="flex flex-col items-center justify-start">
//                       <button 
//                       onClick={()=>{navigate('/ai-read')}}
//                       className="cursor-pointer [border:none] p-0 bg-darkorange-200 relative rounded-lg w-[94.56px] h-9">
//                         <div className="absolute top-[calc(50%_-_18px)] left-[calc(50%_-_31.28px)] w-[62.56px] h-9 overflow-hidden">
//                           <div className="absolute top-[10px] left-[calc(50%_-_31.28px)] text-sm leading-[36px] font-medium font-roboto text-text-white text-center flex items-center justify-center w-[62.76px] h-4">
//                             AIRead
//                           </div>
//                         </div>
//                       </button>
//                     </div>
//               <div className="w-[94.56px] flex flex-col items-center justify-start">
//                 <button
//                   className="cursor-pointer [border:none] p-0 bg-darkorange-200 self-stretch relative rounded-lg h-9"
//                   onClick={() => navigate('/leader-board')}

//                 >
//                   <div className="absolute top-[calc(50%_-_28px)] left-[calc(50%_-_59.5px)] overflow-hidden flex flex-row py-2.5 px-0 items-center justify-center">
//                     <div className="relative text-sm leading-[36px] font-medium font-roboto text-text-white text-center flex items-center justify-center w-[119px] shrink-0"
//                    >
//                       LeaderBoard
//                     </div>
//                   </div>
//                 </button>
//               </div>
//               <div className="flex flex-row items-center justify-center">
//                       <div className="flex flex-row items-start justify-start">
//                         <button className="cursor-pointer [border:none] p-0 bg-text-white rounded-81xl flex flex-row items-center justify-center">
//                           <div className="flex flex-col items-start justify-start">
//                             <button className="cursor-pointer [border:none] py-0 px-2.5 bg-[transparent] rounded-tl-lg rounded-tr-none rounded-br-none rounded-bl-lg flex flex-row items-center justify-start gap-[6px]">
//                               <img
//                                 className="relative w-6 h-6 overflow-hidden shrink-0"
//                                 alt=""
//                                 src="/svg2.svg"
//                               />
//                               <div className="relative text-sm leading-[36px] font-medium font-roboto text-darkorange-200 text-center">
//                                 169
//                               </div>
//                             </button>
//                           </div>
//                           <div className="relative bg-gray-600 w-px h-6" />
//                           <img
//                             className="relative w-[52px] h-9"
//                             alt=""
//                             src="/ytdtogglebuttonrenderer.svg"
//                           />
//                         </button>
//                       </div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </div>
//             <div className="self-stretch relative h-px" />
//           </div>
//         </div>
//         <div className="h-[763px] flex flex-col items-center justify-center gap-[8px] text-center text-black lg:hidden">
//           <div className="flex flex-row p-2.5 items-center justify-center h-[780px]">
//           <Chatbot 
//   headerText="Mimi AI"
//   config={config}
//   messageParser={(params) => MessageParser(params, videoList[currentVideoIndex])}
//   actionProvider={ActionProvider}
// />


//           </div>
//           <div className="self-stretch flex-1 flex flex-row items-center justify-start gap-[8px]">
//             <div className="self-stretch flex-1 flex flex-row items-center justify-start">
//             <button className="self-stretch flex-1 relative leading-[12px] flex items-center justify-center" 
//             onClick={handleUpNextClick}>
//               Up Next
//           </button>

//             </div>
//             <div className="flex-1 flex flex-col items-end justify-center gap-[8px] text-left text-[10px] text-text-white">
//               <div className="bg-text-white overflow-hidden flex flex-col items-start justify-start relative gap-[10px]">
//                 <div className="relative w-[168px] h-[94px] z-[0]">
//                 <iframe
//               className="absolute h-full w-full top-[0px] right-[0px] bottom-[0px] left-[0px] max-w-full overflow-hidden max-h-full object-cover"
//               src={videoList[currentVideoIndex+1]}
//               frameBorder="0"
//               allowFullScreen
//             />
//                 </div>
//                 <div className="my-0 mx-[!important] absolute right-[4px] bottom-[4px] bg-black flex flex-row py-[3px] px-1 items-start justify-start z-[1]">
//                   <div className="relative">23:45</div>
//                 </div>
//               </div>
//               <div className="w-[168px] flex flex-col py-0 pr-2 pl-0 box-border items-start justify-start text-xs text-black">
//                 <div className="flex flex-col py-[7px] px-0 items-start justify-start">
//                   <div className="w-[145px] flex flex-row items-start justify-start">
//                     <div className="flex-1 relative leading-[18px]">
//                       Python Boolean
//                     </div>
//                   </div>
//                   <div className="w-[145px] flex flex-col py-0.5 px-0 box-border items-start justify-start text-gray-color">
//                     <div className="w-[145px] flex flex-row items-start justify-start">
//                       <div className="flex-1 relative leading-[18px]">
//                         Intermediate
//                       </div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Example;



// import React from 'react';
// import Chatbot from './Chatbot'; // Import the Chatbot component

//   return (
//     <div className='flex bg-black p-2 pb-[1000px]'>
//     <div className="w-full h-full">
//       {/* YouTube Embed */}
//       <iframe
//         className="w-full h-full"
//         src={videoList[currentVideoIndex]}
//         frameBorder="0"
//         height={200}
//         allowFullScreen
//       />

//     <h2 className="text-xl font-bold">Video Title</h2>
//     <div className="flex space-x-2">
//       <button className="bg-blue-500 text-white px-4 py-2">Button 1</button>
//       <button className="bg-red-500 text-white px-4 py-2">Button 2</button>
//       <button className="bg-green-500 text-white px-4 py-2">Button 3</button>
//       <button className="bg-yellow-500 text-white px-4 py-2">Button 4</button>
//       {/* ... */}
//     </div>
//     <p className="flex-1">Description of the video...</p>
//   </div>

//   <div className="h-max">
//     {/* Chatbot */}
//     <Chatbot
//       headerText="Mimi AI"
//       config={config}
//       messageParser={(params) => MessageParser(params, videoList[currentVideoIndex])}
//       actionProvider={ActionProvider}
//     />
//     <div className="flex items-start justify-start space-x-2 mt-4">
//       {/* Next Video Button */}
//       <button 
//         onClick={handleUpNextClick}
//         className="bg-black text-white px-4 py-2">
//         Next Video
//       </button>
//       {/* Iframe */}
//       <div className=" relative">
//         <iframe
//           className="absolute w-full h-full top-0 left-0"
//           src={videoList[currentVideoIndex + 1]}
//           frameBorder="0"
//           allowFullScreen
//         />
//       </div>
//     </div>
//   </div>
// </div>

//   )  


const Example = () => {
  

  const videoList = [
    'https://www.youtube.com/embed/QXeEoD0pB3E?rel=0',
    'https://www.youtube.com/embed/hEgO047GxaQ?rel=0',
    'https://www.youtube.com/embed/mbryl4MZJms?rel=0',
    'https://www.youtube.com/embed/DWgzHbglNIo?rel=0',
    'https://www.youtube.com/embed/Z3UsF2Hurew?rel=0',
    'https://www.youtube.com/embed/TqPzwenhMj0?rel=0',
    'https://www.youtube.com/embed/Eaz5e6M8tL4?rel=0',
    'https://www.youtube.com/embed/Mf7eFtbVxFM?rel=0',
    'https://www.youtube.com/embed/2IsF7DEtVjg?rel=0',
    'https://www.youtube.com/embed/4V14G5_CNGg?rel=0',
    'https://www.youtube.com/embed/1U8TI16AR4s?rel=0',
    'https://www.youtube.com/embed/_OZIAHg5i7M?rel=0',
    'https://www.youtube.com/embed/gCCVsvgR2KU?rel=0',
  ];
  
    const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  
    const handleUpNextClick = () => {
      if (currentVideoIndex < videoList.length - 1) {
        setCurrentVideoIndex((prevIndex) => prevIndex + 1);
  
      }
    };

    const [disturbed, setDisturbed] = useState(false);

    const checkDisturbance = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/camera', {
          method: 'POST',
        });
  
        if (response.ok) {
          const data = await response.json();
          setDisturbed(data.disturbed);
        } else {
          console.error('Failed to fetch disturbance status');
        }
      } catch (error) {
        console.error('Error fetching disturbance status:', error);
      }
    };
  
    // Fetch disturbance status every 1 second
    useEffect(() => {
      const intervalId = setInterval(checkDisturbance, 1000);
  
      return () => {
        clearInterval(intervalId);
      };
    }, []);

return (
  <div className='flex bg-black p-2 pb-[1000px] space-x-2 pt-4'>
      <div className="w-full bg-black">
          {/* YouTube Embed */}
          <iframe
            className="w-full h-full"
            src={videoList[currentVideoIndex]}
            frameBorder="0"
            height={200}
            allowFullScreen
          />

        <h2 className="text-xl font-bold">Video Title</h2>
        <div className='flex justify-between'>
        <div className="flex space-x-2">
          <button className="bg-white rounded-md  text-black px-4 py-2">Ai Code</button>
          <button className="bg-white rounded-md  text-black px-4 py-2">Ai Read</button>
          <button className="bg-white rounded-md  text-black px-4 py-2">Ai Quiz</button>
          <button className="bg-white rounded-md  text-black px-4 py-2">Leader Board</button>
          {/* ... */}
        </div>
        <div className="flex space-x-2">
          <button className="bg-gray-600 rounded-md text-white px-4 py-2">Completed ➡</button>
          {/* ... */}
        </div>
        </div>
        <p className="flex-1 text-white mt-4 font-manrope">Learning Python is the greatest thing that you can do for yourselves, believe me you will thank me later.</p>
      </div>


<div className="h-max">
        {/* Chatbot */}
        <Chatbot
          headerText="Mimi AI"
          config={config}
          messageParser={(params) => MessageParser(params, videoList[currentVideoIndex])}
          actionProvider={ActionProvider}
        />
        <div className="flex items-start justify-start space-x-2 mt-4">
          {/* Next Video Button */}
          <button 
            onClick={handleUpNextClick}
            className=" text-white rounded-2 rounded-md border-dashed border-2 border-white px-4 py-2">
            Next Video
          </button>
          {/* Iframe */}
          <div className=" relative">
            <iframe
              className="absolute w-full h-full top-0 left-0"
              src={videoList[currentVideoIndex + 1]}
              frameBorder="0"
              allowFullScreen
            />
          </div>
        </div>

        {(disturbed &&
          <div className="absolute left-8 top-1/4 transform translate-y-1/2 bg-red-500 text-white p-2 rounded shadow-md">
            Warning: You are disturbed!
          </div>
        )}

      </div>

      </div>
)
  
};

export default Example;
