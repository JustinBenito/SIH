import MonacoEditor from '@monaco-editor/react';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/themes/prism.css';
import React, { useState, useEffect } from "react";
import Typewriter from 'typewriter-effect';
import OpenAI from "openai";


const AICode = ({lang, skill, title}) => {

  const [resp,setResp]=useState({})
  const [enableTypewriter, setEnableTypewriter] = useState(false);

  const [explain, setExplain]=useState('')

  const [query, setQuery]=useState('')

  const [msg,setMsg]=useState('')
  
  useEffect(()=>{
    if(title !='' && skill!='' && lang!=''){
      const messages = [
        {role:'system',content: "Only use the function I provide"},
        { role: "user", content: `Give me a coding question for ${title} a ${skill} level. I want a JSON response.
        of the question, output, input like this
        ` },
      ];
      const functions = [
        {
          name: "getquestion",
          description: `Get a coding question which is highly relevant to this ${title} and can be solved by a user in ${skill} level in this ${lang} language . I want a JSON response.
          of the question, output, input like this 
          {
            question: "generated question",
            input: "what are the inputs",
            output: "what is the expected output for the generated question"
          }`,
          parameters: {
            type: "object",
            properties: {
              question: {
                type: "string",
                description: `The coding question, e.g Write a program to check palindrome`,
              },
              input: { 
                type: "string", 
                description: `the necessary input values eg. madam or 50`
              },
              output: { 
                type: "string", 
                description: `the expected output for the coding question with the input values eg. true or 65`
              },
            },
            required: ["question",'input','output'],
          },
        },
      ];

    async function fetchData() {
      try {
        // Your async code here
        const response = await openai.chat.completions.create({
          model: 'gpt-3.5-turbo-0613',
          messages: messages,
          functions: functions,
          function_call: 'auto',
        });
        
        console.log(response)
        const response_message = response.choices[0].message;
        if (!response_message.content) {
          const resp_args = JSON.parse(response_message.function_call.arguments);
          setResp(resp_args)
          setQues(resp_args.question)
          console.log('question: ', resp_args.question);
          console.log('input: ', resp_args.input);
          console.log('output: ', resp_args.output);
        }
      } catch (error) {
        if (error instanceof OpenAI.APIError) {
          console.error(error.status);  // e.g. 401
          console.error(error.message); // e.g. The authentication token you passed was invalid...
          console.error(error.code);  // e.g. 'invalid_api_key'
          console.error(error.type);  // e.g. 'invalid_request_error'
        } else {
          // Non-API error
          console.log(error);
        }
      }
    }
  
    fetchData();

    }

  },[title,skill,lang])

  
  const [input, setInput]=React.useState('');

  const [output, setOutput]=React.useState('');

  const [translate,setTranslate]=React.useState('');

  const [ques, setQues]=React.useState(`Question 1: What is the output of the following code?

  name = "Mario"
  string_number = "22"
  
  print(name + string_number)
  
  Answer: Mario22`)
  
  const [code, setCode] = React.useState("");
  
  
  const handleSubmit = ()=>{
  
  }
  
  const handleRun = async() => {
    const url = 'https://code-compiler10.p.rapidapi.com/';
    const options = {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-compile': 'rapidapi',
        'Content-Type': 'application/json',
        'X-RapidAPI-Key': 'fdd20bf5b8msh7ae40a7bc92e6eep1156dfjsn9c466cb6de64',
        'X-RapidAPI-Host': 'code-compiler10.p.rapidapi.com'
      },
      body: {
        langEnum: [
          'php',
          'python',
          'c',
          'c_cpp',
          'csharp',
          'kotlin',
          'golang',
          'r',
          'java',
          'typescript',
          'nodejs',
          'ruby',
          'perl',
          'swift',
          'fortran',
          'bash'
        ],
        lang: lang,
        code: code,
        input: input,
      }
    };
    
    try {
      const response = await fetch(url, options);
      const result = await response.text();
      setOutput(result);
      console.log(result);
      
    } catch (error) {
      console.error(error);
    }
  }
  
  useEffect(()=>{
    if(output==resp.output){
      setMsg('Great you have done it, Congrats!')
    }
  },[output])

  const userQuery = async()=>{
    if(output!=resp.output){
      const messages = [
        {role:'system',content: "Only use the function I provide"},
        { role: "user", content: `Whats wrong in this code eg. Print("hello)
        ` },
      ];
      const functions = [
        {
          name: "getcurrect",
          description: `give me the explanation why this doesnt work in this JSON format
          {
            explanation: eg. The code misses the closing (") so it is not working
          }`,
          parameters: {
            type: "object",
            properties: {
              explanation: {
                type: "string",
                description: `explain why the code does not work as expected`,
              },
              
            },
            required: ["explanation"],
          },
        },
      ];

    async function fetchData() {
      try {
        // Your async code here
        const response = await openai.chat.completions.create({
          model: 'gpt-3.5-turbo-0613',
          messages: messages,
          functions: functions,
          function_call: 'auto',
        });
        
        console.log(response)
        const response_message = response.choices[0].message;
        if (!response_message.content) {
          const resp_args = JSON.parse(response_message.function_call.arguments);
          setExplain(resp_args.explanation)
          
        }
      } catch (error) {
        if (error instanceof OpenAI.APIError) {
          console.error(error.status);  // e.g. 401
          console.error(error.message); // e.g. The authentication token you passed was invalid...
          console.error(error.code);  // e.g. 'invalid_api_key'
          console.error(error.type);  // e.g. 'invalid_request_error'
        } else {
          // Non-API error
          console.log(error);
        }
      }
    }
  
    fetchData();
    }
  }


  useEffect(()=>{
    const handleQuery = async (msg) => {
      try {
          console.log('hereiam')
          const url = 'https://deep-translate1.p.rapidapi.com/language/translate/v2';
          const options = {
            method: 'POST',
            headers: {
              'content-type': 'application/json',
              'X-RapidAPI-Key': 'fdd20bf5b8msh7ae40a7bc92e6eep1156dfjsn9c466cb6de64',
              'X-RapidAPI-Host': 'deep-translate1.p.rapidapi.com'
            },
            body: {
              q: explain,
              source: 'en',
              target: 'ta'
            }
          };
  
        const tamils = await fetch(url, options);
        const translationResponse = await tamils.json();
        const tamilTranslation = translationResponse.data.translations[0].translatedText;
        setTranslate(tamilTranslation)
        
      } catch (error) {
        console.log('Request error:', error);
      }
    };
    handleQuery();
  },[explain])


  
  // const handleTypewriterStart = () => {
  //   setEnableTypewriter(true);

  // };



    return (
      <div className="h-screen flex flex-col p-4 bg-black"> {/* Change background color to black */}
        {/* First Row */}
        <div className="flex-row flex gap-2 border border-gray-900 rounded-2xl ">
          {/* Left Column */}
          <div className="bg-black p-4 rounded-lg flex flex-col w-3/4">
            {/* Title */}
            <div className="text-center font-bold text-2xl mb-2 font-manrope text-white"> {/* Change text color to white */}
              AICode
            </div>
            {/* Monaco Editor */}
            <div className=" flex flex-col  rounded-2xl">
              <MonacoEditor
              height={600}
              value={code}
              defaultValue="//write your code here"
              onChange={(e)=>{setCode(e)}}
                theme="vs-dark"
                language="javascript"
                options={{ automaticLayout: true }}
              />
              {/* Input Text Areas */}
              <div className="rounded-lg flex flex-row w-full gap-2 my-2">
                <button 
                onClick={handleSubmit}
                className="bg-white text-black px-4 py-2 rounded"> {/* Change button styles */}
                  Submit
                </button>
                <button 
                onClick={handleRun}
                className="bg-white text-black px-4 py-2 rounded"> {/* Change button styles */}
                  Run Code
                </button>
              </div>
            </div>
          </div>
          {/* Right Column */}
          <div className="flex flex-col space-y-2 flex-1 my-2 mx-2">
            <textarea placeholder='Input' className="border p-2 rounded-lg flex-1" value={input} onChange={(e)=>{setInput(e.target.value)}}></textarea>
            <textarea placeholder='Output' className="border p-2 rounded-lg flex-1" value={output} readOnly={true}></textarea>
          </div>
        </div>
  
        {/* Second Row */}
        <div className="w-full py-4 flex-1">
          <div className="border border-gray-900 p-4 rounded-lg bg-darkgray"> {/* Change background color to dark grey */}
            <div className="font-manrope text-green-300"> {/* Change text color to white */}
              {/* Text */}
              {!enableTypewriter && (<Typewriter
            onInit={(typewriter) => {
              typewriter.typeString(ques)
                .callFunction(() => {
                  console.log('String typed out!');
                })
            }}
              options={{
                deleteSpeed: 1000000000,
                loop:false,
                strings: [ques],
                autoStart: true,
                delay: 50,
              }}
            />)}
              {enableTypewriter && (
            <Typewriter
            onInit={(typewriter) => {
              typewriter.typeString(translate)
                .callFunction(() => {
                  console.log('Explanation typed out!');
                  window.responsiveVoice.speak(translate,'Tamil Female');
                })
            }}
              options={{
                deleteSpeed: 100000,
                loop:false,
                strings: [translate],
                autoStart: true,
                delay: 50,
              }}
            />
          )}
            </div>
            {/* Chat Input */}
            <div className="flex mt-4 ">
              <input
                type="text"
                className="border p-2 rounded-l w-full bg-darkgray text-black placeholder-black" 
                placeholder="Ask Something" 
                value={query}
                onChange={(e)=>{setQuery(e.target.value)}}
              />
              <button 
              onClick={userQuery}
              className="bg-white text-black px-4 rounded-r"> {/* Change button styles */}
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

export default AICode;