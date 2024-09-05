import React from 'react'
import { useState, useEffect } from 'react';



export default function AiQuiz() {

    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [selectedOption, setSelectedOption] = useState('');
    const [score, setScore] = useState(0);
  
    const handleOptionClick = (option) => {
      if (selectedOption) {
        return;
      }
  
      setSelectedOption(option);
      const currentQuestion = quizData.PythonMcq[currentQuestionIndex];
      if (currentQuestion.correct_option === option) {
        setScore((prevScore) => prevScore + 1);
      }
  
      setTimeout(() => {
        if (currentQuestionIndex === quizData.PythonMcq.length - 1) {
          console.log('Final Score:', score);
          // You can display the score or navigate to another page here
        } else {
          setSelectedOption('');
          setCurrentQuestionIndex((prevIndex) => prevIndex + 1);
        }
      }, 500);
    };

  // Your quiz data, you can replace this with your actual quiz data
  const quizData = {
    "PythonMcq": [
        {
            "question": "Which of the following is NOT a valid identifier in most programming languages?",
            "options": {
                "a": "my_variable",
                "b": "_variable",
                "c": "123_variable",
                "d": "variable123"
            },
            "correct_option": "c"
        },
        {
            "question": "Which of the following is used to add a single-line comment in Python?",
            "options": {
                "a": "/* This is a comment */",
                "b": "// This is a comment",
                "c": "# This is a comment",
                "d": "<!-- This is a comment -->"
            },
            "correct_option": "c"
        },
        {
            "question": "What is the correct way to declare and initialize an empty list in Python?",
            "options": {
                "a": "list_name = []",
                "b": "list_name = {}",
                "c": "list_name = ()",
                "d": "list_name = \"\""
            },
            "correct_option": "a"
        },
        {
            "question": "Which loop is guaranteed to execute at least once in Python?",
            "options": {
                "a": "for loop",
                "b": "while loop",
                "c": "do-while loop",
                "d": "Python does not have a loop that executes at least once."
            },
            "correct_option": "c"
        },
        {
            "question": "What is the output of the following code snippet?",
            "options": {
                "a": "[2, 4, 6, 8, 10]",
                "b": "[4, 8]",
                "c": "[1, 2, 3, 4, 5]",
                "d": "[1, 4, 3, 8, 5]"
            },
            "correct_option": "a"
        },
        {
            "question": "In Python, what is the purpose of the import keyword?",
            "options": {
                "a": "It is used to import external libraries or modules into your code.",
                "b": "It is used to declare variables in Python.",
                "c": "It is used to include comments in the code.",
                "d": "It is used to execute a system command."
            },
            "correct_option": "a"
        },
        {
            "question": "What will be the output of the following code snippet?",
            "options": {
                "a": "[2, 3, 4]",
                "b": "[1, 2, 3]",
                "c": "[2, 3]",
                "d": "[1, 2, 3, 4]"
            },
            "correct_option": "a"
        },
        {
            "question": "What is the purpose of the super() function in Python?",
            "options": {
                "a": "It is used to access attributes of the parent class in a derived class.",
                "b": "It is used to terminate the execution of a loop prematurely.",
                "c": "It is used to pause the execution of a program for a specified duration.",
                "d": "It is used to create a new instance of a class."
            },
            "correct_option": "a"
        },
        {
            "question": "Which of the following is an example of a built-in Python decorator?",
            "options": {
                "a": "@staticmethod",
                "b": "@serialize",
                "c": "@memoize",
                "d": "@increment"
            },
            "correct_option": "a"
        },
        {
            "question": "Which of the following is the correct syntax to open a file named \"data.txt\" in read mode in Python?",
            "options": {
                "a": "file = open(\"data.txt\", \"r\")",
                "b": "file = open(\"data.txt\", mode=\"read\")",
                "c": "file = read(\"data.txt\")",
                "d": "file = read_file(\"data.txt\", mode=\"r\")"
            },
            "correct_option": "a"
        }
    ]
};

const currentQuestion = quizData.PythonMcq[currentQuestionIndex];
const options = Object.entries(currentQuestion.options);

return (
    <div className="flex  items-center justify-center h-screen bg-black text-white font-manrope">
      <div className="bg-white rounded-xl w-full mx-40 ">
        <div className="bg-black flex flex-row p-5 items-center justify-center">
          <div className="relative leading-[32px] whitespace-pre-wrap flex items-center p-2 w-full">
            <span className="w-full">
              <p className="mb-3">
                <b>
                  <span>Question</span>
                  <span className="text-2xl">{currentQuestionIndex + 1}</span>
                  <span className="text-lg">{`/${quizData.PythonMcq.length}  `}</span>
                </b>
              </p>
              <p className="m-0">{currentQuestion.question}</p>
            </span>
          </div>
        </div>
        <div className="flex flex-col space-y-5 text-black p-2">
          {options.map(([optionKey, optionValue]) => (
            <button
              key={optionKey}
              onClick={() => handleOptionClick(optionKey)}
              className={`flex p-3 items-center justify-start border rounded-lg ${
                selectedOption === optionKey
                  ? optionKey === currentQuestion.correct_option
                    ? 'bg-green-500 text-white'
                    : 'bg-red-500 text-white'
                  : 'bg-darkorange-300'
              }`}
            >
              <div className="w-3 h-3 rounded-full border border-black mr-3" />
              <div className="flex flex-row items-start justify-start">
                <div className="leading-[32px]">{optionValue}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
