import { useState,useEffect } from 'react';
import { useNavigate,useLocation} from 'react-router-dom';
import {
  collection,
  doc,
  getDoc,
  updateDoc,
  setDoc,
} from "firebase/firestore";
import { auth, db } from '../firebaseconfig';
const Quiz = () => {
  const location = useLocation();
  const userEmail = auth.currentUser.email;
  const [uEmail, setUserEmail]=useState('');
  useEffect(() => {
    // Here you should have a mechanism to set the userEmail state
    // after you have authenticated the user. For example, you can
    // pass it as a prop to this component or fetch it from the context.
    // For demonstration purposes, I'll assume it's set through a prop.
    setUserEmail(userEmail);
  }, []);
  const scoreCollection = collection(db, 'score');
  const navigate=useNavigate()
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

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState('');
  const [score, setScore] = useState(0);

  const handleOptionClick = (option) => {
    // If the user already selected an answer, do nothing
    if (selectedOption) {
      return;
    }
  
    // Set the selected option and check if it's correct
    setSelectedOption(option);
    const currentQuestion = quizData.PythonMcq[currentQuestionIndex];
    if (currentQuestion.correct_option === option) {
      setScore((prevScore) => prevScore + 1);
    }
  
    // Wait for a short time to show the user's selected answer
    setTimeout(async () => {
      if (currentQuestionIndex === quizData.PythonMcq.length - 1) {
        // Calculate the final score after the quiz is done
        let finalScore = score;
  
        // Get the user's email
        try {
          // Check if the user's score document exists in the score collection
          const scoreDocRef = doc(db, "scores", userEmail);
          const scoreDocSnap = await getDoc(scoreDocRef);
  
          if (scoreDocSnap.exists()) {
            // User's email exists, update the score in the existing document
            const existingScore = scoreDocSnap.data().score || 0;
            finalScore += existingScore;
            await updateDoc(scoreDocRef, { score: finalScore });
          } 
          else {
            // User's email doesn't exist, create a new document with the email as ID and the score field inside it
            await setDoc(scoreDocRef, { score: finalScore, email:userEmail });
          }
  
          // Navigate to the example route after handling the score
          navigate("/example");
        } catch (error) {
          console.error("Error updating user score:", error);
          // Handle the error and navigate to the example route
          navigate("/example");
        }
      } else {
        setSelectedOption(""); // Reset selectedOption
        setCurrentQuestionIndex((prevIndex) => prevIndex + 1); // Go to the next question
      }
    }, 500);
  };
  

  const currentQuestion = quizData.PythonMcq[currentQuestionIndex];
  const options = Object.entries(currentQuestion.options);

  return (
    <div className="relative [background:linear-gradient(110.8deg,_#ffc989,_#fffcb9_48.96%,_rgba(255,_255,_255,_0))] w-full overflow-hidden flex flex-col py-[124px] px-36 box-border items-start justify-start gap-[10px] text-center text-41xl text-black font-roboto">
      <div className="self-stretch relative leading-[75.5px] capitalize font-medium">
        Take a Skill Test
      </div>
      <div className="self-stretch rounded-21xl bg-gray-700 [backdrop-filter:blur(30px)] overflow-hidden flex flex-col p-[130px] items-start justify-start gap-[20px] text-left text-13xl font-caption-heavy border-[1px] border-solid border-text-white">
        <div className="self-stretch relative">
          <span>Question {currentQuestionIndex + 1}/</span>
          <span className="text-[24px]">{quizData.PythonMcq.length}</span>
        </div>
        <div className="self-stretch h-[328px] flex flex-col items-start justify-start gap-[40px] text-[25px]">
          <div className="self-stretch flex flex-col items-center justify-center">
            <div className="self-stretch relative">
              {currentQuestion.question}
            </div>
          </div>
          <div className="self-stretch flex flex-col py-[26px] px-0 items-center justify-center gap-[20px]">
            {options.map(([optionKey, optionValue]) => (
              <button
                key={optionKey}
                onClick={() => handleOptionClick(optionKey)}
                className={`cursor-pointer [border:none] py-5 px-[52px] ${
                  selectedOption === optionKey
                    ? optionKey === currentQuestion.correct_option
                      ? 'bg-green-500'
                      : 'bg-red-500'
                    : 'bg-darkorange-300'
                } self-stretch rounded-2xl flex flex-row items-center justify-center`}
              >
                <div className="flex flex-row items-end justify-start gap-[6px]">
                  <div className={`relative text-xl leading-[16px] font-semibold font-caption-heavy ${
                      selectedOption === optionKey ? 'text-text-white' : 'text-black'
                  } text-left`}>
                    {optionValue}
                  </div>
                  {selectedOption === optionKey && (
                    <img
                      className="relative w-3.5 h-3.5"
                      alt=""
                      src={
                        optionKey === currentQuestion.correct_option
                          ? "/checkmark-circle.svg"
                          : "/x-circle.svg"
                      }
                    />
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Quiz;
