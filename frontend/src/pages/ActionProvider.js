import React from 'react';
import axios from 'axios';

const ActionProvider = ({ createChatBotMessage, setState, children }) => {
  const handleQuery = async (msg) => {
    try {
      const response = await axios.get(`http://127.0.0.1:8000/ai?q=${msg}`);
      
      const text = await response.data.res.response;
        console.log("response",text)
      const url = 'https://google-translate1.p.rapidapi.com/language/translate/v2';
      const options = {
        method: 'POST',
        headers: {
          'content-type': 'application/x-www-form-urlencoded',
          'Accept-Encoding': 'application/gzip',
          'X-RapidAPI-Key': 'fdd20bf5b8msh7ae40a7bc92e6eep1156dfjsn9c466cb6de64',
          'X-RapidAPI-Host': 'google-translate1.p.rapidapi.com'
        },
        body: new URLSearchParams({
          q: text, // Use the text to be translated
          target: 'ta', // Specify the target language as 'ta' for Tamil
          source: 'en' // Specify the source language as 'en' for English
        }).toString()
      };

      const tamils = await fetch(url, options);
      const translationResponse = await tamils.json();
      const tamilTranslation = translationResponse.data.translations[0].translatedText;

      const botMessage = createChatBotMessage(tamilTranslation);
      
      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, botMessage],
      }));
    } catch (error) {
      console.log('Request error:', error);
    }
  };

  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          actions: { handleQuery },
        });
      })}
    </div>
  );
};

export default ActionProvider;