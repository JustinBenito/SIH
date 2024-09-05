import { createChatBotMessage } from 'react-chatbot-kit';

const config = {
  initialMessages: [createChatBotMessage(`Hi This is Mimi, Your personal AI Guru`)],
  customStyles: {
    botMessageBox: {
      backgroundColor: '#000000',
    },
    chatButton: {
      backgroundColor: 'black',
    },
  },
};

export default config;