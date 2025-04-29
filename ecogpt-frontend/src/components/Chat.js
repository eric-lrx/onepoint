import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { FaPaperPlane, FaLeaf } from 'react-icons/fa';
import ChatMessage from './ChatMessage';
import apiService from '../services/apiService';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 800px;
  margin: 0 auto;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
`;

const Header = styled.div`
  background-color: #3e9f85;
  color: white;
  padding: 16px;
  display: flex;
  align-items: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h1`
  margin: 0;
  margin-left: 10px;
  font-size: 1.5rem;
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  background-color: #f9f9f9;
`;

const InputContainer = styled.div`
  display: flex;
  padding: 16px;
  background-color: white;
  border-top: 1px solid #eee;
`;

const Input = styled.input`
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #ddd;
  border-radius: 24px;
  font-size: 16px;
  &:focus {
    outline: none;
    border-color: #3e9f85;
  }
`;

const SendButton = styled.button`
  background-color: #3e9f85;
  color: white;
  border: none;
  border-radius: 50%;
  width: 48px;
  height: 48px;
  margin-left: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #328a75;
  }
  
  &:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }
`;

const WelcomeMessage = styled.div`
  text-align: center;
  margin: 32px 0;
  color: #666;
`;

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Add a welcome message when the component mounts
  useEffect(() => {
    setMessages([
      { text: "Bonjour ! Je suis EcoGPT, votre assistant IA. Comment puis-je vous aider aujourd'hui ?", isUser: false }
    ]);
  }, []);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;
    
    const userMessage = inputValue;
    setInputValue('');
    
    // Add user message to chat
    setMessages(prev => [...prev, { text: userMessage, isUser: true }]);
    
    // Show loading indicator
    setIsLoading(true);
    
    try {
      // Send message to API
      const response = await apiService.askQuestion(userMessage);
      
      // Add AI response to chat
      setMessages(prev => [...prev, { text: response.response || "Désolé, je n'ai pas pu traiter votre demande.", isUser: false }]);
    } catch (error) {
      console.error('Error getting response:', error);
      setMessages(prev => [...prev, { 
        text: "Désolé, une erreur s'est produite. Veuillez réessayer plus tard.", 
        isUser: false 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <ChatContainer>
      <Header>
        <FaLeaf size={24} />
        <Title>EcoGPT</Title>
      </Header>
      
      <MessagesContainer>
        {messages.length === 0 && (
          <WelcomeMessage>
            <p>Posez une question à EcoGPT</p>
          </WelcomeMessage>
        )}
        
        {messages.map((message, index) => (
          <ChatMessage 
            key={index} 
            message={message.text} 
            isUser={message.isUser} 
          />
        ))}
        
        {isLoading && (
          <ChatMessage 
            message="En train de réfléchir..." 
            isUser={false} 
          />
        )}
        
        <div ref={messagesEndRef} />
      </MessagesContainer>
      
      <InputContainer>
        <Input
          type="text"
          placeholder="Tapez votre message..."
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <SendButton 
          onClick={handleSendMessage} 
          disabled={inputValue.trim() === '' || isLoading}
        >
          <FaPaperPlane size={20} />
        </SendButton>
      </InputContainer>
    </ChatContainer>
  );
};

export default Chat;