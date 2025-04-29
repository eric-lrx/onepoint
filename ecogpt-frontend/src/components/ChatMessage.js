import React from 'react';
import styled from 'styled-components';
import { FaRobot, FaUser } from 'react-icons/fa';

const MessageContainer = styled.div`
  display: flex;
  margin-bottom: 16px;
`;

const Avatar = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;
  background-color: ${props => props.isUser ? '#3e9f85' : '#2c4a73'};
  color: white;
`;

const MessageContent = styled.div`
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 18px;
  background-color: ${props => props.isUser ? '#e6f7f2' : '#f0f2f5'};
  color: #333;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
`;

const ChatMessage = ({ message, isUser }) => {
  return (
    <MessageContainer style={{ justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
      {!isUser && (
        <Avatar isUser={isUser}>
          <FaRobot size={20} />
        </Avatar>
      )}
      <MessageContent isUser={isUser}>
        {message}
      </MessageContent>
      {isUser && (
        <Avatar isUser={isUser}>
          <FaUser size={20} />
        </Avatar>
      )}
    </MessageContainer>
  );
};

export default ChatMessage;