import React from 'react';
import { createGlobalStyle } from 'styled-components';
import Chat from './components/Chat';

const GlobalStyle = createGlobalStyle`
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Roboto', 'Segoe UI', 'Arial', sans-serif;
  }
  
  body {
    background-color: #f0f2f5;
  }
`;

function App() {
  return (
    <>
      <GlobalStyle />
      <Chat />
    </>
  );
}

export default App;