
import React from 'react';
import Chatbot from '../components/Chatbot/Chatbot';

// Default implementation, that you can customize
export default function Root({ children }) {
    return (
        <>
            {children}
            <Chatbot />
        </>
    );
}
