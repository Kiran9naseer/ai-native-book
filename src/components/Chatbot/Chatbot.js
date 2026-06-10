
import React, { useState, useRef, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './Chatbot.module.css';

const MAX_HISTORY = 100; // Limit local history

// Helper to format source links
const SourceLink = ({ source }) => {
    if (!source || !source.path) return null;
    return (
        <a
            href={source.path}
            className={styles.sourceLink}
            target="_blank"
            rel="noopener noreferrer"
        >
            📚 {source.title || 'Source'}
        </a>
    );
};

export default function Chatbot() {
    const { siteConfig } = useDocusaurusContext();
    const apiUrl = siteConfig.customFields?.chatbotApiUrl || 'http://localhost:8000';

    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: 'Hi! I can answer questions about the AI-Native Book. What would you like to know?'
        }
    ]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);
    const [conversationId, setConversationId] = useState(null);

    // Auto-scroll to bottom of messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isOpen]);

    const toggleChat = () => setIsOpen(!isOpen);

    const handleInput = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const sendMessage = async () => {
        if (!inputValue.trim() || isLoading) return;

        const question = inputValue.trim();
        setInputValue('');
        setIsLoading(true);

        // Add user message immediately
        setMessages(prev => [...prev, { role: 'user', content: question }]);

        try {
            const response = await fetch(`${apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    conversation_id: conversationId
                }),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }

            const data = await response.json();

            if (data.conversation_id) {
                setConversationId(data.conversation_id);
            }

            setMessages(prev => [
                ...prev,
                {
                    role: 'assistant',
                    content: data.answer,
                    sources: data.sources
                }
            ]);

        } catch (error) {
            console.error('Chat error:', error);
            setMessages(prev => [
                ...prev,
                {
                    role: 'assistant',
                    content: 'Sorry, the chatbot backend is currently offline. The AI assistant requires a running backend server.',
                    isError: true
                }
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={styles.chatbotContainer}>
            <div className={`${styles.chatWindow} ${isOpen ? styles.open : ''}`}>
                <div className={styles.header}>
                    <span>AI Assistant</span>
                    <button onClick={toggleChat} className={styles.closeButton}>×</button>
                </div>

                <div className={styles.messages}>
                    {messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={`${styles.message} ${msg.role === 'user' ? styles.userMessage : styles.aiMessage}`}
                        >
                            <div>{msg.content}</div>
                            {msg.sources && msg.sources.length > 0 && (
                                <div className={styles.sourceContainer}>
                                    <div className={styles.sourceTitle}>Sources:</div>
                                    {msg.sources.map((src, i) => (
                                        <SourceLink key={i} source={src} />
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                    {isLoading && (
                        <div className={styles.typingIndicator}>
                            <div className={styles.dot}></div>
                            <div className={styles.dot}></div>
                            <div className={styles.dot}></div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div className={styles.inputArea}>
                    <input
                        type="text"
                        className={styles.input}
                        placeholder="Ask a question..."
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleInput}
                        disabled={isLoading}
                    />
                    <button
                        className={styles.sendButton}
                        onClick={sendMessage}
                        disabled={isLoading || !inputValue.trim()}
                    >
                        ➤
                    </button>
                </div>
            </div>

            <button className={styles.chatbotToggle} onClick={toggleChat}>
                {isOpen ? '✕' : '💬'}
            </button>
        </div>
    );
}
