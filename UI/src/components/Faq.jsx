import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaChevronDown } from 'react-icons/fa';
import { HashLink as Link } from 'react-router-hash-link';
const Faq = () => {
  const [activeIndex, setActiveIndex] = useState(null);

  const faqs = [
    {
      question: 'What is Product Review AI?',
      answer: 'Product Review AI is an intelligent tool that analyzes product reviews using artificial intelligence to help you make informed purchasing decisions. It identifies key features, sentiment, and authenticity of reviews.'
    },
    {
      question: 'How does the review analysis work?',
      answer: 'Our AI system processes reviews using advanced natural language processing techniques. It analyzes sentiment, identifies key product features, confident score, and provides a comprehensive summary to help you understand the true product quality.'
    },
    {
      question: 'Is the browser extension free to use?',
      answer: 'Yes, our browser extension is completely free to use. We believe in making online shopping safer and more informed for everyone.'
    },
    {
      question: 'Which e-commerce platforms are supported?',
      answer: 'Currently, we support Daraz. We are continuously working to add support for more platforms.'
    },
   
    {
      question: 'How can I contact support?',
      answer: 'You can reach our support team through the contact form on our website or by emailing imrajesh2005@gmail.com We typically respond within 24 hours.'
    }
  ];

  const toggleQuestion = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-yellow-50 to-white py-28">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Frequently Asked <span className="text-purple-600">Questions</span>
          </h1>
          <p className="text-lg text-gray-600">
            Find answers to common questions about our Product Review Analysis tool.
          </p>
        </motion.div>

        {/* FAQ List */}
        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white rounded-lg shadow-md overflow-hidden"
            >
              <button
                onClick={() => toggleQuestion(index)}
                className="w-full px-6 py-4 text-left flex items-center justify-between hover:bg-gray-50 transition-colors duration-200"
              >
                <span className="text-lg font-semibold text-gray-900">
                  {faq.question}
                </span>
                <FaChevronDown
                  className={`w-5 h-5 text-gray-500 transition-transform duration-200 ${
                    activeIndex === index ? 'transform rotate-180' : ''
                  }`}
                />
              </button>
              
              <AnimatePresence>
                {activeIndex === index && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-6 py-4 bg-gray-50 text-gray-600">
                      {faq.answer}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>

       
      </div>
    </div>
  );
};

export default Faq;