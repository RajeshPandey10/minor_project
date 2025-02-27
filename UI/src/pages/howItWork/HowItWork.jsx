import React from 'react';
import { motion } from 'framer-motion';
import { FaChrome, FaSearch, FaRobot, FaCheckCircle } from 'react-icons/fa';

const HowItWork = () => {
  const steps = [
    {
      icon: <FaChrome className="w-12 h-12 text-purple-600" />,
      title: "Install Extension",
      description: "Add our Chrome extension to your browser with just one click. It's free and takes less than a minute.",
      image: "https://images.unsplash.com/photo-1607706189992-eae578626c86?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80"
    },
    {
      icon: <FaSearch className="w-12 h-12 text-purple-600" />,
      title: "Browse Products",
      description: "Shop as usual on your favorite e-commerce sites. Our extension works seamlessly in the background.",
      image: "https://images.unsplash.com/photo-1563013544-824ae1b704d3?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80"
    },
    {
      icon: <FaRobot className="w-12 h-12 text-purple-600" />,
      title: "AI Analysis",
      description: "Click the extension icon to analyze product reviews instantly using our advanced AI algorithms.",
      image: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80"
    },
    {
      icon: <FaCheckCircle className="w-12 h-12 text-purple-600" />,
      title: "Make Decision",
      description: "Get instant insights about product quality, reliability, and potential issues to make informed decisions.",
      image: "https://images.unsplash.com/photo-1434626881859-194d67b2b86f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2074&q=80"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-yellow-50 to-white py-28">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-20"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            How It
            <span className="text-purple-600"> Works</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Get started with our AI-powered review analysis tool in just a few simple steps
          </p>
        </motion.div>

        {/* Steps Section */}
        <div className="space-y-20">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: index * 0.2 }}
              className={`flex flex-col ${
                index % 2 === 0 ? 'lg:flex-row' : 'lg:flex-row-reverse'
              } items-center gap-12`}
            >
              {/* Content */}
              <div className="flex-1 space-y-6">
                <div className="bg-purple-50 w-20 h-20 rounded-2xl flex items-center justify-center mb-6">
                  {step.icon}
                </div>
                <div className="flex items-center gap-4 mb-4">
                  <div className="bg-purple-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">
                    {index + 1}
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900">{step.title}</h3>
                </div>
                <p className="text-lg text-gray-600">{step.description}</p>
              </div>

              {/* Image */}
              <div className="flex-1">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  transition={{ duration: 0.3 }}
                  className="relative"
                >
                  <div className="absolute inset-0 bg-purple-600 rounded-2xl transform rotate-3"></div>
                  <img
                    src={step.image}
                    alt={step.title}
                    className="relative rounded-2xl shadow-xl w-full h-[300px] object-cover"
                  />
                </motion.div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="text-center mt-20"
        >
          <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-2xl p-12 shadow-xl">
            <h2 className="text-3xl font-bold mb-6">
              Ready to Start Making Better Shopping Decisions?
            </h2>
            <p className="text-lg mb-8 text-purple-100">
              Join thousands of smart shoppers who use our AI-powered review analysis.
            </p>
            <button className="bg-white text-purple-600 px-8 py-4 rounded-lg font-semibold hover:bg-purple-50 transition-colors duration-300 shadow-lg">
              Install Extension Now - It's Free
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default HowItWork;