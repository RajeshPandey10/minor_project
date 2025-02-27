import React from 'react';
import { motion } from 'framer-motion';
import { FaChartBar, FaRobot, FaStar, FaShieldAlt, FaBolt, FaChartLine } from 'react-icons/fa';

const Features = () => {
  const features = [
    {
      icon: <FaChartBar className="w-8 h-8 text-purple-600" />,
      title: "Sentiment Analysis",
      description: "Advanced AI algorithms analyze review sentiments to provide accurate positive, negative, and neutral ratings."
    },
    {
      icon: <FaRobot className="w-8 h-8 text-purple-600" />,
      title: "Natural Language Processing",
      description: "Understand context and nuances in reviews using state-of-the-art NLP technology."
    },
    {
      icon: <FaStar className="w-8 h-8 text-purple-600" />,
      title: "Key Feature Extraction",
      description: "Automatically identify and highlight the most important product features mentioned in reviews."
    },
    {
      icon: <FaShieldAlt className="w-8 h-8 text-purple-600" />,
      title: "Confident Score",
      description: "Used Compound score and provide a confident score to help you understand the true product quality."
    },
    {
      icon: <FaBolt className="w-8 h-8 text-purple-600" />,
      title: "Real-time Analysis",
      description: "Get instant insights as you browse products with our Chrome extension."
    },
    {
      icon: <FaChartLine className="w-8 h-8 text-purple-600" />,
      title: "Recomondation",
      description: "recomonding product based on the review  and confident scorre",
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-yellow-50 to-white py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Powerful Features for
            <span className="text-purple-600"> Smart Shopping</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Our AI-powered review analysis tool comes packed with features to help you make confident purchasing decisions.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mt-12">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow duration-300"
            >
              <div className="bg-purple-50 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">
                {feature.title}
              </h3>
              <p className="text-gray-600">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="text-center mt-20"
        >
          <div className="bg-purple-600 text-white rounded-2xl p-12 shadow-xl">
            <h2 className="text-3xl font-bold mb-6">
              Ready to Make Smarter Shopping Decisions?
            </h2>
            <p className="text-lg mb-8 text-purple-100">
              Install our Chrome extension and start analyzing product reviews instantly.
            </p>
            <button className="bg-white text-purple-600 px-8 py-4 rounded-lg font-semibold hover:bg-purple-50 transition-colors duration-300 shadow-lg">
              Add to Chrome - It's Free
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Features;