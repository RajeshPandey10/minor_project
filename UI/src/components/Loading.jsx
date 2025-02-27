import React from 'react';
import { motion } from 'framer-motion';

const Loading = () => {
  return (
    <div className="fixed inset-0 bg-white bg-opacity-80 z-50 flex items-center justify-center">
      <motion.div
        className="flex flex-col items-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <div className="flex space-x-2">
          <motion.div
            className="w-4 h-4 bg-purple-600 rounded-full"
            animate={{
              y: [0, -10, 0],
              scale: [1, 0.8, 1],
            }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: 0,
            }}
          />
          <motion.div
            className="w-4 h-4 bg-purple-600 rounded-full"
            animate={{
              y: [0, -10, 0],
              scale: [1, 0.8, 1],
            }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: 0.2,
            }}
          />
          <motion.div
            className="w-4 h-4 bg-purple-600 rounded-full"
            animate={{
              y: [0, -10, 0],
              scale: [1, 0.8, 1],
            }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: 0.4,
            }}
          />
        </div>
        <motion.p
          className="mt-4 text-purple-600 font-medium"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Loading...
        </motion.p>
      </motion.div>
    </div>
  );
};

export default Loading;
