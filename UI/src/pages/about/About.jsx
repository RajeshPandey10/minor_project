import React from 'react';
import { motion } from 'framer-motion';
import { FaGithub, FaLinkedin, FaUser } from 'react-icons/fa';

const About = () => {
  const technologies = [
    {
      name: "Machine Learning",
      description: "Sentiment analysis and natural language processing for review interpretation",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      )
    },
    {
      name: "React.js",
      description: "Modern and responsive front-end user interface",
      icon: (
        <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 13.5a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3Z"/>
          <path d="M12 22.5c-1.9 0-3.6-.3-4.9-.9-1.5-.7-2.6-1.7-3.2-2.9-.6-1.2-.7-2.5-.3-3.7.4-1.2 1.3-2.3 2.5-3.2-.3-.5-.5-1-.7-1.5-.2-.7-.3-1.3-.2-1.9.1-.6.3-1.1.7-1.6.4-.5.9-.8 1.5-1.1.6-.3 1.3-.4 2-.3.7.1 1.4.3 2.1.7.4-.7.8-1.3 1.3-1.7.6-.5 1.2-.9 1.9-1.1.7-.2 1.3-.2 2 0 .6.2 1.2.5 1.7 1s.9 1.1 1.1 1.7c.2.7.2 1.4 0 2-.2.7-.5 1.3-1 1.9.7.4 1.3.8 1.7 1.3.5.6.9 1.2 1.1 1.9.2.7.2 1.3 0 2-.2.6-.5 1.2-1 1.7s-1.1.9-1.7 1.1c-.7.2-1.4.2-2 0-.7-.2-1.3-.5-1.9-1-.4.7-.8 1.3-1.3 1.7-.6.5-1.2.9-1.9 1.1-.7.2-1.3.2-2 0-.6-.2-1.2-.5-1.7-1s-.9-1.1-1.1-1.7Z"/>
        </svg>
      )
    },
    {
      name: "Chrome Extension",
      description: "Browser extension for seamless review analysis",
      icon: (
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M16.214 8.69l6.715-1.679A12.027 12.027 0 0 1 24 12c0 1.191-.171 2.346-.486 3.43l-7.3-6.74zm-1.214 2.338L21.17 16.3A12.04 12.04 0 0 1 12 24c-1.191 0-2.346-.171-3.43-.486l6.74-7.3zm-2.359-1.197L6.3 2.83A12.04 12.04 0 0 1 12 0c1.191 0 2.346.171 3.43.486l-2.789 9.345zM7.31 16.214l-6.74 7.3A12.027 12.027 0 0 1 0 12c0-1.191.171-2.346.486-3.43l6.824 7.644z"/>
        </svg>
      )
    }
  ];

  const teamMembers = [
    {
      name: 'Rajesh Pandey',
      role: 'Computer Engineer',
      github: 'https://github.com/Rajeshp2005',
      linkedin: 'https://www.linkedin.com/in/rajesh-pandey-915649286/',
      image: 'https://www.rajeshpandey10.com.np/assets/img/mee.png'
    },
   
    {
      name: 'Ravi Pandit',
      role: 'Computer Engineer',
      github: 'https://github.com/ravix2001',
      linkedin: 'https://www.linkedin.com/in/ravipandit2001/',
      image: 'https://www.ravi-pandit.com.np/image.png' // Add image URL when available
    },
    {
      name: 'Nigam Yadav',
      role: 'Computer Engineer',
      github: 'https://github.com/nigamyadav72',
      linkedin: 'https://www.linkedin.com/in/nigam-yadav-099447199/',
      image: 'https://www.nigamyadav.com.np/image/new%20pp.jpeg' // Add image URL when available
    },
    {
      name: 'Sonu Gupta',
      role: 'Computer Engineer',
      github: 'https://github.com/sonu-gupta',//https://github.com/creationsbysonu
      linkedin: 'https://www.linkedin.com/in/sonu-gupta-681329340/',
      image: 'https://www.guptasonu.com.np/assets/photo-BiFIU-sz.jpg' // Add image URL when available
    },
  ];

  return (
    <section className="py-20 lg:py-28 bg-gradient-to-b from-purple-50 to-white" id="about">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Project Introduction */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            About Our Project
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            A mini project developed by Computer Engineering students to revolutionize how people make purchasing decisions through AI-powered review analysis.
          </p>
        </motion.div>

        {/* Project Overview */}
        <div  className="grid md:grid-cols-2 gap-12 items-center mb-20">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <h3  className="text-2xl font-bold text-gray-900">
              Project Overview
            </h3>
            <p className="text-gray-600">
              Our Product Review Analysis tool is designed to help consumers make informed decisions by analyzing product reviews using advanced machine learning algorithms. The project combines natural language processing with sentiment analysis to provide meaningful insights from customer reviews.
            </p>
            <div className="space-y-4">
              <h4 className="font-semibold text-gray-900">Key Features:</h4>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-center">
                  <svg className="w-5 h-5 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Sentiment Analysis of Reviews
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Key Feature Extraction
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Real-time Analysis
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Browser Extension Integration
                </li>
              </ul>
            </div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="relative"
          >
            <img
              src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2340&q=80"
              alt="Project Overview"
              className="rounded-2xl shadow-2xl"
            />
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-transparent rounded-2xl"></div>
          </motion.div>
        </div>

        {/* Technologies Used */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <h3 className="text-2xl font-bold text-gray-900 text-center mb-12">
            Technologies Used
          </h3>
          <div className="grid md:grid-cols-3 gap-8">
            {technologies.map((tech, index) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300"
              >
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center text-purple-600 mb-4">
                  {tech.icon}
                </div>
                <h4 className="text-xl font-semibold text-gray-900 mb-2">{tech.name}</h4>
                <p className="text-gray-600">{tech.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Team Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <h3 className="text-2xl font-bold text-gray-900 mb-4">
            Meet Our Team
          </h3>
          <p className="text-gray-600 max-w-2xl mx-auto mb-12">
            A group of passionate computer engineering students working together to make online shopping smarter and more informed through technology.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="flex flex-col items-center group"
              >
                {/* Circular Image Container */}
                <div className="relative mb-6 transform transition-all duration-300 group-hover:scale-110">
                  <div className="w-48 h-48 rounded-full bg-gradient-to-br from-purple-400 to-purple-600 p-1 transition-all duration-300 ">
                    <div className="w-full h-full rounded-full bg-white flex items-center justify-center overflow-hidden transition-all duration-300 group-hover:bg-green-50">
                      {member.image ? (
                        <img
                          src={member.image}
                          alt={member.name}
                          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(
                              member.name
                            )}&background=9333ea&color=fff&size=200`;
                          }}
                        />
                      ) : (
                        <div className="w-full h-full rounded-full bg-purple-100 flex items-center justify-center group-hover:bg-green-50">
                          <FaUser className="w-24 h-24 text-purple-400 transition-all duration-300 group-hover:text-green-400" />
                        </div>
                      )}
                    </div>
                  </div>
                  {/* Hover Effect Ring */}
                  <div className="absolute -inset-2 rounded-full bg-black opacity-1 scale-0 transform transition-all duration-300 group-hover:opacity-25 group-hover:scale-90" />
                </div>

                {/* Member Info */}
                <div className="text-center">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-2 transition-colors duration-300 group-hover:text-green-600">
                    {member.name}
                  </h3>
                  <p className="text-purple-600 mb-2 transition-colors duration-300 group-hover:text-green-500">
                    {member.role}
                  </p>
                  <p className="text-gray-600 text-sm mb-4">
                    IOE Purwanchal Campus, Dharan
                  </p>
                  
                  {/* Social Links */}
                  <div className="flex justify-center space-x-4">
                    <motion.a
                      whileHover={{ scale: 1.2 }}
                      href={member.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-600 hover:text-green-600 transition-colors duration-300"
                    >
                      <FaGithub className="w-6 h-6" />
                    </motion.a>
                    <motion.a
                      whileHover={{ scale: 1.2 }}
                      href={member.linkedin}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-600 hover:text-green-600 transition-colors duration-300"
                    >
                      <FaLinkedin className="w-6 h-6" />
                    </motion.a>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default About;