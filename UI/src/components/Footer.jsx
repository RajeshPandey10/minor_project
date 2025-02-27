import React from 'react';
import { FaGithub, FaLinkedin, FaHeart } from 'react-icons/fa';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const teamMembers = [
    {
      name: 'Rajesh Pandey',
      github: 'https://github.com/rajeshpandey179',
      linkedin: 'https://linkedin.com/in/rajesh-pandey'
    },
    {
      name: 'Sonu Gupta',
      github: 'https://github.com/creationsbysonu',
      linkedin: 'https://linkedin.com/in/sonu-gupta'
    },
    {
      name: 'Ravi Pandit',
      github: 'https://github.com/ravix2001',
      linkedin: 'https://www.linkedin.com/in/ravipandit2001/'
    },
    {
      name: 'Nigam Yadav',
      github: 'https://github.com/nigamyadav72',
      linkedin: 'https://www.linkedin.com/in/nigam-yadav-099447199/'
    }
  ];

  return (
    <footer className="bg-gray-900 text-gray-300 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center space-y-6">
          {/* Team Section */}
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">Our Team</h3>
            <div className="flex flex-wrap justify-center gap-6">
              {teamMembers.map((member, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <span className="text-purple-400">{member.name}</span>
                  <a
                    href={member.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-400 hover:text-purple-500 transition-colors duration-300"
                  >
                    <FaGithub className="w-5 h-5" />
                  </a>
                  <a
                    href={member.linkedin}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-400 hover:text-purple-500 transition-colors duration-300"
                  >
                    <FaLinkedin className="w-5 h-5" />
                  </a>
                </div>
              ))}
            </div>
          </div>

          {/* Special Thanks */}
          <div className="pt-4 border-t border-gray-800">
            <p className="text-sm text-gray-400">
              Special thanks to{' '}
              <a
                href="https://pcampus.edu.np/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300"
              >
                IOE Purwanchal Campus, Dharan
              </a>
            </p>
          </div>

          {/* Copyright */}
          <div className="pt-4 border-t border-gray-800 flex items-center justify-center space-x-2 text-sm">
            <span> {currentYear} Product Review AI.</span>
            <span>Made with</span>
            <FaHeart className="text-red-500 w-4 h-4" />
            <span>by Rajesh Pandey</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
