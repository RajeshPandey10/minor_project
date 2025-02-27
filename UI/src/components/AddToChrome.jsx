import React from 'react';

const AddToChrome = ({ className = '', isMobile = false }) => {
  return (
    <li
      data-aos={isMobile ? "fade-up" : "fade-left"}
      className={`${className} ${isMobile ? 'mt-6' : 'mb-0'}`}
    >
      <a 
        href="/Users/rajeshpandey/Documents/PAFR/popup.html"
        className={`inline-flex items-center ${isMobile ? 'px-6 py-3' : 'px-4 py-2'} bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg font-semibold shadow-md hover:from-purple-700 hover:to-indigo-700 transition-all duration-300 ease-in-out`}
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg className={`${isMobile ? 'w-5 h-5' : 'w-4 h-4'} mr-2`} fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M16.214 8.69l6.715-1.679A12.027 12.027 0 0 1 24 12c0 1.191-.171 2.346-.486 3.43l-7.3-6.74zm-1.214 2.338L21.17 16.3A12.04 12.04 0 0 1 12 24c-1.191 0-2.346-.171-3.43-.486l6.74-7.3zm-2.359-1.197L6.3 2.83A12.04 12.04 0 0 1 12 0c1.191 0 2.346.171 3.43.486l-2.789 9.345zM7.31 16.214l-6.74 7.3A12.027 12.027 0 0 1 0 12c0-1.191.171-2.346.486-3.43l6.824 7.644z"/>
        </svg>
        Add to Chrome
      </a>
    </li>
  );
};

export default AddToChrome;