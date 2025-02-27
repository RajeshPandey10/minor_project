import React, { useState, useEffect } from 'react';
import { FaBars, FaTimes } from "react-icons/fa";
import { Link, useLocation, useNavigate } from 'react-router-dom';
import AOS from "aos";
import "aos/dist/aos.css";
import AddToChrome from './AddToChrome';

const Navbar = () => {
  const [nav, setNav] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    AOS.init({
      duration: 1000,
      once: true,
    });

    const handleScroll = () => {
      const isScrolled = window.scrollY > 0;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [scrolled]);

  const handleNavClick = (path, section) => {
    if (location.pathname !== '/') {
      navigate(path);
    } else {
      const element = document.getElementById(section);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
    setNav(false);
  };

  const handleHomeClick = (e) => {
    e.preventDefault();
    if (location.pathname !== '/') {
      navigate('/');
    } else {
      window.scrollTo({ top: 0, behavior: 'auto' });
    }
    setNav(false);
  };

  const links = [
    { id: 1, data: "home", path: "/", section: "home" },
    { id: 2, data: "about", path: "/about", section: "about" },
    { id: 3, data: "features", path: "/features", section: "features" },
    { id: 4, data: "how it work", path: "/how-to-use", section: "how-it-works" },
    { id: 5, data: "faq", path: "/faq", section: "faq" },
    { id: 6, data: "contact", path: "/contact", section: "contact" }
  ];

  const isLinkActive = (path, section) => {
    if (location.pathname === '/' && section) {
      const element = document.getElementById(section);
      if (element) {
        const rect = element.getBoundingClientRect();
        return rect.top >= 0 && rect.bottom <= window.innerHeight;
      }
    }
    return location.pathname === path;
  };

  const renderLink = (link, onClick = null) => {
    const isActive = isLinkActive(link.path, link.section);
    const baseClasses = "capitalize transition-colors duration-300";
    const activeClasses = isActive ? "bg-gray-100 text-purple-600 font-bold" : "text-gray-700 hover:text-purple-600";

    return (
      <Link 
        to={link.path}
        className={`${baseClasses} ${activeClasses}`}
        onClick={(e) => {
          e.preventDefault();
          handleNavClick(link.path, link.section);
          onClick && onClick();
        }}
      >
        {link.data}
      </Link>
    );
  };

  useEffect(() => {
    if (nav) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }
  }, [nav]);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      scrolled ? 'bg-yellow-200 shadow-lg' : 'bg-yellow-200'
    }`}>
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between p-4 md:flex-nowrap md:p-6">
        {/* Logo */}
        <Link
          to="/"
          onClick={handleHomeClick}
          className="text-2xl font-extrabold text-gray-800 tracking-wide drop-shadow-md md:text-3xl md:mr-12 cursor-pointer"
          data-aos="fade-right"
        >
          Product Review Analysis
        </Link>

        {/* Mobile Menu Icon */}
        <div
          onClick={() => setNav(!nav)}
          className="cursor-pointer text-gray-800 z-50 md:hidden"
        >
          {nav ? <FaTimes size={24} /> : <FaBars size={24} />}
        </div>

        {/* Desktop Menu */}
        <ul className="hidden md:flex space-x-6">
          {links.map((link, index) => (
            <li
              key={link.id}
              className={`text-lg font-semibold p-2 rounded-lg hover:bg-gray-100 transition-all duration-300 ease-in-out ${
                isLinkActive(link.path, link.section) ? 'bg-gray-100' : ''
              }`}
            >
              {renderLink(link)}
            </li>
          ))}
          <AddToChrome />
        </ul>
      </div>

      {/* Mobile Menu */}
      {nav && (
        <div className="fixed top-0 left-0 w-full h-screen bg-yellow-200 backdrop-blur-sm flex flex-col items-center justify-center z-40">
          <ul className="flex flex-col items-center space-y-8">
            {links.map((link, index) => (
              <li
                key={link.id}
                className={`text-xl font-semibold capitalize transition-all duration-300 ease-in-out ${
                  isLinkActive(link.path, link.section) 
                    ? 'bg-gray-100 text-purple-600' 
                    : 'text-gray-800 hover:text-purple-600'
                }`}
                data-aos="fade-up"
                data-aos-delay={index * 100}
              >
                {renderLink(link, () => setNav(!nav))}
              </li>
            ))}
            <AddToChrome />
          </ul>
        </div>
      )}
    </nav>
  );
};

export default Navbar;