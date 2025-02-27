import React, { useState } from "react";
import { motion } from "framer-motion";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import AddToChrome from "../../components/AddToChrome";
import About from "../about/About";
import Features from "../features/Features";
import HowItWork from "../howItWork/HowItWork";
import Contact from "../message/Contact";
import Faq from "../../components/Faq";
import { Link } from "react-router-dom";

const Home = () => {
  const [showExtensionPrompt, setShowExtensionPrompt] = useState(false);
  const [showModal, setShowModal] = useState(false);

  const handleAnalyzeClick = () => {
    // Check if the extension is installed
    if (!window.chrome || !window.chrome.runtime || !window.chrome.runtime.id) {
      toast.info(
        <div className="flex flex-col space-y-4 py-2">
          <p className="text-gray-800">
            Please install our Chrome extension to analyze reviews.
          </p>
          <div className="flex justify-end space-x-2">
            <button
              onClick={() => toast.dismiss()}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <AddToChrome className="!m-0 list-none" />
          </div>
        </div>,
        {
          position: "top-center",
          autoClose: false,
          hideProgressBar: true,
          closeOnClick: false,
          pauseOnHover: true,
          draggable: false,
          progress: undefined,
          className: "!bg-white !text-gray-800 !rounded-xl !p-6 !shadow-xl",
        }
      );
    } else {
      // Extension is installed, proceed with analysis
      toast.success("Starting analysis...", {
        position: "top-center",
        autoClose: 2000,
      });
      // Add your analysis logic here
    }
  };

  return (
    <section
      className="min-h-screen bg-gradient-to-b from-yellow-50 to-white  overflow-hidden"
      id="home"
    >
      <ToastContainer />

      {/* Main Content */}
      <div className="pt-16">
        {/* Hero Section */}
        <section className="min-h-screen">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-20">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Left Content */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                className="space-y-8"
              >
                <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
                  Make Better Product Decisions with
                  <span className="text-purple-600"> AI-Powered Reviews</span>
                </h1>
                <p className="text-lg md:text-xl text-gray-600 max-w-2xl">
                  Transform your shopping experience with our advanced review
                  analysis tool. Get instant insights from thousands of customer
                  reviews using cutting-edge AI technology.
                </p>
                <div className="flex flex-col sm:flex-row gap-4">
                  <button
                    onClick={handleAnalyzeClick}
                    className="px-8 py-4 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition-colors duration-300 shadow-lg hover:shadow-xl"
                  >
                    Analyze Reviews Now
                  </button>
                  <button
                    onClick={() => setShowModal(true)}
                    className="px-8 py-4 bg-white text-purple-600 rounded-lg font-semibold border-2 border-purple-600 hover:bg-purple-50 transition-colors duration-300"
                  >
                    Watch Demo
                  </button>
                  {/* Modal */}
                  {showModal && (
                    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-[9999]">
                      <div className="bg-white p-6 rounded-lg shadow-lg max-w-lg w-full relative">
                        {/* Close Button */}
                        <button
                          onClick={() => setShowModal(false)}
                          className="absolute top-2 right-2 text-gray-600 hover:text-black text-xl"
                        >
                          &times;
                        </button>

                        {/* YouTube Video */}
                        <div className="relative w-full h-64">
                          <iframe
                            className="w-full h-full rounded-lg"
                            src="https://www.youtube.com/embed/dQw4w9WgXcQ"
                            title="YouTube Video"
                            allowFullScreen
                          ></iframe>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-8 pt-4">
                  <div className="text-center">
                    <h3 className="text-3xl font-bold text-gray-900">100+</h3>
                    <p className="text-gray-600">Reviews Analyzed</p>
                  </div>
                  <div className="text-center">
                    <h3 className="text-3xl font-bold text-gray-900">80%</h3>
                    <p className="text-gray-600">Accuracy Rate</p>
                  </div>
                  <div className="text-center">
                    <h3 className="text-3xl font-bold text-gray-900">50+</h3>
                    <p className="text-gray-600">Happy Users</p>
                  </div>
                </div>
              </motion.div>

              {/* Right Content - Animation */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="relative"
              >
                <div className="relative z-10">
                  <img
                    src="https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2426&q=80"
                    alt="Product Review Analysis Dashboard"
                    className="rounded-2xl shadow-2xl"
                  />
                  {/* Floating Elements */}
                  <div className="absolute -top-4 -right-4 bg-white p-4 rounded-lg shadow-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <span className="text-sm font-semibold">
                        Sentiment Analysis
                      </span>
                    </div>
                  </div>
                  <div className="absolute -bottom-4 -left-4 bg-white p-4 rounded-lg shadow-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                      <span className="text-sm font-semibold">
                        Real-time Updates
                      </span>
                    </div>
                  </div>
                </div>

                {/* Background Decoration */}
                <div className="absolute -z-10 top-1/2 right-1/2 transform translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-purple-200 rounded-full blur-3xl opacity-20"></div>
              </motion.div>

              {/* Background Elements */}
              <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
                <div className="absolute top-0 right-0 w-96 h-96 bg-yellow-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
                <div className="absolute top-0 right-96 w-96 h-96 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
                <div className="absolute -bottom-8 left-20 w-96 h-96 bg-pink-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
              </div>
            </div>
          </div>
        </section>

        {/* About Section */}
        <section id="about" className="min-h-screen">
          <About />
        </section>

        {/* Features Section */}
        <section id="features" className="min-h-screen">
          <Features />
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" className="min-h-screen">
          <HowItWork />
        </section>

        {/* FAQ Section */}
        <section id="faq" className="min-h-screen">
          <Faq />
        </section>

        {/* Contact Section */}
        <section id="contact" className="min-h-screen">
          <Contact />
        </section>
      </div>
    </section>
  );
};

export default Home;
