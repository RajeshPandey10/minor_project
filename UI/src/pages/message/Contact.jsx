import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";
import {
  FaPaperPlane,
  FaQuestionCircle,
  FaPhone,
  FaEnvelope,
} from "react-icons/fa";
import { ToastContainer, toast } from "react-toastify";
import axios from "axios"; // <-- Added axios import
import "react-toastify/dist/ReactToastify.css";
import Loading from "../../components/Loading";

const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState({});

  const contactInfo = {
    email: "imrajesh2005@gmail.com",
    phone: "+9779841208969",
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.name.trim()) {
      newErrors.name = "Name is required";
    }
    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = "Please enter a valid email";
    }
    if (!formData.message.trim()) {
      newErrors.message = "Message is required";
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    if (errors[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: "",
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsSubmitting(true);

    try {
      // Removed erroneous console.log(data)
      const res = await axios.post(
        "http://localhost:3000/api/contact",
        formData
      );
      if (res.data.success) {
        toast.success(
          "Message sent successfully! We will get back to you soon.",
          {
            position: "top-right",
            autoClose: 5000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
            progress: undefined,
            theme: "colored",
          }
        );
        setFormData({ name: "", email: "", message: "" });
        setErrors({});
      } else {
        toast.error("Something went wrong. Please try again later.", {
          position: "top-right",
          autoClose: 5000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true,
          progress: undefined,
          theme: "colored",
        });
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      toast.error("Something went wrong. Please try again later.", {
        position: "top-right",
        autoClose: 5000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "colored",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const inputClasses = (fieldName) => `
    w-full px-4 py-3 rounded-lg border 
    ${
      errors[fieldName]
        ? "border-red-300 focus:ring-red-500"
        : "border-gray-300 focus:ring-purple-600"
    } 
    focus:border-transparent transition-colors duration-200
  `;

  return (
    <>
      <AnimatePresence>{isSubmitting && <Loading />}</AnimatePresence>

      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="colored"
      />

      <div className="min-h-screen bg-gradient-to-b from-yellow-50 to-white py-28">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-4 sm:mb-6">
              Get in <span className="text-purple-600">Touch</span>
            </h1>
            <p className="text-base sm:text-lg text-gray-600 mb-6 sm:mb-8 px-4">
              Have questions or feedback? We'd love to hear from you.
            </p>

            {/* Quick Contact Options */}
            <div className="flex flex-col sm:flex-row justify-center items-center gap-4 sm:gap-6 mb-6 sm:mb-8">
              <motion.a
                href={`tel:${contactInfo.phone}`}
                className="flex items-center space-x-2 text-purple-600 hover:text-green-600 transition-colors duration-300 group w-full sm:w-auto justify-center"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="bg-purple-100 p-2 sm:p-3 rounded-full group-hover:bg-green-100 transition-colors duration-300">
                  <FaPhone className="w-4 h-4 sm:w-5 sm:h-5" />
                </div>
                <span className="font-medium text-sm sm:text-base">
                  {contactInfo.phone}
                </span>
              </motion.a>

              <motion.a
                href={`mailto:${contactInfo.email}`}
                className="flex items-center space-x-2 text-purple-600 hover:text-green-600 transition-colors duration-300 group w-full sm:w-auto justify-center"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="bg-purple-100 p-2 sm:p-3 rounded-full group-hover:bg-green-100 transition-colors duration-300">
                  <FaEnvelope className="w-4 h-4 sm:w-5 sm:h-5" />
                </div>
                <span className="font-medium text-sm sm:text-base break-all sm:break-normal">
                  {contactInfo.email}
                </span>
              </motion.a>
            </div>
          </motion.div>

          {/* Contact Form */}
          <motion.form
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            onSubmit={handleSubmit}
            className="bg-white rounded-xl shadow-lg p-4 sm:p-6 md:p-8 mx-4 sm:mx-auto max-w-2xl"
          >
            <div className="space-y-4 sm:space-y-6">
              <div>
                <label
                  htmlFor="name"
                  className="block text-gray-700 font-medium mb-1 sm:mb-2 text-sm sm:text-base"
                >
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className={`${inputClasses("name")} text-sm sm:text-base`}
                  placeholder="Your name"
                />
                {errors.name && (
                  <p className="mt-1 text-xs sm:text-sm text-red-600">
                    {errors.name}
                  </p>
                )}
              </div>

              <div>
                <label
                  htmlFor="email"
                  className="block text-gray-700 font-medium mb-1 sm:mb-2 text-sm sm:text-base"
                >
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className={`${inputClasses("email")} text-sm sm:text-base`}
                  placeholder="your@email.com"
                />
                {errors.email && (
                  <p className="mt-1 text-xs sm:text-sm text-red-600">
                    {errors.email}
                  </p>
                )}
              </div>

              <div>
                <label
                  htmlFor="message"
                  className="block text-gray-700 font-medium mb-1 sm:mb-2 text-sm sm:text-base"
                >
                  Message
                </label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  required
                  rows="4"
                  className={`${inputClasses(
                    "message"
                  )} resize-none text-sm sm:text-base`}
                  placeholder="Your message..."
                />
                {errors.message && (
                  <p className="mt-1 text-xs sm:text-sm text-red-600">
                    {errors.message}
                  </p>
                )}
              </div>

              <motion.button
                type="submit"
                disabled={isSubmitting}
                className={`w-full py-2 sm:py-3 px-4 sm:px-6 rounded-lg text-white font-semibold flex items-center justify-center space-x-2 transition-colors duration-300 text-sm sm:text-base ${
                  isSubmitting
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-purple-600 hover:bg-green-600"
                }`}
                whileHover={!isSubmitting ? { scale: 1.02 } : {}}
                whileTap={!isSubmitting ? { scale: 0.98 } : {}}
              >
                <FaPaperPlane
                  className={`w-4 h-4 sm:w-5 sm:h-5 ${
                    isSubmitting ? "animate-pulse" : ""
                  }`}
                />
                <span>{isSubmitting ? "Sending..." : "Send Message"}</span>
              </motion.button>
            </div>
          </motion.form>
        </div>
      </div>
    </>
  );
};

export default Contact;
