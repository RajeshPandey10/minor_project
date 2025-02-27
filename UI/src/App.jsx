import { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import Loader from "./components/Loader.jsx";
import BackToTop from "./components/BackToTop";
import Footer from "./components/Footer";
import Home from "./pages/home/Home.jsx";
import Navbar from "./components/Navbar";
import About from "./pages/about/About";
import Features from "./pages/features/Features";
import HowItWork from "./pages/howItWork/HowItWork";
import Faq from "./components/Faq";
import Contact from "./pages/message/Contact";
import Admin from "./components/Admin.jsx";

const AppContent = () => {
  const [loading, setLoading] = useState(true);
  const location = useLocation();

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 2000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen bg-white">
      {loading ? (
        <Loader />
      ) : (
        <div className="relative">
          {/* Conditionally render Navbar if not on admin path */}
          {location.pathname !== "/admin" && <Navbar />}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/features" element={<Features />} />
            <Route path="/how-to-use" element={<HowItWork />} />
            <Route path="/faq" element={<Faq />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/admin" element={<Admin />} />
          </Routes>
          {/* Conditionally render BackToTop and Footer if not on admin path */}
          {location.pathname !== "/admin" && (
            <>
              <BackToTop />
              <Footer />
            </>
          )}
        </div>
      )}
    </div>
  );
};

const App = () => (
  <Router>
    <AppContent />
  </Router>
);

export default App;
