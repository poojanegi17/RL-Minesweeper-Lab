import { Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Layout } from "@/components/layout/Layout";
import { Home } from "@/pages/Home";
import { Agents } from "@/pages/Agents";
import { AgentDetail } from "@/pages/AgentDetail";
import { Research } from "@/pages/Research";
import { Compare } from "@/pages/Compare";
import { Replay } from "@/pages/Replay";
import { About } from "@/pages/About";
import { NotFound } from "@/pages/NotFound";

function App() {
  const location = useLocation();

  return (
    <Layout>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.18, ease: "easeOut" }}
        >
          <Routes location={location}>
            <Route path="/" element={<Home />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/agents/:agentId" element={<AgentDetail />} />
            <Route path="/research" element={<Research />} />
            <Route path="/research/:agentSlug" element={<Research />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/replay" element={<Replay />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </motion.div>
      </AnimatePresence>
    </Layout>
  );
}

export default App;
