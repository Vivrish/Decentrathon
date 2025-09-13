import "./App.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import MainPage from "./pages/MainPage";
import TipsPage from "./pages/TipsPage";
import UploadImagePage from "./pages/UploadImagePage";
import UploadImageResultPage from "./pages/UploadImageResultPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/tips" element={<TipsPage />} />
        <Route path="/uploadImage" element={<UploadImagePage />} />
        <Route path="/uploadImage/result" element={<UploadImageResultPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
