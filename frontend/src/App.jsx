import { BrowserRouter, Routes, Route } from "react-router-dom";
import UploadComponent from "./pages/upload.jsx";
import InvoiceDetail from "./pages/invoiceDetail.jsx";
import Dashboard from "./pages/dashboard.jsx";
import InvoicesList from "./pages/invoiceList.jsx";
import "./index.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
       <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<UploadComponent />} />
        <Route path="/invoices" element={<InvoicesList />} />
        <Route path="/invoice/:id" element={<InvoiceDetail />} />
        
        {/* Optional: Redirect any unknown routes to dashboard */}
        <Route path="*" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
