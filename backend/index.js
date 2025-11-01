const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const multer = require('multer'); 
require('dotenv').config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

require('dotenv').config();
const connectDB = require('./config/db');
connectDB();




// Create uploads folder if not exists
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Routes
const invoiceRoutes = require('./routes/invoiceRoutes');
app.use('/api/invoices', invoiceRoutes);

// Serve static files 
app.use('/uploads', express.static(uploadDir));



// Error handling middleware for multer
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError && error.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({ message: 'File size exceeds 5MB limit' });
  }
  res.status(500).json({ message: error.message });
});




// Test route
app.get('/', (req, res) => res.json({ status: 'backend ok' }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
