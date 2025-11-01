# AI-Powered Invoice Processing System

A comprehensive full-stack application for automated invoice processing using AI/ML techniques for data extraction, classification, and management.

## Project Overview

This project consists of three main components:
1. **API Service** - Python-based ML service for invoice processing
2. **Backend** - Node.js/Express server for API and database management
3. **Frontend** - React-based user interface with modern design

### Features

- 🤖 AI-powered OCR for invoice text extraction
- 📊 Machine learning-based invoice classification
- 📋 Automated data parsing and extraction
- 💾 Secure document storage and management
- 📱 Responsive web interface
- 📈 Dashboard with analytics and visualization
- 🔍 Detailed invoice viewing and management

## Tech Stack

### Frontend
- React 19 with Vite
- TailwindCSS for styling
- React Router for navigation
- Recharts for data visualization
- Axios for API communication

### Backend
- Node.js with Express
- MongoDB with Mongoose
- Multer for file uploads
- CORS enabled
- Environment variables with dotenv

### ML/API Service
- Python-based ML pipeline
- FastAPI/Flask for API endpoints
- OpenCV for image processing
- Scikit-learn for ML models
- Jupyter notebooks for development/testing

## Project Structure

```
├── api-service/           # ML and OCR processing service
│   ├── api/              # Python virtual environment
│   ├── data/             # Training and sample data
│   ├── models/           # Trained ML models
│   ├── notebooks/        # Jupyter notebooks
│   ├── utils/           # Utility functions
│   └── main.py          # Main API service
├── backend/             # Node.js backend server
│   ├── config/          # Database configuration
│   ├── controllers/     # Request handlers
│   ├── middleware/      # Custom middleware
│   ├── models/          # Database models
│   ├── routes/          # API routes
│   └── index.js         # Server entry point
└── frontend/            # React frontend
    ├── public/          # Static assets
    ├── src/             # Source code
    │   ├── assets/      # Images and resources
    │   ├── pages/       # Page components
    │   └── App.jsx      # Main application
    └── index.html       # HTML entry point
```

## Getting Started

### Prerequisites
- Node.js (v18+)
- Python 3.13+
- MongoDB
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-invoice
   ```

2. **Setup API Service**
   ```bash
   cd api-service
   python -m venv api
   source api/bin/activate  # On Windows: api\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Setup Backend**
   ```bash
   cd backend
   npm install
   cp .env.example .env  # Configure your environment variables
   ```

4. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   cp .env.example .env  # Configure your environment variables
   ```

### Running the Application

1. **Start the API Service**
   ```bash
   cd api-service
   uvicorn main:app --reload
   ```

2. **Start the Backend Server**
   ```bash
   cd backend
   npm run dev
   ```

3. **Start the Frontend Development Server**
   ```bash
   cd frontend
   npm run dev
   ```

Access the application at `http://localhost:5173`

## Development

- Frontend development server runs on port 5173
- Backend API runs on port 3000
- ML service API runs on port 8000

## Features in Detail

1. **Invoice Upload**
   - Supports PDF and image formats
   - Automatic OCR processing
   - Real-time upload status

2. **Invoice Processing**
   - Text extraction using OCR
   - Automated data field extraction
   - ML-based document classification
   - Data validation and verification

3. **Dashboard**
   - Summary statistics
   - Processing status
   - Recent uploads
   - Analytics visualization

4. **Invoice Management**
   - Detailed view of processed invoices
   - Edit and update capabilities
   - Search and filter functionality
   - Export options

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License 

## Acknowledgments

- OpenCV for image processing capabilities
- Scikit-learn for machine learning tools
- MongoDB for database solutions
- React and Vite communities for frontend tools
