

const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');
const Invoice = require('../models/invoice');

const uploadDir = path.join(__dirname, '../uploads');
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
const OCR_TIMEOUT = 60000; // 60 seconds


exports.uploadInvoice = async (req, res) => {
  let localPath = null;

  try {
    if (!req.file) {
      return res.status(400).json({ 
        error: 'No file uploaded',
        message: 'Please provide a file to upload',
        status: 'error'
      });
    }

    localPath = req.file.path;
    const filename = req.file.filename;
    const originalName = req.file.originalname;

    console.log(`[UPLOAD] Processing file: ${originalName}`);

    // Validate file type
    const allowedMimes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
    if (!allowedMimes.includes(req.file.mimetype)) {
      fs.unlinkSync(localPath); 
      return res.status(400).json({ 
        error: 'Invalid file type',
        message: 'Only JPEG, PNG, and PDF files are allowed',
        status: 'error'
      });
    }

    // Prepare form data for FastAPI
    const formData = new FormData();
    formData.append('file', fs.createReadStream(localPath), {
      filename: originalName,
      contentType: req.file.mimetype
    });

    console.log(`[FASTAPI] Sending to: ${FASTAPI_URL}/analyze`);

    // Send to FastAPI for complete analysis
    const aiResponse = await axios.post(
      `${FASTAPI_URL}/analyze`,
      formData,
      {
        headers: formData.getHeaders(),
        timeout: OCR_TIMEOUT
      }
    );

    const processedData = aiResponse.data;

    if (!processedData || Object.keys(processedData).length === 0) {
      throw new Error('FastAPI processing failed - empty response');
    }

    console.log(`[FASTAPI] Response received:`, {
      vendor: processedData.vendor,
      category: processedData.category,
      total: processedData.total || processedData.amount,
      source: processedData.source,
      confidence: processedData.confidence
    });

    // Build complete invoice document for MongoDB
    const invoiceDoc = {
      // File Information
      filename,
      originalname: originalName,
      file_type: req.file.mimetype,
      path: localPath,
      
      // Basic Invoice Data
      vendor: processedData.vendor || 'Unknown',
      invoice_number: processedData.invoice_number || null,
      po_number: processedData.po_number || null,
      description: processedData.description || '',
      
      // Dates
      invoice_date: processedData.invoice_date 
        ? new Date(processedData.invoice_date) 
        : new Date(),
      due_date: processedData.due_date 
        ? new Date(processedData.due_date) 
        : null,
      
      // Financial Data - ALL AMOUNTS
      amount: processedData.amount || processedData.total || 0,
      total: processedData.total || processedData.amount || 0,
      subtotal: processedData.subtotal || null,
      tax: processedData.tax || null,
      
      // Additional Information
      gst_number: processedData.gst_number || null,
      email: processedData.email || null,
      phone: processedData.phone || null,
      
      // ML Classification with Source Tracking
      category: processedData.category || 'other',
      classification_confidence: processedData.confidence || 0,
      classification_source: processedData.source || 'rule',
      
      // OCR Data
      raw_text: processedData.raw_text || '',
      char_count: processedData.char_count || (processedData.raw_text ? processedData.raw_text.length : 0),
      
      // Processing Status
      status: 'completed',
      processed_at: new Date(),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    // Store processing errors if any
    if (processedData.error || processedData.warning) {
      invoiceDoc.processing_errors = {
        error: processedData.error || null,
        warning: processedData.warning || null
      };
    }

    // Save to MongoDB
    console.log(`[DB] Saving to MongoDB...`);
    const savedInvoice = await Invoice.create(invoiceDoc);
    console.log(`[SUCCESS] Invoice saved with ID: ${savedInvoice._id}`);

    // Return complete response with ALL invoice data
    res.status(201).json({
      message: 'Invoice uploaded and analyzed successfully',
      status: 'success',
      data: {
        // File Information
        file: {
          filename: savedInvoice.filename,
          originalName: savedInvoice.originalname,
          size: req.file.size,
          mimetype: req.file.mimetype,
          uploadedAt: savedInvoice.createdAt,
          path: `/uploads/${savedInvoice.filename}`
        },
        
        // Complete Invoice Data
        invoice: {
          _id: savedInvoice._id,
          
          // Basic Info
          vendor: savedInvoice.vendor,
          invoice_number: savedInvoice.invoice_number,
          po_number: savedInvoice.po_number,
          description: savedInvoice.description,
          
          // Dates
          invoice_date: savedInvoice.invoice_date,
          due_date: savedInvoice.due_date,
          
          // Financial Data (ALL amounts)
          amount: savedInvoice.amount,
          total: savedInvoice.total,
          subtotal: savedInvoice.subtotal,
          tax: savedInvoice.tax,
          
          // Additional Information
          gst_number: savedInvoice.gst_number,
          email: savedInvoice.email,
          phone: savedInvoice.phone,
          
          // ML Classification
          category: savedInvoice.category,
          confidence: savedInvoice.classification_confidence, 
          source: savedInvoice.classification_source
        },
        
        // AI Analysis Summary
        ai_analysis: {
          category: savedInvoice.category,
          source: savedInvoice.classification_source,
          confidence: savedInvoice.classification_confidence,
          characters_extracted: savedInvoice.char_count
        }
      }
    });

  } catch (error) {
    console.error('[ERROR] Upload/Analysis failed:', error.message);

    // Clean up uploaded file on error
    if (localPath && fs.existsSync(localPath)) {
      fs.unlinkSync(localPath);
      console.log('[CLEANUP] Removed temporary file');
    }

    // Return specific error messages
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'FastAPI service unavailable',
        message: 'AI service is not running. Please start FastAPI server.',
        status: 'error',
        details: `Expected FastAPI at: ${FASTAPI_URL}`
      });
    }

    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      return res.status(504).json({
        error: 'Processing timeout',
        message: 'File took too long to process. Try with a smaller file.',
        status: 'error'
      });
    }

    if (error.response) {
      // FastAPI returned an error
      return res.status(error.response.status || 500).json({
        error: 'FastAPI processing error',
        message: error.response.data?.detail || error.message,
        status: 'error',
        details: error.response.data
      });
    }

    res.status(500).json({
      error: 'Upload and analysis failed',
      message: error.message,
      status: 'error'
    });
  }
};



exports.getInvoices = async (req, res) => {
  try {
    const { 
      page = 1, 
      limit = 10, 
      category, 
      vendor, 
      startDate, 
      endDate, 
      minAmount, 
      maxAmount,
      source
    } = req.query;
    
    const skip = (page - 1) * limit;
    let query = {};

    // Filter by category
    if (category) {
      query.category = category;
    }

    // Filter by classification source (model/rule)
    if (source) {
      query.classification_source = source;
    }

    // Filter by vendor (case-insensitive)
    if (vendor) {
      query.vendor = new RegExp(vendor, 'i');
    }

    // Filter by date range
    if (startDate || endDate) {
      query.invoice_date = {};
      if (startDate) {
        query.invoice_date.$gte = new Date(startDate);
      }
      if (endDate) {
        query.invoice_date.$lte = new Date(endDate);
      }
    }

    // Filter by amount range
    if (minAmount || maxAmount) {
      query.amount = {};
      if (minAmount) {
        query.amount.$gte = parseFloat(minAmount);
      }
      if (maxAmount) {
        query.amount.$lte = parseFloat(maxAmount);
      }
    }

    // Fetch invoices with pagination
    const invoices = await Invoice.find(query)
      .skip(skip)
      .limit(parseInt(limit))
      .sort({ createdAt: -1 })
      .select('-raw_text'); // Exclude large raw_text field

    const total = await Invoice.countDocuments(query);

    res.status(200).json({
      message: 'Invoices retrieved successfully',
      status: 'success',
      data: invoices,
      pagination: {
        total,
        page: parseInt(page),
        limit: parseInt(limit),
        pages: Math.ceil(total / limit)
      }
    });

  } catch (error) {
    console.error('[ERROR] Fetch invoices failed:', error.message);
    res.status(500).json({
      error: 'Error fetching invoices',
      message: error.message,
      status: 'error'
    });
  }
};



exports.getInvoiceById = async (req, res) => {
  try {
    const invoice = await Invoice.findById(req.params.id);

    if (!invoice) {
      return res.status(404).json({
        error: 'Invoice not found',
        message: `No invoice found with ID: ${req.params.id}`,
        status: 'error'
      });
    }

    res.status(200).json({
      message: 'Invoice retrieved successfully',
      status: 'success',
      data: invoice
    });

  } catch (error) {
    console.error('[ERROR] Fetch invoice by ID failed:', error.message);
    res.status(500).json({
      error: 'Error fetching invoice',
      message: error.message,
      status: 'error'
    });
  }
};



exports.downloadInvoice = (req, res) => {
  try {
    const filename = req.params.filename;
    const filepath = path.join(uploadDir, filename);

    // Security check: prevent path traversal
    if (!filepath.startsWith(uploadDir)) {
      return res.status(403).json({
        error: 'Access denied',
        message: 'Invalid file path',
        status: 'error'
      });
    }

    if (!fs.existsSync(filepath)) {
      return res.status(404).json({
        error: 'File not found',
        message: `File ${filename} does not exist`,
        status: 'error'
      });
    }

    console.log(`[DOWNLOAD] Serving file: ${filename}`);
    res.download(filepath);

  } catch (error) {
    console.error('[ERROR] Download failed:', error.message);
    res.status(500).json({
      error: 'Error downloading file',
      message: error.message,
      status: 'error'
    });
  }
};



exports.deleteInvoice = async (req, res) => {
  try {
    const invoiceId = req.params.id;

    // Find and delete from MongoDB
    const invoice = await Invoice.findByIdAndDelete(invoiceId);

    if (!invoice) {
      return res.status(404).json({
        error: 'Invoice not found',
        message: `No invoice found with ID: ${invoiceId}`,
        status: 'error'
      });
    }

    // Delete associated file
    if (invoice.path && fs.existsSync(invoice.path)) {
      fs.unlinkSync(invoice.path);
      console.log(`[DELETE] Removed file: ${invoice.path}`);
    }

    console.log(`[DELETE] Invoice deleted: ${invoiceId}`);

    res.status(200).json({
      message: 'Invoice deleted successfully',
      status: 'success',
      data: {
        deletedId: invoiceId,
        vendor: invoice.vendor,
        amount: invoice.amount,
        category: invoice.category
      }
    });

  } catch (error) {
    console.error('[ERROR] Delete failed:', error.message);
    res.status(500).json({
      error: 'Error deleting invoice',
      message: error.message,
      status: 'error'
    });
  }
};



exports.updateInvoice = async (req, res) => {
  try {
    const invoiceId = req.params.id;
    
    if (!invoiceId) {
      return res.status(400).json({ 
        error: 'Invoice ID required',
        status: 'error' 
      });
    }

  
    const allowedFields = [
      'vendor', 'invoice_number', 'invoice_date', 'due_date',
      'subtotal', 'tax', 'total', 'amount', 'category', 'description',
      'email', 'phone', 'po_number', 'gst_number'
    ];

    const updateData = {};
    allowedFields.forEach(field => {
      if (req.body[field] !== undefined) {
        updateData[field] = req.body[field];
      }
    });

    
    updateData.updatedAt = new Date();

    console.log(`[UPDATE] Updating invoice ${invoiceId}:`, updateData);

    const updatedInvoice = await Invoice.findByIdAndUpdate(
      invoiceId,
      updateData,
      { new: true, runValidators: true }
    );

    if (!updatedInvoice) {
      return res.status(404).json({
        error: 'Invoice not found',
        message: `No invoice found with ID: ${invoiceId}`,
        status: 'error'
      });
    }

    console.log(`[SUCCESS] Invoice updated: ${invoiceId}`);

    res.json({
      message: 'Invoice updated successfully',
      status: 'success',
      data: updatedInvoice
    });

  } catch (error) {
    console.error('[ERROR] Update failed:', error.message);
    res.status(500).json({
      error: 'Error updating invoice',
      message: error.message,
      status: 'error'
    });
  }
};



exports.getStatistics = async (req, res) => {
  try {
   
    const categoryStats = await Invoice.aggregate([
      {
        $group: {
          _id: '$category',
          count: { $sum: 1 },
          totalAmount: { $sum: '$amount' },
          avgAmount: { $avg: '$amount' },
          maxAmount: { $max: '$amount' },
          minAmount: { $min: '$amount' }
        }
      },
      {
        $sort: { totalAmount: -1 }
      }
    ]);

    const sourceStats = await Invoice.aggregate([
      {
        $group: {
          _id: '$classification_source',
          count: { $sum: 1 },
          avgConfidence: { $avg: '$classification_confidence' }
        }
      }
    ]);

    const totalInvoices = await Invoice.countDocuments();
    
    const financials = await Invoice.aggregate([
      {
        $group: {
          _id: null,
          total: { $sum: '$amount' },
          avg: { $avg: '$amount' },
          max: { $max: '$amount' },
          min: { $min: '$amount' }
        }
      }
    ]);

   
    const recentInvoices = await Invoice.find()
      .sort({ createdAt: -1 })
      .limit(5)
      .select('vendor amount category classification_source invoice_date createdAt');

    res.status(200).json({
      message: 'Statistics retrieved successfully',
      status: 'success',
      data: {
        overview: {
          totalInvoices,
          totalAmount: financials[0]?.total || 0,
          averageAmount: financials[0]?.avg || 0,
          maxAmount: financials[0]?.max || 0,
          minAmount: financials[0]?.min || 0
        },
        byCategory: categoryStats,
        bySource: sourceStats,
        recentInvoices
      }
    });

  } catch (error) {
    console.error('[ERROR] Statistics fetch failed:', error.message);
    res.status(500).json({
      error: 'Error fetching statistics',
      message: error.message,
      status: 'error'
    });
  }
};



exports.exportForTraining = async (req, res) => {
  try {
    const invoices = await Invoice.find({})
      .select('vendor description amount category')
      .lean();

    const trainingData = invoices.map(inv => ({
      vendor: inv.vendor || '',
      description: inv.description || '',
      amount: inv.amount || 0,
      category: inv.category || 'other'
    }));

    res.status(200).json({
      message: 'Training data exported successfully',
      status: 'success',
      count: trainingData.length,
      data: trainingData
    });

  } catch (error) {
    console.error('[ERROR] Export failed:', error.message);
    res.status(500).json({
      error: 'Error exporting data',
      message: error.message,
      status: 'error'
    });
  }
};



exports.checkFastAPIHealth = async (req, res) => {
  try {
    const response = await axios.get(`${FASTAPI_URL}/health`, {
      timeout: 5000
    });

    res.status(200).json({
      message: 'FastAPI service is healthy',
      status: 'success',
      fastapi: {
        url: FASTAPI_URL,
        status: response.data.status,
        version: response.data.version || 'unknown',
        ml_model_loaded: response.data.ml_model_loaded || false
      }
    });

  } catch (error) {
    res.status(503).json({
      message: 'FastAPI service is unavailable',
      status: 'error',
      fastapi: {
        url: FASTAPI_URL,
        error: error.message
      }
    });
  }
};
