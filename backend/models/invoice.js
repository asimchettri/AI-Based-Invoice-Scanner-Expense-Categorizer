const mongoose = require('mongoose');

const InvoiceSchema = new mongoose.Schema({
  // File Information
  filename: {
    type: String,
    required: true,
    trim: true
  },
  originalname: {
    type: String,
    required: true,
    trim: true
  },
  file_type: {
    type: String,
    enum: [ 'image/png',
    'image/jpeg',
    'image/jpg',
    'image/webp',
    'image/bmp',
    'image/tiff',
    'application/pdf'],
    required: true
  },
  path: {
    type: String,
    required: true
  },
  char_count: {
    type: Number,
    default: 0
  },

  // OCR Data
  raw_text: {
    type: String,
    required: true
  },

  // Parsed Invoice Data
  vendor: {
    type: String,
    trim: true,
    index: true
  },
  invoice_number: {
    type: String,
    trim: true,
    sparse: true,
    index: true
  },
  po_number: {
    type: String,
    trim: true,
    sparse: true
  },
  invoice_date: {
    type: Date,
    sparse: true,
    index: true
  },
  due_date: {
    type: Date,
    sparse: true
  },
  
  // Financial Information
  subtotal: {
    type: Number,
    sparse: true,
    min: 0
  },
  tax: {
    type: Number,
    sparse: true,
    min: 0
  },
  total: {
    type: Number,
    required: false,
    index: true,
    min: 0
  },

  // Tax Information
  gst_number: {
    type: String,
    trim: true,
    sparse: true
  },

  // Contact Information
  email: {
    type: String,
    trim: true,
    lowercase: true,
    sparse: true,
    match: [/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/, 'Please provide a valid email']
  },
  phone: {
    type: String,
    trim: true,
    sparse: true
  },

  // Classification
  category: {
    type: String,
    enum: [
      'travel',
      'meals',
      'saas',
      'office',
      'utilities',
      'healthcare',
      'retail',
      'education',
      'entertainment',
      'maintenance',
      'transport',
      'other'
    ],
    default: 'other',
    index: true
  },
  classification_confidence: {
    type: Number,
    min: 0,
    max: 1,
    sparse: true
  },

  classification_source: {
     type: String,
     enum: ['model', 'rule'],
     default: 'rule' },

  // Processing Information
  status: {
    type: String,
    enum: ['pending', 'processing', 'completed', 'failed'],
    default: 'completed'
  },
  processing_errors: {
    parse_error: String,
    classification_error: String,
    ocr_error: String
  },

  // Metadata
  created_by: {
    type: String,
    trim: true,
    sparse: true
  },
  createdAt: {
    type: Date,
    default: Date.now,
    index: true
  },
  updatedAt: {
    type: Date,
    default: Date.now
  },
  processed_at: {
    type: Date,
    sparse: true
  }
}, {
  timestamps: true,
  collection: 'invoices'
});


InvoiceSchema.index({ vendor: 1, invoice_date: -1 });
InvoiceSchema.index({ category: 1, createdAt: -1 });
InvoiceSchema.index({ invoice_number: 1, vendor: 1 });


InvoiceSchema.virtual('fileUrl').get(function() {
  return `/uploads/${this.filename}`;
});

// Pre-save middleware to update timestamp
InvoiceSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Method to format invoice data
InvoiceSchema.methods.toJSON = function() {
  const invoice = this.toObject();
  delete invoice.__v;
  return invoice;
};


InvoiceSchema.statics.findByCategory = function(category) {
  return this.find({ category: category }).sort({ createdAt: -1 });
};


InvoiceSchema.statics.findByVendor = function(vendor) {
  return this.find({ vendor: new RegExp(vendor, 'i') }).sort({ createdAt: -1 });
};


InvoiceSchema.statics.getStats = function() {
  return this.aggregate([
    {
      $group: {
        _id: '$category',
        count: { $sum: 1 },
        total_amount: { $sum: '$total' },
        avg_amount: { $avg: '$total' }
      }
    },
    {
      $sort: { count: -1 }
    }
  ]);
};

module.exports = mongoose.model('Invoice', InvoiceSchema);