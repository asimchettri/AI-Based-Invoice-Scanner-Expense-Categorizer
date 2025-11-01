const express = require('express');
const router = express.Router();
const upload = require('../middleware/upload');

const {
  uploadInvoice,
  getInvoices,
  getInvoiceById,
  downloadInvoice,
  deleteInvoice,
  getStatistics,updateInvoice
} = require('../controllers/invoiceController');



router.post('/upload', upload.single('invoice'), uploadInvoice);
router.get('/', getInvoices);
router.get('/statistics', getStatistics);
router.get('/download/:filename', downloadInvoice);

//  Put update BEFORE get by ID
router.put('/:id', updateInvoice);
router.get('/:id', getInvoiceById);
router.delete('/:id', deleteInvoice);

module.exports = router;