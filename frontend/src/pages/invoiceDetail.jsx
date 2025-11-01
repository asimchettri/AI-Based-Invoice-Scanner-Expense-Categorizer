import React, { useState, useEffect } from 'react';
import { ArrowLeft, Edit2, Save, X, Loader, Download, Trash2, Calendar, DollarSign, FileText, Mail, Phone, Building } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';

const API_BASE = 'http://localhost:3000/api/invoices';

const CATEGORIES = ['travel', 'meals', 'saas', 'office', 'utilities', 'healthcare', 'retail', 'education', 'entertainment', 'maintenance', 'other'];

const InvoiceDetail = ({ invoiceId }) => {
  const [invoice, setInvoice] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(false);
  const [editedInvoice, setEditedInvoice] = useState(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ text: '', type: '' });
  
  const navigate = useNavigate();
  const { id: paramId } = useParams(); // If using route params
  const id = invoiceId || paramId;


  useEffect(() => {
    if (id) {
      fetchInvoice();
    }
  }, [id]);

  const fetchInvoice = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/${id}`);
      
      if (!response.ok) {
        throw new Error('Invoice not found');
      }
      
      const data = await response.json();
      const invoiceData = data.data || data;
      setInvoice(invoiceData);
      setEditedInvoice(invoiceData);
    } catch (err) {
      console.error('Fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFieldChange = (field, value) => {
    setEditedInvoice(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      
      const response = await fetch(`${API_BASE}/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(editedInvoice),
      });

      if (!response.ok) {
        throw new Error('Failed to save changes');
      }

      const data = await response.json();
      setInvoice(data.data);
      setEditedInvoice(data.data);
      setEditing(false);
      
      setMessage({ text: 'Invoice updated successfully!', type: 'success' });
      setTimeout(() => setMessage({ text: '', type: '' }), 3000);
    } catch (err) {
      console.error('Save error:', err);
      setMessage({ text: `Failed to save: ${err.message}`, type: 'error' });
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this invoice? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete invoice');
      }

      navigate('/invoices');
    } catch (err) {
      setMessage({ text: `Failed to delete: ${err.message}`, type: 'error' });
    }
  };

 const handleDownload = async () => {
  if (!invoice?.file?.filename) {
    setMessage({ text: 'No file available to download', type: 'error' });
    return;
  }

  try {
    
    const response = await fetch(`${API_BASE}/download/${invoice.file.filename}`);
    
    if (!response.ok) {
      throw new Error('Failed to download file');
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = invoice.file.originalName || 'invoice';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (err) {
    setMessage({ text: `Failed to download: ${err.message}`, type: 'error' });
  }
};

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-400">Loading invoice...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-8 max-w-md text-center">
          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <X className="w-8 h-8 text-red-400" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Invoice Not Found</h2>
          <p className="text-slate-400 mb-6">{error}</p>
          <button
            onClick={() => navigate('/invoices')}
            className="px-6 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg font-semibold transition-colors"
          >
            Back to Invoices
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-2 text-slate-400 hover:text-white mb-4 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            Back
          </button>
          
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Invoice Details</h1>
              <p className="text-slate-400">Invoice ID: {invoice._id}</p>
            </div>
            
            <div className="flex gap-2">
              {!editing ? (
                <>
                  <button
                    onClick={() => setEditing(true)}
                    className="px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg font-semibold flex items-center gap-2 transition-colors"
                  >
                    <Edit2 className="w-4 h-4" />
                    Edit
                  </button>
                  <button
                    onClick={handleDownload}
                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold flex items-center gap-2 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </button>
                  <button
                    onClick={handleDelete}
                    className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg font-semibold flex items-center gap-2 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={handleSave}
                    disabled={saving}
                    className="px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg font-semibold flex items-center gap-2 transition-colors disabled:opacity-50"
                  >
                    {saving ? (
                      <Loader className="w-4 h-4 animate-spin" />
                    ) : (
                      <Save className="w-4 h-4" />
                    )}
                    Save
                  </button>
                  <button
                    onClick={() => {
                      setEditing(false);
                      setEditedInvoice(invoice);
                    }}
                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition-colors"
                  >
                    Cancel
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Messages */}
        {message.text && (
          <div
            className={`mb-6 p-4 rounded-lg flex items-center gap-3 ${
              message.type === 'success'
                ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
                : 'bg-red-500/10 border border-red-500/30 text-red-400'
            }`}
          >
            {message.text}
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Main Details */}
          <div className="lg:col-span-2 space-y-6">
            {/* Basic Info Card */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
              <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <Building className="w-5 h-5 text-cyan-400" />
                Vendor Information
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                    Vendor Name
                  </label>
                  {editing ? (
                    <input
                      type="text"
                      value={editedInvoice.vendor || ''}
                      onChange={(e) => handleFieldChange('vendor', e.target.value)}
                      className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                  ) : (
                    <p className="text-white text-lg font-semibold">
                      {invoice.vendor || 'N/A'}
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                      <Mail className="w-3 h-3 inline mr-1" />
                      Email
                    </label>
                    {editing ? (
                      <input
                        type="email"
                        value={editedInvoice.email || ''}
                        onChange={(e) => handleFieldChange('email', e.target.value)}
                        className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white">{invoice.email || 'N/A'}</p>
                    )}
                  </div>

                  <div>
                    <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                      <Phone className="w-3 h-3 inline mr-1" />
                      Phone
                    </label>
                    {editing ? (
                      <input
                        type="tel"
                        value={editedInvoice.phone || ''}
                        onChange={(e) => handleFieldChange('phone', e.target.value)}
                        className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white">{invoice.phone || 'N/A'}</p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Invoice Details Card */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
              <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <FileText className="w-5 h-5 text-cyan-400" />
                Invoice Details
              </h2>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                    Invoice Number
                  </label>
                  {editing ? (
                    <input
                      type="text"
                      value={editedInvoice.invoice_number || ''}
                      onChange={(e) => handleFieldChange('invoice_number', e.target.value)}
                      className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                  ) : (
                    <p className="text-white font-semibold">{invoice.invoice_number || 'N/A'}</p>
                  )}
                </div>

                <div>
                  <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                    <Calendar className="w-3 h-3 inline mr-1" />
                    Invoice Date
                  </label>
                  {editing ? (
                    <input
                      type="date"
                      value={editedInvoice.invoice_date ? new Date(editedInvoice.invoice_date).toISOString().split('T')[0] : ''}
                      onChange={(e) => handleFieldChange('invoice_date', e.target.value)}
                      className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                  ) : (
                    <p className="text-white font-semibold">
                      {invoice.invoice_date ? new Date(invoice.invoice_date).toLocaleDateString() : 'N/A'}
                    </p>
                  )}
                </div>

                <div className="col-span-2">
                  <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                    Category
                  </label>
                  {editing ? (
                    <select
                      value={editedInvoice.category || 'other'}
                      onChange={(e) => handleFieldChange('category', e.target.value)}
                      className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    >
                      {CATEGORIES.map(cat => (
                        <option key={cat} value={cat} className="capitalize">{cat}</option>
                      ))}
                    </select>
                  ) : (
                    <span className="inline-block px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm font-semibold capitalize">
                      {invoice.category || 'other'}
                      {invoice.category_confidence && (
                        <span className="ml-2 text-xs opacity-70">
                          ({(invoice.category_confidence * 100).toFixed(0)}%)
                        </span>
                      )}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* File Info Card */}
            {invoice.file && (
              <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
                <h2 className="text-xl font-bold text-white mb-4">File Information</h2>
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-cyan-500/20 rounded-lg">
                    <FileText className="w-6 h-6 text-cyan-400" />
                  </div>
                  <div className="flex-1">
                    <p className="text-white font-semibold">{invoice.file.originalName}</p>
                    <p className="text-slate-400 text-sm">{invoice.file.mimetype}</p>
                  </div>
                  <button
                    onClick={handleDownload}
                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Financial Summary */}
          <div className="space-y-6">
            {/* Financial Card */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-800/50 rounded-xl border border-slate-700 p-6 sticky top-6">
              <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-emerald-400" />
                Financial Summary
              </h2>
              
              <div className="space-y-4">
                <div className="pb-4 border-b border-slate-700">
                  <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                    Subtotal
                  </label>
                  {editing ? (
                    <input
                      type="number"
                      step="0.01"
                      value={editedInvoice.subtotal || ''}
                      onChange={(e) => handleFieldChange('subtotal', parseFloat(e.target.value) || 0)}
                      className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                  ) : (
                    <p className="text-2xl font-bold text-white">
                      ${(invoice.subtotal || 0).toFixed(2)}
                    </p>
                  )}
                </div>

                <div className="pb-4 border-b border-slate-700">
                  <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                    Tax
                  </label>
                  {editing ? (
                    <input
                      type="number"
                      step="0.01"
                      value={editedInvoice.tax || ''}
                      onChange={(e) => handleFieldChange('tax', parseFloat(e.target.value) || 0)}
                      className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                  ) : (
                    <p className="text-2xl font-bold text-white">
                      ${(invoice.tax || 0).toFixed(2)}
                    </p>
                  )}
                </div>

                <div className="pt-2">
                  <label className="block text-xs text-slate-400 uppercase tracking-wide mb-2">
                    Total Amount
                  </label>
                  {editing ? (
                    <input
                      type="number"
                      step="0.01"
                      value={editedInvoice.total || ''}
                      onChange={(e) => handleFieldChange('total', parseFloat(e.target.value) || 0)}
                      className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    />
                  ) : (
                    <p className="text-3xl font-bold text-emerald-400">
                      ${(invoice.total || 0).toFixed(2)}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Metadata Card */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
              <h2 className="text-lg font-bold text-white mb-4">Metadata</h2>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-slate-400">Created</p>
                  <p className="text-white">
                    {invoice.createdAt ? new Date(invoice.createdAt).toLocaleString() : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-slate-400">Last Updated</p>
                  <p className="text-white">
                    {invoice.updatedAt ? new Date(invoice.updatedAt).toLocaleString() : 'N/A'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InvoiceDetail;