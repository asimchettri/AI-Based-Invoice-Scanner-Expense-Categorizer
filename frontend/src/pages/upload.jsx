import React, { useState, useRef } from 'react';
import { Upload, X, CheckCircle, AlertCircle, Loader, Edit2, Save } from 'lucide-react';

const BASE_URL = import.meta.env.VITE_API_BASE_URL;

const CONFIG = {
  API_ENDPOINT: `${BASE_URL}/invoices/upload`,
  UPDATE_ENDPOINT: `${BASE_URL}/invoices`,
  MAX_FILE_SIZE: 5 * 1024 * 1024,
  ALLOWED_TYPES: ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'],
  MESSAGE_DURATION: 5000,
};


const CATEGORIES = [
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
  'other'
];

export default function UploadComponent() {
  const fileInputRef = useRef(null);
  const [state, setState] = useState({
    file: null,
    message: '',
    messageType: '',
    loading: false,
    uploadProgress: 0,
    dragActive: false,
  });

  const [invoice, setInvoice] = useState(null);
  const [editing, setEditing] = useState(false);
  const [editedInvoice, setEditedInvoice] = useState(null);
  const [savingCorrection, setSavingCorrection] = useState(false);

  const updateState = (updates) => {
    setState((prev) => ({ ...prev, ...updates }));
  };

  const validateAndSetFile = (selectedFile) => {
    if (!CONFIG.ALLOWED_TYPES.includes(selectedFile.type)) {
      updateState({
        message: 'Only PDF and image files are allowed!',
        messageType: 'error',
        file: null,
      });
      return;
    }

    if (selectedFile.size > CONFIG.MAX_FILE_SIZE) {
      updateState({
        message: 'File size exceeds 5MB limit!',
        messageType: 'error',
        file: null,
      });
      return;
    }

    updateState({
      file: selectedFile,
      message: '',
      messageType: '',
    });
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      validateAndSetFile(selectedFile);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      updateState({ dragActive: true });
    } else if (e.type === 'dragleave') {
      updateState({ dragActive: false });
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    updateState({ dragActive: false });
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      validateAndSetFile(droppedFile);
    }
  };

  const simulateProgress = async () => {
    const intervals = [100, 300, 500, 800, 1200, 1500, 1800, 2000];
    let currentProgress = 10;
    for (const interval of intervals) {
      await new Promise((resolve) => setTimeout(resolve, interval));
      currentProgress = Math.min(currentProgress + Math.random() * 20, 90);
      updateState({ uploadProgress: currentProgress });
    }
  };

  const handleUpload = async () => {
  if (!state.file) {
    updateState({
      message: 'Please select a file first!',
      messageType: 'error',
    });
    return;
  }

  const formData = new FormData();
  formData.append('invoice', state.file);

  try {
    updateState({ loading: true, uploadProgress: 10 });
    const progressPromise = simulateProgress();

    const response = await fetch(CONFIG.API_ENDPOINT, {
      method: 'POST',
      body: formData,
    });

    await progressPromise;

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Upload failed');
    }

    const data = await response.json();
    
    // Debug logging
    console.log('Full upload response:', data);
    console.log('Invoice data:', data.data);
    console.log('Invoice ID:', data.data?.invoice?._id);

    
    setInvoice(data.data); 
    setEditedInvoice(data.data.invoice || {});
    setEditing(false);

    updateState({
      message: 'Invoice uploaded and processed successfully!',
      messageType: 'success',
      file: null,
      uploadProgress: 100,
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    setTimeout(() => {
      updateState({ message: '', messageType: '' });
    }, CONFIG.MESSAGE_DURATION);
  } catch (error) {
    console.error('Upload failed:', error);
    updateState({
      message: error.message || 'Upload failed. Please try again.',
      messageType: 'error',
    });
  } finally {
    updateState({ loading: false, uploadProgress: 0 });
  }
};

  const handleClear = () => {
    updateState({
      file: null,
      message: '',
      messageType: '',
      uploadProgress: 0,
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const startEditing = () => {
    setEditedInvoice({ ...invoice.invoice });
    setEditing(true);
  };

  const cancelEditing = () => {
    setEditing(false);
    setEditedInvoice(null);
  };

  const handleFieldChange = (field, value) => {
    setEditedInvoice((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const saveCorrection = async () => {
    
    if (!invoice?.invoice?._id) {
      console.error('Invoice ID not found. Invoice object:', invoice);
      updateState({
        message: 'Invoice ID not found. Please upload again.',
        messageType: 'error',
      });
      return;
    }

    const invoiceId = invoice.invoice._id;
    console.log('Saving invoice ID:', invoiceId);
    console.log('Updated data:', editedInvoice);

    try {
      setSavingCorrection(true);

      const url = `${CONFIG.UPDATE_ENDPOINT}/${invoiceId}`;
      console.log('Update URL:', url);

      const response = await fetch(url, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(editedInvoice),
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Update error:', errorData);
        throw new Error(errorData.message || 'Failed to save corrections');
      }

      const updatedData = await response.json();
      console.log('Update response:', updatedData);
      
   
      setInvoice({ ...invoice, invoice: updatedData.data });
      setEditing(false);

      updateState({
        message: 'Invoice corrected and saved successfully!',
        messageType: 'success',
      });

      setTimeout(() => {
        updateState({ message: '', messageType: '' });
      }, CONFIG.MESSAGE_DURATION);
    } catch (error) {
      console.error('Save failed:', error);
      updateState({
        message: `Failed to save: ${error.message}`,
        messageType: 'error',
      });
    } finally {
      setSavingCorrection(false);
    }
  };

  if (invoice) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
        <div className="w-full max-w-2xl">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Invoice Results
            </h1>
            <p className="text-slate-400 text-sm">
              Review and correct the extracted data
            </p>
          </div>

          {/* Main Card */}
          <div className="bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-700">
            {/* File Info */}
            <div className="p-8 border-b border-slate-700">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wide">
                    File Name
                  </p>
                  <p className="text-white font-semibold mt-1">
                    {invoice.file?.originalName}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wide">
                    Invoice ID
                  </p>
                  <p className="text-white font-semibold mt-1 text-xs">
                    {invoice.invoice?._id}
                  </p>
                </div>
              </div>
            </div>

            {/* Invoice Data */}
            <div className="p-8 space-y-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-white">
                  Extracted Invoice Data
                </h2>
                {!editing ? (
                  <button
                    onClick={startEditing}
                    className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg transition-colors"
                  >
                    <Edit2 className="w-4 h-4" />
                    Edit
                  </button>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={saveCorrection}
                      disabled={savingCorrection}
                      className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg transition-colors disabled:opacity-50"
                    >
                      {savingCorrection ? (
                        <Loader className="w-4 h-4 animate-spin" />
                      ) : (
                        <Save className="w-4 h-4" />
                      )}
                      Save
                    </button>
                    <button
                      onClick={cancelEditing}
                      className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                )}
              </div>

              <div className="space-y-4">
                {/* Vendor */}
                <div>
                  <label className="text-xs text-slate-400 uppercase tracking-wide">
                    Vendor
                  </label>
                  {editing ? (
                    <input
                      type="text"
                      value={editedInvoice?.vendor || ''}
                      onChange={(e) => handleFieldChange('vendor', e.target.value)}
                      className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                    />
                  ) : (
                    <p className="text-white font-semibold mt-2">
                      {invoice.invoice?.vendor || 'N/A'}
                    </p>
                  )}
                </div>

                {/* Invoice Number and Date */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-slate-400 uppercase tracking-wide">
                      Invoice #
                    </label>
                    {editing ? (
                      <input
                        type="text"
                        value={editedInvoice?.invoice_number || ''}
                        onChange={(e) => handleFieldChange('invoice_number', e.target.value)}
                        className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white font-semibold mt-2">
                        {invoice.invoice?.invoice_number || 'N/A'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 uppercase tracking-wide">
                      Date
                    </label>
                    {editing ? (
                      <input
                        type="date"
                        value={
                          editedInvoice?.invoice_date
                            ? new Date(editedInvoice.invoice_date).toISOString().split('T')[0]
                            : ''
                        }
                        onChange={(e) => handleFieldChange('invoice_date', e.target.value)}
                        className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white font-semibold mt-2">
                        {invoice.invoice?.invoice_date
                          ? new Date(invoice.invoice.invoice_date).toLocaleDateString()
                          : 'N/A'}
                      </p>
                    )}
                  </div>
                </div>

                {/* Financial Details */}
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="text-xs text-slate-400 uppercase tracking-wide">
                      Subtotal
                    </label>
                    {editing ? (
                      <input
                        type="number"
                        step="0.01"
                        value={editedInvoice?.subtotal || ''}
                        onChange={(e) => handleFieldChange('subtotal', parseFloat(e.target.value))}
                        className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white font-semibold mt-2">
                        ${invoice.invoice?.subtotal?.toFixed(2) || '0.00'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 uppercase tracking-wide">
                      Tax
                    </label>
                    {editing ? (
                      <input
                        type="number"
                        step="0.01"
                        value={editedInvoice?.tax || ''}
                        onChange={(e) => handleFieldChange('tax', parseFloat(e.target.value))}
                        className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white font-semibold mt-2">
                        ${invoice.invoice?.tax?.toFixed(2) || '0.00'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 uppercase tracking-wide">
                      Total
                    </label>
                    {editing ? (
                      <input
                        type="number"
                        step="0.01"
                        value={editedInvoice?.total || ''}
                        onChange={(e) => handleFieldChange('total', parseFloat(e.target.value))}
                        className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white font-semibold mt-2 text-emerald-400">
                        ${invoice.invoice?.total?.toFixed(2) || '0.00'}
                      </p>
                    )}
                  </div>
                </div>

                {/* Category */}
                <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <label className="text-xs text-slate-400 uppercase tracking-wide">
                    Category
                  </label>
                  {editing ? (
                    <select
                      value={editedInvoice?.category || 'other'}
                      onChange={(e) => handleFieldChange('category', e.target.value)}
                      className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    >
                      {CATEGORIES.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="flex items-center gap-2 mt-2">
                      <p className="text-white font-semibold capitalize">
                        {invoice.invoice?.category || 'other'}
                      </p>
                      {invoice.invoice?.category_confidence && (
                        <p className="text-xs text-slate-400">
                          (Confidence: {(invoice.invoice.category_confidence * 100).toFixed(1)}%)
                        </p>
                      )}
                    </div>
                  )}
                </div>

                {/* Contact Info */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-slate-400 uppercase tracking-wide">
                      Email
                    </label>
                    {editing ? (
                      <input
                        type="email"
                        value={editedInvoice?.email || ''}
                        onChange={(e) => handleFieldChange('email', e.target.value)}
                        className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white font-semibold mt-2 text-sm">
                        {invoice.invoice?.email || 'N/A'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="text-xs text-slate-400 uppercase tracking-wide">
                      Phone
                    </label>
                    {editing ? (
                      <input
                        type="tel"
                        value={editedInvoice?.phone || ''}
                        onChange={(e) => handleFieldChange('phone', e.target.value)}
                        className="w-full mt-2 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                      />
                    ) : (
                      <p className="text-white font-semibold mt-2 text-sm">
                        {invoice.invoice?.phone || 'N/A'}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Messages */}
            {state.message && (
              <div
                className={`mx-8 mb-8 p-4 rounded-lg flex items-start gap-3 ${
                  state.messageType === 'success'
                    ? 'bg-emerald-500/10 border border-emerald-500/30'
                    : 'bg-red-500/10 border border-red-500/30'
                }`}
                role="alert"
              >
                {state.messageType === 'success' ? (
                  <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                )}
                <p
                  className={`text-sm font-medium ${
                    state.messageType === 'success' ? 'text-emerald-300' : 'text-red-300'
                  }`}
                >
                  {state.message}
                </p>
              </div>
            )}

            {/* Upload Another Button */}
            <div className="p-8 border-t border-slate-700">
              <button
                onClick={() => {
                  setInvoice(null);
                  setEditedInvoice(null);
                  setEditing(false);
                }}
                className="w-full py-3 px-4 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition-colors"
              >
                Upload Another Invoice
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Upload UI (unchanged)
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-xl mb-4 shadow-lg shadow-cyan-500/30">
            <Upload className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">
            Upload Invoice
          </h1>
          <p className="text-slate-400 text-sm">
            Securely upload your documents in seconds
          </p>
        </div>

        <div className="bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-700">
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`p-8 border-2 border-dashed transition-all duration-300 ${
              state.dragActive
                ? 'border-cyan-400 bg-cyan-500/10 scale-[1.02]'
                : 'border-slate-600 bg-slate-700/50 hover:border-cyan-400/50'
            }`}
          >
            <input
              ref={fileInputRef}
              id="fileInput"
              type="file"
              onChange={handleFileChange}
              accept=".pdf,.jpg,.jpeg,.png"
              className="hidden"
            />

            <label htmlFor="fileInput" className="cursor-pointer block text-center">
              <div className="flex justify-center mb-4">
                <div className="p-3 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-xl">
                  <Upload className="w-6 h-6 text-white" />
                </div>
              </div>
              <p className="text-white font-semibold mb-1">
                Click to upload or drag and drop
              </p>
              <p className="text-xs text-slate-400">
                PDF or Image (PNG, JPG) • Max 5MB
              </p>
            </label>

            {state.file && (
              <div className="mt-5 p-4 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-lg border border-cyan-500/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center flex-1 min-w-0">
                    <div className="p-2 bg-cyan-500/20 rounded-lg mr-3">
                      <Upload className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-white truncate">
                        {state.file.name}
                      </p>
                      <p className="text-xs text-slate-400">
                        {(state.file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <button onClick={handleClear} className="ml-2 p-1 hover:bg-red-500/20 rounded-lg transition-colors">
                    <X className="w-4 h-4 text-slate-400 hover:text-red-400" />
                  </button>
                </div>
              </div>
            )}
          </div>

          {state.uploadProgress > 0 && state.uploadProgress < 100 && (
            <div className="px-8 py-4 bg-slate-700/50 border-t border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-300">Uploading...</span>
                <span className="text-sm font-semibold text-cyan-400">
                  {Math.round(state.uploadProgress)}%
                </span>
              </div>
              <div className="w-full h-2 bg-slate-600 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full transition-all duration-300"
                  style={{ width: `${state.uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          <div className="p-8 flex gap-3">
            <button
              onClick={handleUpload}
              disabled={state.loading || !state.file}
              className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
                state.loading || !state.file
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-400 to-blue-500 text-white hover:shadow-lg hover:shadow-cyan-500/50'
              }`}
            >
              {state.loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  Upload
                </>
              )}
            </button>

            <button
              onClick={handleClear}
              disabled={state.loading}
              className={`py-3 px-4 rounded-lg font-semibold transition-all ${
                state.loading
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {state.message && (
            <div
              className={`mx-8 mb-8 p-4 rounded-lg flex items-start gap-3 ${
                state.messageType === 'success'
                  ? 'bg-emerald-500/10 border border-emerald-500/30'
                  : 'bg-red-500/10 border border-red-500/30'
              }`}
            >
              {state.messageType === 'success' ? (
                <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              )}
              <p
                className={`text-sm font-medium ${
                  state.messageType === 'success' ? 'text-emerald-300' : 'text-red-300'
                }`}
              >
                {state.message}
              </p>
            </div>
          )}
        </div>

        <p className="text-center text-xs text-slate-500 mt-6">
          Your files are encrypted and processed securely
        </p>
      </div>
    </div>
  );
}