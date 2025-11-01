import React, { useState, useEffect } from 'react';
import { Search, Filter, Download, Trash2, Eye, ChevronUp, ChevronDown, Loader, FileText, Calendar, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const API_BASE = 'http://localhost:3000/api/invoices';

const CATEGORIES = ['travel', 'meals', 'saas', 'office', 'utilities', 'healthcare', 'retail', 'education', 'entertainment', 'maintenance', 'other'];

const InvoicesList = () => {
  const [invoices, setInvoices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  
  
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [sortField, setSortField] = useState('invoice_date');
  const [sortOrder, setSortOrder] = useState('desc');
  
  // UI State
  const [selectedInvoices, setSelectedInvoices] = useState(new Set());
  const [showFilters, setShowFilters] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  useEffect(() => {
    fetchInvoices();
  }, [sortField, sortOrder]);

  const fetchInvoices = async () => {
  try {
    setLoading(true);
    setError(null);
    
    const sortParam = sortOrder === 'desc' ? `-${sortField}` : sortField;
    const response = await fetch(`${API_BASE}?sort=${sortParam}`);
    
    if (!response.ok) throw new Error('Failed to fetch invoices');
    
    const data = await response.json();
    
    // ✅ Use the correct data access pattern that worked in debug
    setInvoices(data.data || []);
    
  } catch (err) {
    console.error('Fetch error:', err);
    setError(err.message);
  } finally {
    setLoading(false);
  }
};

  const handleDelete = async (id) => {
    try {
      const response = await fetch(`${API_BASE}/${id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Failed to delete invoice');
      
      setInvoices(invoices.filter(inv => inv._id !== id));
      setDeleteConfirm(null);
    } catch (err) {
      alert('Failed to delete invoice: ' + err.message);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedInvoices.size === 0) return;
    
    if (!confirm(`Delete ${selectedInvoices.size} selected invoices?`)) return;
    
    try {
      await Promise.all(
        Array.from(selectedInvoices).map(id =>
          fetch(`${API_BASE}/${id}`, { method: 'DELETE' })
        )
      );
      
      setInvoices(invoices.filter(inv => !selectedInvoices.has(inv._id)));
      setSelectedInvoices(new Set());
    } catch (err) {
      alert('Failed to delete some invoices');
    }
  };

  const handleExport = () => {
    const selected = selectedInvoices.size > 0
      ? filteredInvoices.filter(inv => selectedInvoices.has(inv._id))
      : filteredInvoices;
    
    const csv = [
      ['Vendor', 'Invoice #', 'Date', 'Category', 'Subtotal', 'Tax', 'Total'].join(','),
      ...selected.map(inv => [
        inv.vendor,
        inv.invoice_number,
        inv.invoice_date ? new Date(inv.invoice_date).toLocaleDateString() : '',
        inv.category,
        inv.subtotal,
        inv.tax,
        inv.total
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `invoices-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  const toggleSelection = (id) => {
    const newSet = new Set(selectedInvoices);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedInvoices(newSet);
  };

  const toggleSelectAll = () => {
    if (selectedInvoices.size === filteredInvoices.length) {
      setSelectedInvoices(new Set());
    } else {
      setSelectedInvoices(new Set(filteredInvoices.map(inv => inv._id)));
    }
  };

  // Apply filters
  const filteredInvoices = invoices.filter(invoice => {
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const matchesSearch = 
        invoice.vendor?.toLowerCase().includes(query) ||
        invoice.invoice_number?.toLowerCase().includes(query) ||
        invoice.category?.toLowerCase().includes(query);
      if (!matchesSearch) return false;
    }
    
    // Category filter
    if (categoryFilter !== 'all' && invoice.category !== categoryFilter) {
      return false;
    }
    
    // Date range filter
    if (startDate && invoice.invoice_date) {
      if (new Date(invoice.invoice_date) < new Date(startDate)) return false;
    }
    if (endDate && invoice.invoice_date) {
      if (new Date(invoice.invoice_date) > new Date(endDate)) return false;
    }
    
    return true;
  });

  const clearFilters = () => {
    setSearchQuery('');
    setCategoryFilter('all');
    setStartDate('');
    setEndDate('');
  };

  const SortIcon = ({ field }) => {
    if (sortField !== field) return null;
    return sortOrder === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-400">Loading invoices...</p>
        </div>
      </div>
    );
  }

  if (error) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 max-w-md text-center">
        <p className="text-red-400 mb-4">{error}</p>
        <button
          onClick={fetchInvoices}
          className="px-6 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg font-semibold transition-colors"
        >
          Retry
        </button>
      </div>
    </div>
  );
}

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">Invoice History</h1>
          <p className="text-slate-400">Manage and search all your invoices</p>
        </div>

        {/* Search and Filter Bar */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 mb-6">
          <div className="flex flex-col lg:flex-row gap-4 mb-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                placeholder="Search by vendor, invoice #, or category..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-cyan-500"
              />
            </div>

            {/* Filter Toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`px-6 py-3 rounded-lg font-semibold flex items-center gap-2 transition-all ${
                showFilters
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              <Filter className="w-5 h-5" />
              Filters
            </button>
          </div>

          {/* Advanced Filters */}
          {showFilters && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-slate-700">
              <div>
                <label className="block text-xs text-slate-400 mb-2 uppercase tracking-wide">Category</label>
                <select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="all">All Categories</option>
                  {CATEGORIES.map(cat => (
                    <option key={cat} value={cat} className="capitalize">{cat}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-2 uppercase tracking-wide">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="block text-xs text-slate-400 mb-2 uppercase tracking-wide">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                />
              </div>
            </div>
          )}

          {/* Active Filters */}
          {(searchQuery || categoryFilter !== 'all' || startDate || endDate) && (
            <div className="flex items-center gap-2 mt-4 pt-4 border-t border-slate-700">
              <span className="text-sm text-slate-400">Active filters:</span>
              {searchQuery && (
                <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm">
                  Search: {searchQuery}
                </span>
              )}
              {categoryFilter !== 'all' && (
                <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm capitalize">
                  {categoryFilter}
                </span>
              )}
              {(startDate || endDate) && (
                <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm">
                  {startDate && `From ${new Date(startDate).toLocaleDateString()}`}
                  {startDate && endDate && ' - '}
                  {endDate && `To ${new Date(endDate).toLocaleDateString()}`}
                </span>
              )}
              <button
                onClick={clearFilters}
                className="ml-auto text-sm text-slate-400 hover:text-white flex items-center gap-1"
              >
                <X className="w-4 h-4" />
                Clear all
              </button>
            </div>
          )}
        </div>

        {/* Bulk Actions */}
        {selectedInvoices.size > 0 && (
          <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4 mb-6 flex items-center justify-between">
            <span className="text-cyan-400 font-semibold">
              {selectedInvoices.size} invoice{selectedInvoices.size !== 1 ? 's' : ''} selected
            </span>
            <div className="flex gap-3">
              <button
                onClick={handleExport}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold flex items-center gap-2 transition-colors"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
              <button
                onClick={handleBulkDelete}
                className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg font-semibold flex items-center gap-2 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
            </div>
          </div>
        )}

        {/* Table */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="overflow-x-auto">
            {filteredInvoices.length > 0 ? (
              <table className="w-full">
                <thead className="bg-slate-700/50">
                  <tr>
                    <th className="px-6 py-3 text-left">
                      <input
                        type="checkbox"
                        checked={selectedInvoices.size === filteredInvoices.length && filteredInvoices.length > 0}
                        onChange={toggleSelectAll}
                        className="w-4 h-4 rounded bg-slate-600 border-slate-500 text-cyan-500 focus:ring-cyan-500"
                      />
                    </th>
                    <th
                      onClick={() => handleSort('vendor')}
                      className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white"
                    >
                      <div className="flex items-center gap-1">
                        Vendor <SortIcon field="vendor" />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort('invoice_number')}
                      className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white"
                    >
                      <div className="flex items-center gap-1">
                        Invoice # <SortIcon field="invoice_number" />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort('invoice_date')}
                      className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white"
                    >
                      <div className="flex items-center gap-1">
                        Date <SortIcon field="invoice_date" />
                      </div>
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                      Category
                    </th>
                    <th
                      onClick={() => handleSort('total')}
                      className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white"
                    >
                      <div className="flex items-center gap-1">
                        Amount <SortIcon field="total" />
                      </div>
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                  {filteredInvoices.map((invoice) => (
                    <tr key={invoice._id} className="hover:bg-slate-700/30 transition-colors">
                      <td className="px-6 py-4">
                        <input
                          type="checkbox"
                          checked={selectedInvoices.has(invoice._id)}
                          onChange={() => toggleSelection(invoice._id)}
                          className="w-4 h-4 rounded bg-slate-600 border-slate-500 text-cyan-500 focus:ring-cyan-500"
                        />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-white font-medium">
                        {invoice.vendor || 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                        {invoice.invoice_number || 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                        {invoice.invoice_date ? new Date(invoice.invoice_date).toLocaleDateString() : 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="px-2 py-1 text-xs font-semibold rounded-full bg-cyan-500/20 text-cyan-400 capitalize">
                          {invoice.category || 'other'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-emerald-400">
                        ${(invoice.total || 0).toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <div className="flex gap-2">
                          <button
                            onClick={() => navigate(`/invoice/${invoice._id}`)}
                            className="p-2 hover:bg-slate-600 rounded-lg transition-colors text-cyan-400"
                            title="View Details"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setDeleteConfirm(invoice._id)}
                            className="p-2 hover:bg-red-500/20 rounded-lg transition-colors text-red-400"
                            title="Delete"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="p-12 text-center">
                <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400 text-lg mb-2">No invoices found</p>
                <p className="text-slate-500 text-sm">Try adjusting your filters or upload a new invoice</p>
              </div>
            )}
          </div>
        </div>

        {/* Results Count */}
        <div className="mt-4 text-center text-slate-400 text-sm">
          Showing {filteredInvoices.length} of {invoices.length} invoices
        </div>

        {/* Delete Confirmation Modal */}
        {deleteConfirm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 max-w-md w-full">
              <h3 className="text-xl font-bold text-white mb-4">Confirm Deletion</h3>
              <p className="text-slate-300 mb-6">
                Are you sure you want to delete this invoice? This action cannot be undone.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => handleDelete(deleteConfirm)}
                  className="flex-1 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg font-semibold transition-colors"
                >
                  Delete
                </button>
                <button
                  onClick={() => setDeleteConfirm(null)}
                  className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InvoicesList;