import React, { useState, useEffect } from 'react';
import { Upload, DollarSign, FileText, TrendingUp, Calendar, ArrowUpRight, Loader, Eye } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useNavigate } from "react-router-dom";

const API_BASE = 'http://localhost:3000/api/invoices';


const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [statistics, setStatistics] = useState(null);
  const [recentInvoices, setRecentInvoices] = useState([]);
  const [error, setError] = useState(null);

  const navigate = useNavigate();

  useEffect(() => {
    fetchDashboardData();
  }, []);


const fetchDashboardData = async () => {
  try {
    setLoading(true);
    setError(null);

    console.log('Fetching from:', `${API_BASE}/statistics`);
    
    const [statsRes, invoicesRes] = await Promise.all([
      fetch(`${API_BASE}/statistics`),
      fetch(`${API_BASE}?limit=5&sort=-createdAt`)
    ]);

    

    const statsData = await statsRes.json();
    const invoicesData = await invoicesRes.json();

    
    setStatistics(statsData.data);
    setRecentInvoices(invoicesData.data || []);

    
  } catch (err) {
    console.error('Dashboard fetch error:', err);
    setError(err.message);
  } finally {
    setLoading(false);
  }
};

// Prepare chart data from statistics
const prepareChartData = () => {
  if (!statistics) return { monthly: [], category: [] };

  console.log('Statistics for charts:', statistics);

  
  const monthly = statistics.recentInvoices?.map(invoice => ({
    month: new Date(invoice.createdAt).toLocaleDateString('en-US', { month: 'short' }),
    amount: 1 
  })) || [];

  // Show invoice counts by category 
  const category = statistics.byCategory?.map(item => ({
    name: item._id,
    value: item.count, 
    count: item.count
  })) || [];

  

  return { monthly, category };
};

  const { monthly, category } = prepareChartData();

  const COLORS = ['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444', '#6366f1', '#14b8a6', '#f97316'];

  

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 max-w-md">
          <p className="text-red-400 text-center">{error}</p>
          <button
            onClick={fetchDashboardData}
            className="mt-4 w-full py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
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
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Dashboard</h1>
          <p className="text-slate-400">Overview of your invoice management</p>
        </div>

       {/* Quick Stats */}
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
  {/* Total Invoices - This one works */}
  <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 hover:border-cyan-500/50 transition-all">
    <div className="flex items-center justify-between mb-4">
      <div className="p-3 bg-cyan-500/20 rounded-lg">
        <FileText className="w-6 h-6 text-cyan-400" />
      </div>
      <TrendingUp className="w-5 h-5 text-emerald-400" />
    </div>
    <p className="text-slate-400 text-sm mb-1">Total Invoices</p>
    <p className="text-3xl font-bold text-white">{statistics?.overview?.totalInvoices || 0}</p>
  </div>

  {/* Categories Count */}
  <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 hover:border-emerald-500/50 transition-all">
    <div className="flex items-center justify-between mb-4">
      <div className="p-3 bg-emerald-500/20 rounded-lg">
        <DollarSign className="w-6 h-6 text-emerald-400" />
      </div>
    </div>
    <p className="text-slate-400 text-sm mb-1">Categories</p>
    <p className="text-3xl font-bold text-white">
      {statistics?.byCategory?.length || 0}
    </p>
  </div>

  {/* This Month's Invoices */}
  <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 hover:border-blue-500/50 transition-all">
    <div className="flex items-center justify-between mb-4">
      <div className="p-3 bg-blue-500/20 rounded-lg">
        <Calendar className="w-6 h-6 text-blue-400" />
      </div>
    </div>
    <p className="text-slate-400 text-sm mb-1">This Month</p>
    <p className="text-3xl font-bold text-white">
      {statistics?.recentInvoices?.filter(inv => {
        const invoiceDate = new Date(inv.createdAt);
        const now = new Date();
        return invoiceDate.getMonth() === now.getMonth() && 
               invoiceDate.getFullYear() === now.getFullYear();
      }).length || 0}
    </p>
  </div>

  {/* Most Common Category */}
  <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 hover:border-purple-500/50 transition-all">
    <div className="flex items-center justify-between mb-4">
      <div className="p-3 bg-purple-500/20 rounded-lg">
        <TrendingUp className="w-6 h-6 text-purple-400" />
      </div>
    </div>
    <p className="text-slate-400 text-sm mb-1">Top Category</p>
    <p className="text-3xl font-bold text-white capitalize">
      {statistics?.byCategory?.[0]?._id || 'N/A'}
    </p>
  </div>
</div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Monthly Spending Chart */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold text-white mb-6">Monthly Spending Trend</h2>
            {monthly.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={monthly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="month" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Line type="monotone" dataKey="amount" stroke="#06b6d4" strokeWidth={2} dot={{ fill: '#06b6d4', r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-slate-500">
                No monthly data available
              </div>
            )}
          </div>

          {/* Category Breakdown Chart */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold text-white mb-6">Spending by Category</h2>
            {category.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={category}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {category.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-slate-500">
                No category data available
              </div>
            )}
          </div>
        </div>

        {/* Recent Invoices */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="p-6 border-b border-slate-700 flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Recent Invoices</h2>
            <button
              onClick={() => navigate('/invoices')}
              className="text-cyan-400 hover:text-cyan-300 text-sm font-semibold flex items-center gap-1"
            >
              View All <ArrowUpRight className="w-4 h-4" />
            </button>
          </div>

          {recentInvoices.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-700/50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Vendor</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Invoice #</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Date</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Category</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Amount</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                  {recentInvoices.map((invoice) => (
                    <tr key={invoice._id} className="hover:bg-slate-700/30 transition-colors">
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
                        <button
                          onClick={() => navigate(`/invoice/${invoice._id}`)}
                          className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
                        >
                          <Eye className="w-4 h-4" />
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-12 text-center">
              <FileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-500 mb-4">No invoices yet</p>
              <button
                onClick={() => navigate(`/upload`)}
                className="px-6 py-2 bg-gradient-to-r from-cyan-400 to-blue-500 text-white rounded-lg font-semibold hover:shadow-lg hover:shadow-cyan-500/50 transition-all"
              >
                <Upload className="w-4 h-4 inline mr-2" />
                Upload Your First Invoice
              </button>
            </div>
          )}
        </div>

        {/* Quick Upload Button (Floating) */}
        <button
          onClick={() => navigate(`/upload`)}
          className="fixed bottom-8 right-8 p-4 bg-gradient-to-r from-cyan-400 to-blue-500 text-white rounded-full shadow-2xl hover:shadow-cyan-500/50 transition-all hover:scale-110"
          title="Quick Upload"
        >
          <Upload className="w-6 h-6" />
        </button>
      </div>
    </div>
  );
};

export default Dashboard;