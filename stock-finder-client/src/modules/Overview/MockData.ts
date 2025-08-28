// src/mock/financialMetrics.ts

export type FinancialItem = {
  id: string;
  name: string; // "Metric"
  value: number; // raw number (useful for math/sorting)
  display: string; // formatted for the UI
};

export const items: FinancialItem[] = [
  { id: 'revenue_ttm', name: 'Revenue (TTM)', value: 371399000000, display: '371,399,000,000' },
  { id: 'net_income_ttm', name: 'Net Income (TTM)', value: 115573000000, display: '115,573,000,000' },
  { id: 'cogs_ttm', name: 'COGS (TTM)', value: 152487000000, display: '152,487,000,000' },
  { id: 'total_debt', name: 'Total Debt', value: 23607000000, display: '23,607,000,000' },
  { id: 'total_equity', name: 'Total Equity', value: 478446000000, display: '478,446,000,000' },
  { id: 'pe', name: 'P/E', value: 21.746, display: '21.746' },
  { id: 'psr', name: 'PSR', value: 6.767, display: '6.767' },
  { id: 'pb', name: 'P/B', value: 5.253, display: '5.253' },
  { id: 'debt_equity', name: 'Debt/Equity', value: 0.049, display: '0.049' },
  { id: 'eps_ttm', name: 'EPS (TTM)', value: 9.541, display: '9.541' },
  { id: 'dividend_ttm', name: 'Dividend (TTM)', value: 1.02, display: '1.020' },
];
