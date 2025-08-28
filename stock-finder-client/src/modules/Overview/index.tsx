import { Text } from '@chakra-ui/react';
import CompanyInfo from './CompanyInfo';
import styles from './index.module.scss';

const Overview = () => {
  const renderFooter = () => {
    return (
      <div style={{ background: 'red' }}>
        <Text>
          Prices/profiles/peers/news from FMP; TTM financials use FMP when available, else Yahoo Finance fallback.
          Handles negative earnings (no P/E).
        </Text>
      </div>
    );
  };
  return (
    <div className={styles.container}>
      <CompanyInfo />
      {renderFooter()}
    </div>
  );
};

export default Overview;
