import { Grid, GridItem, Heading, Link, Text } from '@chakra-ui/react';
import React from 'react';
import FinancialTable from './FinancialTable';
import styles from './company-info.module.scss';

const CompanyInfo = () => {
  return (
    <div className={styles.container}>
      <Grid templateColumns="repeat(6, 1fr)" gap={'2rem'}>
        <GridItem colSpan={2}>
          <Heading marginBottom={'1rem'}>Google(official name)</Heading>
          <div className={styles.metrics}>
            <Heading>Latest Price: 207.48</Heading>
            <FinancialTable />
          </div>
        </GridItem>
        <GridItem colSpan={4}>
          <div className={styles.description}>
            <Heading marginBottom={'0.5rem'}>Description </Heading>
            <Text marginBottom={'1rem'}>
              Alphabet Inc. provides various products and platforms in the United States, Europe, the Middle East,
              Africa, the Asia-Pacific, Canada, and Latin America. It operates through Google Services, Google Cloud,
              and Other Bets segments. The Google Services segment offers products and services, including ads, Android,
              Chrome, hardware, Gmail, Google Drive, Google Maps, Google Photos, Google Play, Search, and YouTube. It is
              also involved in the sale of apps and in-app purchases and digital content in the Google Play store; and
              Fitbit wearable devices, Google Nest home products, Pixel phones, and other devices, as well as in the
              provision of YouTube non-advertising services. The Google Cloud segment offers infrastructure, platform,
              and other services; Google Workspace that include cloud-based collaboration tools for enterprises, such as
              Gmail, Docs, Drive, Calendar, and Meet; and other services for enterprise customers. The Other Bets
              segment sells health technology and internet services. The company was founded in 1998 and is
              headquartered in Mountain View, California.
            </Text>
            <div className={styles.descriptionText}>
              <Text fontWeight={600}>Website:</Text>
              <Link href="https://www.abc.xyz">https://www.abc.xyz</Link>
            </div>
            <div className={styles.descriptionText}>
              <Text fontWeight={600}>Market Cap (provider or fallback):</Text>
              <Text> 2,513,225,889,772</Text>
            </div>
            <div className={styles.descriptionText}>
              <Text fontWeight={600}>Shares Out:</Text>
              <Text> 12,113,099,776</Text>
            </div>
            <div className={styles.descriptionText}>
              <Text fontWeight={600}>Sector/Industry:</Text>
              <Text> Communication Services / Internet Content & Information</Text>
            </div>
          </div>
        </GridItem>
      </Grid>
    </div>
  );
};

export default CompanyInfo;
