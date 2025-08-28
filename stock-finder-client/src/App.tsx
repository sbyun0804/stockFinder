import Header from './header';
import styles from './App.module.scss';
import { useDisclosure } from '@chakra-ui/react';
import SideSheet from './modules/SideSheet';
import Overview from './modules/Overview';

function App() {
  const { open, onOpen, onClose } = useDisclosure();

  return (
    <div className={styles.app}>
      <Header onOpenDrawer={onOpen} />
      <div className={styles.container}>
        <SideSheet open={open} onOpenChange={(next) => (next ? onOpen() : onClose())} />
        <Overview />
      </div>
    </div>
  );
}

export default App;
