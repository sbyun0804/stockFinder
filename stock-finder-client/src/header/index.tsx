import React, { useState } from 'react';
import styles from './index.module.scss';
import { Button, Group, Input, InputGroup, Link } from '@chakra-ui/react';
import { LuChartNoAxesCombined, LuSearch } from 'react-icons/lu';

type HeaderProps = {
  onOpenDrawer: () => void;
};
const Header: React.FC<HeaderProps> = ({ onOpenDrawer }) => {
  const [value, setValue] = useState('GOOGL');

  return (
    <div className={styles.container}>
      <div className={styles.left}>
        {/* <IconButton size={'xs'} variant={'outline'} onClick={onOpenDrawer}>
          <LuAlignJustify />
        </IconButton> */}
        <LuChartNoAxesCombined />

        <Link href="./">Overview</Link>
      </div>
      <div>
        <Group>
          <InputGroup startElement={<LuSearch />}>
            <Input
              placeholder="ex) AMZN"
              variant="subtle"
              value={value}
              onChange={(e) => {
                setValue(e.currentTarget.value);
              }}
            />
          </InputGroup>
          <Button bg="bg.subtle" variant="outline">
            Search
          </Button>
        </Group>
      </div>
    </div>
  );
};

export default Header;
