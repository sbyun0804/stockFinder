import React from 'react';
import { Box, Container, Drawer, Field, Heading, IconButton, Input, Stack, Text } from '@chakra-ui/react';
import { LuX } from 'react-icons/lu';

type SideSheetProps = {
  open: boolean;
  onOpenChange: (next: boolean) => void;
};

const SideSheet: React.FC<SideSheetProps> = ({ open, onOpenChange }) => {
  return (
    <Drawer.Root open={open} onOpenChange={(e) => onOpenChange(e.open)} placement={'start'}>
      <Drawer.Backdrop />
      <Drawer.Positioner>
        <Drawer.Content>
          <Drawer.CloseTrigger />
          <Drawer.Header>
            <Drawer.Title>Settings</Drawer.Title>
            <IconButton size={'xs'} variant={'outline'} onClick={() => onOpenChange(false)}>
              <LuX />
            </IconButton>
          </Drawer.Header>
          <Drawer.Body>
            <Stack direction={'column'}>
              <Box>
                <Field.Root>
                  <Field.Label>Ticker</Field.Label>
                  <Input placeholder="ex) AMZN" />
                </Field.Root>
              </Box>
              <Box>
                <Text>Reports</Text>
              </Box>
              <Box>
                <Text>Settings</Text>
              </Box>
            </Stack>
          </Drawer.Body>
          <Drawer.Footer></Drawer.Footer>
        </Drawer.Content>
      </Drawer.Positioner>
    </Drawer.Root>
  );
};

export default SideSheet;
