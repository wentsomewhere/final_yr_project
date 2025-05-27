import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  GitHub as GitHubIcon,
  Info as InfoIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';
import { useState } from 'react';

const Navbar = () => {
  const [anchorEl, setAnchorEl] = useState(null);

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <IconButton
          size="large"
          edge="start"
          color="inherit"
          aria-label="menu"
          sx={{ mr: 2 }}
          onClick={handleMenu}
        >
          <MenuIcon />
        </IconButton>

        <Menu
          id="menu-appbar"
          anchorEl={anchorEl}
          anchorOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
          keepMounted
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
          open={Boolean(anchorEl)}
          onClose={handleClose}
        >
          <MenuItem onClick={handleClose}>About</MenuItem>
          <MenuItem onClick={handleClose}>Documentation</MenuItem>
          <MenuItem onClick={handleClose}>Contact</MenuItem>
        </Menu>

        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          SRR-GAN
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton
            color="inherit"
            aria-label="github"
            href="https://github.com/yourusername/SRR-GAN"
            target="_blank"
            rel="noopener noreferrer"
          >
            <GitHubIcon />
          </IconButton>

          <IconButton
            color="inherit"
            aria-label="info"
            sx={{ ml: 1 }}
          >
            <InfoIcon />
          </IconButton>

          <Button color="inherit" sx={{ ml: 2 }}>
            API Docs
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 