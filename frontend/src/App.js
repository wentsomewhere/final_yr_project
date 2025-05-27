import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Paper,
  Grid,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ImageUploader from './components/ImageUploader';
import ImageComparison from './components/ImageComparison';
import MetricsDisplay from './components/MetricsDisplay';
import Navbar from './components/Navbar';

// Create a custom theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
});

// Styled components
const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  margin: theme.spacing(2, 0),
  borderRadius: theme.spacing(1),
}));

const App = () => {
  const [originalImage, setOriginalImage] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [ocrText, setOcrText] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/enhance', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to enhance image');
      }

      const data = await response.json();
      
      // Convert base64 image data to URL
      const enhancedImageUrl = `data:image/png;base64,${data.enhanced_image}`;
      
      setOriginalImage(URL.createObjectURL(file));
      setEnhancedImage(enhancedImageUrl);
      setOcrText(data.ocr_text);
      setMetrics(data.metrics);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <Navbar />
        <Container maxWidth="lg">
          <Box sx={{ my: 4 }}>
            <Typography variant="h1" component="h1" gutterBottom align="center">
              SRR-GAN
            </Typography>
            <Typography variant="h2" component="h2" gutterBottom align="center" color="textSecondary">
              Super-Resolution based Recognition with GAN
            </Typography>

            <StyledPaper elevation={3}>
              <ImageUploader onUpload={handleImageUpload} loading={loading} />
            </StyledPaper>

            {error && (
              <Typography color="error" align="center" sx={{ mt: 2 }}>
                {error}
              </Typography>
            )}

            {originalImage && enhancedImage && (
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <StyledPaper elevation={3}>
                    <ImageComparison
                      originalImage={originalImage}
                      enhancedImage={enhancedImage}
                      ocrText={ocrText}
                    />
                  </StyledPaper>
                </Grid>

                {metrics && (
                  <Grid item xs={12}>
                    <StyledPaper elevation={3}>
                      <MetricsDisplay metrics={metrics} />
                    </StyledPaper>
                  </Grid>
                )}
              </Grid>
            )}
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default App; 