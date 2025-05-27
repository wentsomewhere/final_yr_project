import React from 'react';
import {
  Box,
  Grid,
  Typography,
  Paper,
  Divider,
} from '@mui/material';
import { styled } from '@mui/material/styles';

const ImageContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const StyledImage = styled('img')({
  maxWidth: '100%',
  maxHeight: '400px',
  objectFit: 'contain',
  marginBottom: '16px',
});

const TextContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginTop: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
}));

const ImageComparison = ({ originalImage, enhancedImage, ocrText }) => {
  return (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <ImageContainer elevation={2}>
            <Typography variant="h6" gutterBottom>
              Original Image
            </Typography>
            <StyledImage src={originalImage} alt="Original" />
          </ImageContainer>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <ImageContainer elevation={2}>
            <Typography variant="h6" gutterBottom>
              Enhanced Image
            </Typography>
            <StyledImage src={enhancedImage} alt="Enhanced" />
          </ImageContainer>
        </Grid>
      </Grid>

      <Divider sx={{ my: 3 }} />

      <TextContainer elevation={1}>
        <Typography variant="h6" gutterBottom>
          Extracted Text
        </Typography>
        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
          {ocrText || 'No text detected'}
        </Typography>
      </TextContainer>
    </Box>
  );
};

export default ImageComparison; 