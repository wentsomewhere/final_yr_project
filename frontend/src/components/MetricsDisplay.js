import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const MetricCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  height: '100%',
}));

const MetricsDisplay = ({ metrics }) => {
  // Prepare data for the chart
  const chartData = {
    labels: ['PSNR', 'SSIM', 'OCR Accuracy'],
    datasets: [
      {
        label: 'Performance Metrics',
        data: [
          metrics.psnr,
          metrics.ssim,
          metrics.accuracy,
        ],
        backgroundColor: [
          'rgba(54, 162, 235, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Metrics',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Performance Metrics
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <MetricCard elevation={2}>
            <Typography variant="h6" color="primary" gutterBottom>
              PSNR
            </Typography>
            <Typography variant="h4">
              {metrics.psnr.toFixed(2)} dB
            </Typography>
          </MetricCard>
        </Grid>

        <Grid item xs={12} md={4}>
          <MetricCard elevation={2}>
            <Typography variant="h6" color="primary" gutterBottom>
              SSIM
            </Typography>
            <Typography variant="h4">
              {metrics.ssim.toFixed(3)}
            </Typography>
          </MetricCard>
        </Grid>

        <Grid item xs={12} md={4}>
          <MetricCard elevation={2}>
            <Typography variant="h6" color="primary" gutterBottom>
              OCR Accuracy
            </Typography>
            <Typography variant="h4">
              {(metrics.accuracy * 100).toFixed(1)}%
            </Typography>
          </MetricCard>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Bar data={chartData} options={chartOptions} />
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Additional Metrics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body1">
                  Character Error Rate (CER): {(metrics.cer * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1">
                  Word Error Rate (WER): {(metrics.wer * 100).toFixed(1)}%
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MetricsDisplay; 