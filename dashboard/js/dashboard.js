/*
  // This code would use the API instead of dummy data
  refreshData()
    .then(() => {
      try {
        hideLoading();
      } catch (e) {
        console.warn("Could not hide loading:", e);
      }
      createPriceChart();
      createImpactChart();
      createVolumeChart();
      createHeatmap();
    })
    .catch(error => {
      console.error("Error initializing dashboard:", error);
      try {
        hideLoading();
      } catch (e) {
        console.warn("Could not hide loading:", e);
      }
      showError("Failed to initialize dashboard. Using dummy data instead.");
      useDummyData();
    });
  */
  

  // Initialize the dashboard when the page is loaded
  document.addEventListener('DOMContentLoaded', function() {
    console.log("Document loaded");
    
    // Initialize dashboard with data
    initDashboard();
    
    // Add debug info
    console.log("Dashboard.js version: 1.0.0");
    console.log("API base URL:", API_BASE_URL);
    console.log("Tariff date:", TARIFF_DATE);
    
    // Check D3 version
    if (typeof d3 !== 'undefined') {
      console.log("D3 version:", d3.version);
    } else {
      console.error("D3.js not loaded!");
    }
  });