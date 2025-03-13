// visualizations.js - Adjusted to work with dashboard.js
// This file contains D3.js visualization functions for price trends, impact, and volume charts

// Create a namespace to avoid conflicts
const TariffVisualizations = (function() {
  // Define private variables within the scope
  const API_BASE_URL = 'http://localhost:5000/api';
  const margin = {top: 40, right: 80, bottom: 60, left: 60};
  
  // Fixed tariff event date
  const TARIFF_DATE = "2025-03-05";
  
  // Check if dashboard.js is loaded and use its data if available
  function useDataCacheIfAvailable() {
    return typeof dataCache !== 'undefined' && dataCache;
  }
  
  // Create price trend chart visualization
  function createPriceTrendChart(eventId, sector, timeWindow) {
    // Skip if dashboard.js is handling this
    if (useDataCacheIfAvailable()) {
      console.log("Price trend chart handled by dashboard.js");
      return;
    }
    
    // Clear previous chart
    d3.select("#price-chart").html("");
    
    // Set dimensions
    const width = document.getElementById('price-chart').clientWidth - margin.left - margin.right || 600;
    const height = 400 - margin.top - margin.bottom;
    
    // Create SVG
    const svg = d3.select("#price-chart")
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Loading indicator
    svg.append("text")
      .attr("x", width/2)
      .attr("y", height/2)
      .attr("text-anchor", "middle")
      .text("Loading price data...");
    
    console.log(`Fetching price data with: Event=${eventId}, Sector=${sector}, TimeWindow=${timeWindow}`);
    
    // Fetch data from API - use the new price-data endpoint with time window parameter
    const url = `${API_BASE_URL}/price-data?event_id=${eventId}&sector=${sector}&days=${timeWindow}`;
    
    d3.json(url).then(data => {
      if (!data || !data.series || data.series.length === 0) {
        svg.select("text").text("No price data available");
        return;
      }
      
      // Parse dates
      const parseDate = d3.timeParse("%Y-%m-%d");
      
      // Get event date
      const eventDate = parseDate(data.eventDate);
      
      // Define initial tariff date - FIXED to March 5, 2025
      const initialTariffDate = parseDate(TARIFF_DATE);
      
      // Clear loading text
      svg.selectAll("text").remove();
      
      // Create scales
      const x = d3.scaleTime()
        .domain(d3.extent(data.dates.map(d => parseDate(d))))
        .range([0, width]);
        
      const y = d3.scaleLinear()
        .domain([
          d3.min(data.series, s => d3.min(s.data, d => d.price)) * 0.98,
          d3.max(data.series, s => d3.max(s.data, d => d.price)) * 1.02
        ])
        .range([height, 0]);
      
      // Add x-axis
      svg.append("g")
        .attr("class", "x axis")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat("%b %d")));
      
      // Add y-axis
      svg.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(y).ticks(5));
      
      // Add y-axis label
      svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -margin.left + 20)
        .attr("x", -height / 2)
        .attr("text-anchor", "middle")
        .text("Normalized Price (Base=100)");
      
      // Create line generator
      const line = d3.line()
        .x(d => x(parseDate(d.date)))
        .y(d => y(d.price))
        .curve(d3.curveMonotoneX);
      
      // Add lines for each symbol
      data.series.forEach(series => {
        const companyInfo = data.companies.find(c => c.symbol === series.symbol);
        
        svg.append("path")
          .datum(series.data)
          .attr("class", `line ${companyInfo.sector} ${series.symbol}`)
          .attr("d", line)
          .attr("fill", "none")
          .attr("stroke", companyInfo.color)
          .attr("stroke-width", 2)
          .attr("opacity", 0.8);
      });
      
      // Add initial tariff date line
      svg.append("line")
        .attr("class", "initial-tariff-line")
        .attr("x1", x(initialTariffDate))
        .attr("x2", x(initialTariffDate))
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "blue")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5");
      
      // Add initial tariff date text
      svg.append("text")
        .attr("x", x(initialTariffDate) + 10)
        .attr("y", 40)
        .attr("fill", "blue")
        .text("Initial Tariff Date");
      
      // Add legend
      const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 100}, 20)`);
      
      data.companies.forEach((company, i) => {
        const legendRow = legend.append("g")
          .attr("transform", `translate(0, ${i * 20})`)
          .attr("class", `legend-item ${company.symbol}`)
          .style("cursor", "pointer")
          .on("mouseover", function() {
            // Highlight this line
            svg.select(`.line.${company.symbol}`)
              .attr("stroke-width", 4)
              .attr("opacity", 1);
              
            // Fade other lines
            svg.selectAll(`.line:not(.${company.symbol})`)
              .attr("opacity", 0.2);
              
            // Highlight legend
            d3.select(this).select("text")
              .attr("font-weight", "bold");
          })
          .on("mouseout", function() {
            // Reset lines
            svg.selectAll(".line")
              .attr("stroke-width", 2)
              .attr("opacity", 0.8);
              
            // Reset legend
            d3.select(this).select("text")
              .attr("font-weight", "normal");
          })
          .on("click", function() {
            // Toggle line visibility
            const line = svg.select(`.line.${company.symbol}`);
            const currentOpacity = line.attr("opacity");
            const newOpacity = currentOpacity == 0.8 ? 0 : 0.8;
            
            // Update line
            line.attr("opacity", newOpacity);
            
            // Update legend item
            d3.select(this).style("opacity", newOpacity ? 1 : 0.5);
          });
        
        legendRow.append("rect")
          .attr("width", 10)
          .attr("height", 10)
          .attr("fill", company.color);
        
        legendRow.append("text")
          .attr("x", 15)
          .attr("y", 10)
          .attr("text-anchor", "start")
          .attr("font-size", "12px")
          .text(company.symbol);
      });
      
      // Add interactive hover for date exploration
      const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style("position", "absolute")
        .style("background", "white")
        .style("padding", "10px")
        .style("border", "1px solid #ddd")
        .style("border-radius", "4px")
        .style("pointer-events", "none")
        .style("z-index", "1000");
      
      // Add hover line
      const hoverLine = svg.append("line")
        .attr("class", "hover-line")
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "#999")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "5,5")
        .style("opacity", 0);
      
      // Add overlay for mouse tracking
      svg.append("rect")
        .attr("class", "overlay")
        .attr("width", width)
        .attr("height", height)
        .style("fill", "none")
        .style("pointer-events", "all")
        .on("mousemove", function(event) {
          const [mouseX] = d3.pointer(event);
          const date = x.invert(mouseX);
          
          // Find closest date in data
          const bisectDate = d3.bisector(d => parseDate(d)).left;
          const index = bisectDate(data.dates, date);
          const closestDate = data.dates[index];
          
          if (closestDate) {
            // Update hover line
            hoverLine
              .attr("x1", x(parseDate(closestDate)))
              .attr("x2", x(parseDate(closestDate)))
              .style("opacity", 1);
            
            // Create tooltip content
            let tooltipContent = `<strong>${closestDate}</strong><br>`;
            
            data.series.forEach(series => {
              const point = series.data.find(d => d.date === closestDate);
              if (point) {
                const company = data.companies.find(c => c.symbol === series.symbol);
                tooltipContent += `<span style="color:${company.color}">${series.symbol}</span>: ${point.price.toFixed(2)} (${point.percentChange.toFixed(2)}%)<br>`;
              }
            });
            
            // Show tooltip
            tooltip
              .html(tooltipContent)
              .style("left", (event.pageX + 15) + "px")
              .style("top", (event.pageY - 30) + "px")
              .style("opacity", 1);
          }
        })
        .on("mouseout", function() {
          hoverLine.style("opacity", 0);
          tooltip.style("opacity", 0);
        });
    })
    .catch(error => {
      console.error("Error loading price data:", error);
      svg.select("text").text("Error loading price data");
    });
  }
  
  // Impact Chart - just skeleton, defer to dashboard.js if available
  function createImpactChart(eventId, sector) {
    if (useDataCacheIfAvailable()) {
      console.log("Impact chart handled by dashboard.js");
      return;
    }
    
    // Implement the original impact chart code here
  }
  
  // Volume Chart - just skeleton, defer to dashboard.js if available
  function createVolumeChart(eventId, sector, timeWindow) {
    if (useDataCacheIfAvailable()) {
      console.log("Volume chart handled by dashboard.js");
      return;
    }
    
    // Implement the original volume chart code here
  }
  
  // Initialize visualization only if dashboard.js isn't handling it
  function initVisualizations() {
    if (useDataCacheIfAvailable()) {
      console.log("Visualizations handled by dashboard.js");
      return;
    }
    
    // Get selected values
    const eventId = document.getElementById('event-selector').value || '10';
    const sector = document.getElementById('sector-selector').value || 'Lumber_Companies';
    const timeWindow = document.getElementById('time-window').value || '10';
    
    console.log("Initializing visualizations with:", { eventId, sector, timeWindow });
    
    // Create charts with all parameters including time window
    createPriceTrendChart(eventId, sector, timeWindow);
    createImpactChart(eventId, sector);
    createVolumeChart(eventId, sector, timeWindow);
  }
  
  // Expose public methods
  return {
    createPriceTrendChart: createPriceTrendChart,
    createImpactChart: createImpactChart,
    createVolumeChart: createVolumeChart,
    initVisualizations: initVisualizations,
    TARIFF_DATE: TARIFF_DATE
  };
})();

// Only add event listener if dashboard.js hasn't already added one
if (typeof dataCache === 'undefined') {
  document.addEventListener('DOMContentLoaded', function() {
    console.log("Initializing visualizations from visualizations.js");
    TariffVisualizations.initVisualizations();
    
    // Add event listener for update button only if dashboard.js isn't handling it
    const updateButton = document.getElementById('update-button');
    if (updateButton && !updateButton._hasEventListener) {
      updateButton.addEventListener('click', function() {
        console.log("Update button clicked");
        
        // Get current values from all selectors
        const eventId = document.getElementById('event-selector').value;
        const sector = document.getElementById('sector-selector').value;
        const timeWindow = document.getElementById('time-window').value;
        
        console.log(`Updating with: Event=${eventId}, Sector=${sector}, TimeWindow=${timeWindow}`);
        
        // Make sure these values are passed to your chart creation functions
        TariffVisualizations.createPriceTrendChart(eventId, sector, timeWindow);
        TariffVisualizations.createImpactChart(eventId, sector);
        TariffVisualizations.createVolumeChart(eventId, sector, timeWindow);
      });
      
      // Mark the button as having an event listener to avoid duplicates
      updateButton._hasEventListener = true;
    }
  });
}