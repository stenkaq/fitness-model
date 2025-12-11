// Main JavaScript file for FitRec

// Initialize tooltips and other Bootstrap components
document.addEventListener("DOMContentLoaded", function () {
  // Initialize Bootstrap tooltips if any
  var tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]'),
  );
  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
});

// Auto-hide alerts after 5 seconds
setTimeout(function () {
  const alerts = document.querySelectorAll(".alert:not(.alert-permanent)");
  alerts.forEach(function (alert) {
    const bsAlert = new bootstrap.Alert(alert);
    bsAlert.close();
  });
}, 5000);

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute("href"));
    if (target) {
      target.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  });
});
