// Smooth scrolling and intersection observer
const hero = document.getElementById('heroSection');
const mainApp = document.getElementById('mainApp');

// Intersection Observer for smooth transitions
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.target === mainApp && entry.isIntersecting) {
      mainApp.classList.add('visible');
    }
  });
}, { threshold: 0.1 });

observer.observe(mainApp);

// Auto-scroll to main app on any interaction with hero
hero.addEventListener('click', () => {
  mainApp.scrollIntoView({ behavior: 'smooth' });
});

// File upload handling with visual feedback
const fileInputs = document.querySelectorAll('input[type="file"]');
fileInputs.forEach(input => {
  input.addEventListener('change', function(e) {
    const label = this.closest('.upload-label');
    const fileName = this.files[0]?.name;
    
    if (fileName) {
      const hint = label.querySelector('.upload-hint');
      hint.textContent = fileName;
      hint.style.color = 'var(--accent-color)';
      label.closest('.upload-card').style.borderColor = 'var(--accent-color)';
    }
  });
});

// Form submission with enhanced UX
const form = document.getElementById('analysisForm');
const loadingState = document.getElementById('loadingState');
const resultSection = document.getElementById('resultSection');
const resultContent = document.getElementById('resultContent');

form.addEventListener('submit', async function(e) {
  e.preventDefault();
  
  // Show loading state
  loadingState.style.display = 'block';
  resultSection.style.display = 'none';
  
  // Scroll to loading state
  loadingState.scrollIntoView({ behavior: 'smooth', block: 'center' });
  
  const formData = new FormData(form);
  
  // File handling logic
  const videoFile = document.getElementById('videoUpload').files[0];
  const audioFile = document.getElementById('audioUpload').files[0];
  
  if (videoFile) {
    formData.set('file', videoFile);
  } else if (audioFile) {
    formData.set('file', audioFile);
  } else {
    formData.delete('file');
  }
  
  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Display results
    resultContent.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    resultSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
      resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
    
  } catch (error) {
    resultContent.innerHTML = `
      <div style="color: var(--error); padding: 1rem; background: rgba(239, 68, 68, 0.1); border-radius: 0.5rem;">
        <strong>Error:</strong> ${error.message}
      </div>
    `;
    resultSection.style.display = 'block';
    console.error('Analysis failed:', error);
  } finally {
    loadingState.style.display = 'none';
  }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  // Enter key on hero section scrolls to main app
  if (e.key === 'Enter' && document.activeElement === document.body) {
    const heroRect = hero.getBoundingClientRect();
    if (heroRect.top >= 0 && heroRect.bottom <= window.innerHeight) {
      mainApp.scrollIntoView({ behavior: 'smooth' });
    }
  }
});
